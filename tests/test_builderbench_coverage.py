from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import json

from empowerment_warp.core.rollout import (
    COVERAGE_TRACKS, _mark_coverage, coverage_counts, initial_coverage_grids,
)
from empowerment_warp.utils.evaluation_metrics import builderbench_state_coverage
from empowerment_warp.training.sac.progress import progress_wandb_row


def test_xyz_tracks_distinguish_lifting_and_preserve_pooled_xy() -> None:
    specs = COVERAGE_TRACKS["objects"]
    blocks = np.full((2, 2, 3, 3), 0.025, dtype=np.float32)
    blocks[1, :, 0, 2] = 0.125  # Only block 0 changes height.
    gripper = np.full((2, 2, 3), -0.025, dtype=np.float32)
    gripper[1, :, 2] = 0.075
    grids = initial_coverage_grids(specs)
    mark = jax.jit(lambda grid, block, hand, phase: _mark_coverage(
        grid, specs, (block[..., :2], block[:, 0], block[:, 1], block[:, 2], hand), phase,
    ))
    for step in range(2):
        grids = mark(grids, jnp.asarray(blocks[step]), jnp.asarray(gripper[step]),
                     jnp.full((2,), step == 1))
    training = coverage_counts(grids, specs)
    evaluation = builderbench_state_coverage(
        blocks, gripper, np.ones((2, 2), dtype=bool), onset=1,
    )
    for name, expected in (("cube_xy", 1), ("block_0_xyz", 2), ("block_1_xyz", 1),
                           ("block_2_xyz", 1), ("gripper_xyz", 2)):
        assert training[f"exploration/cells_visited_{name}_total"] == expected
        assert evaluation[name]["cells"] == expected
        for phase in ("potential_phase", "skill_phase"):
            assert training[f"exploration/cells_visited_{name}_{phase}"] == 1
            assert evaluation[name][f"cells_{phase}"] == 1
        assert "coverage_fraction" not in evaluation[name]


def test_dead_lanes_do_not_leak_into_pooled_block_coverage() -> None:
    blocks = np.asarray([[[[0.025, 0.025, 0.025]] * 3,
                          [[0.325, 0.325, 0.325]] * 3]], dtype=np.float32)
    gripper = blocks[:, :, 0]
    coverage = builderbench_state_coverage(
        blocks, gripper, np.asarray([[True, False]]), onset=0,
    )
    assert all(track["cells"] == 1 for track in coverage.values())
    assert all(track["cells_potential_phase"] == 0 for track in coverage.values())


def test_xyz_grid_edges_do_not_clip_outside_visits_into_boundary_cells() -> None:
    specs = COVERAGE_TRACKS["objects"]
    blocks = np.zeros((1, 3, 3), dtype=np.float32)
    blocks[0, 0, 0] = -1.0  # Included lower edge.
    blocks[0, 1, 0] = 1.0   # Excluded upper edge.
    blocks[0, 2, 2] = -1.1  # Outside grid.
    gripper = np.asarray([[1.0, 0, 0]], dtype=np.float32)
    grids = _mark_coverage(initial_coverage_grids(specs), specs,
                           (blocks[..., :2], blocks[:, 0], blocks[:, 1], blocks[:, 2], gripper),
                           jnp.asarray([True]))
    training = coverage_counts(grids, specs)
    evaluation = builderbench_state_coverage(blocks[None], gripper[None],
                                             np.ones((1, 1), dtype=bool), onset=0)
    for name, expected in (("block_0_xyz", 1), ("block_1_xyz", 0),
                           ("block_2_xyz", 0), ("gripper_xyz", 0)):
        assert training[f"exploration/cells_visited_{name}_total"] == expected
        assert evaluation[name]["cells"] == expected


def test_xyz_counts_export_to_progress_wandb_row(tmp_path) -> None:
    blocks = np.zeros((1, 1, 3, 3), dtype=np.float32)
    coverage = builderbench_state_coverage(blocks, blocks[:, :, 0],
                                           np.ones((1, 1), dtype=bool), onset=0)
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"summary": {}, "state_coverage": coverage,
                                "policy_mode": "stochastic"}))
    row = progress_wandb_row({"evaluation": str(path), "checkpoint_step": 10,
                              "target_env_steps": 10})
    for name in coverage:
        assert row[f"evaluation/state_coverage/{name}/cells"] == 1
        assert row[f"evaluation/state_coverage/{name}/cells_skill_phase"] == 1
