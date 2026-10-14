from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.core.rollout import (
    _mark_coverage,
    coverage_counts,
    coverage_floor_masks,
    initial_coverage_grids,
)
from empowerment_warp.envs.locomotion.model import arena_floor_cell_indices
from empowerment_warp.utils.evaluation_metrics import state_coverage


def test_arena_floor_cells_derive_576_open_one_meter_bins() -> None:
    cells = {tuple(cell) for cell in arena_floor_cell_indices(1.0)}

    assert len(cells) == 24 * 24 == 576
    assert (-2, -2) in cells
    assert (21, 21) in cells
    assert (-3, 0) not in cells
    assert (22, 0) not in cells


def test_training_and_evaluation_coverage_share_arena_floor_denominator() -> None:
    specs = (("agent_xy", 1.0, 50.0),)
    masks = coverage_floor_masks(SimpleNamespace(soccer=True), specs)
    assert masks is not None

    # The first two lanes are on floor; the third is in the west wall ring.
    potential = jnp.asarray([[-1.5, -1.5], [21.9, 21.9], [-2.1, 0.0]])
    skill = jnp.asarray([[0.0, 0.0], [21.9, 21.9], [-2.1, 0.0]])
    grids = initial_coverage_grids(specs)
    assert grids is not None
    grids = _mark_coverage(grids, specs, (potential,), jnp.zeros((3,), dtype=jnp.bool_))
    grids = _mark_coverage(grids, specs, (skill,), jnp.ones((3,), dtype=jnp.bool_))
    training = coverage_counts(grids, specs, masks)

    positions = np.stack((np.asarray(potential), np.asarray(skill)))
    alive = np.ones((2, 3), dtype=bool)
    floor_cells = {tuple(cell) for cell in arena_floor_cell_indices(1.0)}
    evaluation = state_coverage(
        positions,
        alive,
        1.0,
        onset=1,
        floor_cells=floor_cells,
    )

    assert training["exploration/cells_visited_agent_xy_total"] == 4.0
    assert evaluation["cells"] == 4.0
    assert training["exploration/coverage_cells_possible_agent_xy"] == 576.0
    assert evaluation["coverage_cells_possible"] == 576.0
    assert training["exploration/coverage_cells_visited_in_floor_agent_xy_total"] == 3.0
    assert evaluation["coverage_cells_visited_in_floor"] == 3.0
    assert training["exploration/coverage_fraction_agent_xy_total"] == pytest.approx(3 / 576)
    assert evaluation["coverage_fraction"] == pytest.approx(3 / 576)


def test_state_coverage_masks_dead_lanes_and_omits_fraction_without_floor() -> None:
    positions = np.asarray(
        [
            [[-1.5, -1.5], [0.0, 0.0]],
            [[0.0, 0.0], [10.0, 10.0]],
        ]
    )
    alive = np.asarray([[True, False], [True, False]])
    floor_cells = {tuple(cell) for cell in arena_floor_cell_indices(1.0)}

    bounded = state_coverage(positions, alive, 1.0, onset=1, floor_cells=floor_cells)
    unbounded = state_coverage(positions, alive, 1.0, onset=1)

    assert bounded["coverage_cells_visited_in_floor"] == 2.0
    assert bounded["coverage_fraction"] == pytest.approx(2 / 576)
    assert "coverage_fraction" not in unbounded
