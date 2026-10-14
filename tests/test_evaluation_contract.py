from __future__ import annotations

import numpy as np
import pytest

from empowerment_warp.utils.evaluation_metrics import _metric_statistics


def test_metric_statistics_reports_population_distribution() -> None:
    result = _metric_statistics(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert result == pytest.approx(
        {
            "mean": 1.5,
            "std": np.sqrt(1.25),
            "min": 0.0,
            "p50": 1.5,
            "p90": 2.7,
            "max": 3.0,
        }
    )


def test_eval_skill_library_falls_back_to_a_seeded_probe_set() -> None:
    """Continuous (sym-CSF) runs have no skill library, so every SAC evaluator would raise or
    silently sweep z=0. `eval_skill_library` returns a seeded probe set of unit vectors,
    reproducible from the run's seed alone."""

    from types import SimpleNamespace

    from empowerment_warp.training.sac.evaluation import eval_skill_library

    config = SimpleNamespace(num_skills=30, skill_dim=2, seed=7)
    probes = np.asarray(eval_skill_library(config))
    assert probes.shape == (30, 2)
    np.testing.assert_allclose(np.linalg.norm(probes, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(probes, np.asarray(eval_skill_library(config)))
    other_seed = np.asarray(eval_skill_library(SimpleNamespace(num_skills=30, skill_dim=2, seed=8)))
    assert not np.allclose(probes, other_seed)


def test_state_coverage_counts_distinct_cells_per_phase_and_per_skill():
    """METRA/CSF-style state coverage: distinct grid cells visited by the pooled trajectories, alive
    steps only, split into potential prefix / skill phase, plus the mean per-skill coverage."""
    import numpy as np
    from empowerment_warp.utils.evaluation_metrics import state_coverage
    T, N = 6, 4
    pos = np.zeros((T, N, 2))
    pos[:, 0, 0] = np.arange(T)            # lane 0 walks +x one unit per step: cells (0..5, 0)
    pos[:, 1, 1] = np.arange(T)            # lane 1 walks +y: cells (0, 0..5)
    pos[:, 2] = 10.0                       # lane 2 parked at (10, 10)
    pos[:, 3, 0] = np.arange(T) * 0.4      # lane 3 creeps: cells 0,0,0,1,1,2 -> {0,1,2} on x
    alive = np.ones((T, N), bool); alive[3:, 2] = False   # lane 2 dies at step 3 (still counted before)
    cov = state_coverage(pos, alive, 1.0, onset=2, skill_indices=np.array([0, 0, 1, 1]))
    assert cov["cells"] == len({(t, 0) for t in range(6)} | {(0, t) for t in range(6)} | {(10, 10)} | {(0, 0), (1, 0), (2, 0)})
    assert cov["cells_potential_phase"] == len({(0, 0), (1, 0), (0, 1), (10, 10)})
    assert cov["cells_skill_phase"] == len({(t, 0) for t in range(2, 6)} | {(0, t) for t in range(2, 6)} | {(10, 10), (0, 0), (1, 0), (2, 0)})
    # skill 0 = lanes 0,1 over steps 2..5: 8 distinct cells; skill 1 = lanes 2,3: (10,10) [alive at step 2 only] + {0,1,2}x{0} = 4
    assert cov["cells_per_skill_mean"] == (8 + 4) / 2
    assert cov["cells"] > cov["cells_skill_phase"] > 0


def test_phase_displacement_splits_the_prefix_from_the_skill_phase():
    """potential = |p(end of z=0 prefix) - p(reset)|, skill = |p(last alive) - p(end of prefix)|;
    a lane that dies inside the prefix has zero skill-phase displacement; dead steps are ignored."""
    import numpy as np
    from empowerment_warp.utils.evaluation_metrics import phase_displacement
    T, N = 6, 3
    initial = np.zeros((N, 2))
    track = np.zeros((T, N, 2))
    track[:, 0, 0] = np.arange(1, T + 1)          # lane 0: 1,2,...,6 on x; prefix (onset 2) ends at x=2
    track[:, 1, 1] = np.arange(1, T + 1)          # lane 1: same on y but dies at step 4 (positions 5,6 ignored)
    track[:, 2, 0] = 100.0                        # lane 2: dies at step 1 -> prefix end = p(step 0) = 100
    alive = np.ones((T, N), bool); alive[4:, 1] = False; alive[1:, 2] = False
    d = phase_displacement(track, alive, initial, onset=2)
    assert np.allclose(d["potential"], [2.0, 2.0, 100.0])
    assert np.allclose(d["skill"], [4.0, 2.0, 0.0])
    # a [T, N, C, 2] track (BuilderBench cubes) is handled per cube
    d2 = phase_displacement(track[:, :, None], alive, initial[:, None], onset=2)
    assert d2["skill"].shape == (N, 1) and np.allclose(d2["skill"][:, 0], [4.0, 2.0, 0.0])
