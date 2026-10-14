"""Per-lane phase metrics and evaluator behavior without simulator or GPU work."""
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.sac import evaluation
from empowerment_warp.utils import evaluation_metrics


def test_phase_metrics_use_each_lane_boundary_and_alive_mask():
    positions = np.zeros((4, 3, 2))
    positions[:, :, 0] = np.arange(1, 5)[:, None] + np.array([0, 10, 20])
    initial = np.array([[0, 0], [10, 0], [20, 0]])
    alive = np.ones((4, 3), dtype=bool)
    alive[1:, 2] = False
    onset = np.array([0, 3, 2])
    displacement = evaluation_metrics.phase_displacement(positions, alive, initial, onset=onset)
    np.testing.assert_array_equal(displacement['potential'], [0, 3, 1])
    np.testing.assert_array_equal(displacement['skill'], [4, 1, 0])
    coverage = evaluation_metrics.state_coverage(positions, alive, 1, onset=onset,
                                         skill_indices=np.array([0, 1, 2]))
    assert coverage['cells'] == 9
    assert coverage['cells_potential_phase'] == 4
    assert coverage['cells_skill_phase'] == 5
    assert coverage['cells_per_skill_mean'] == 5 / 3


@pytest.mark.parametrize('onset', [0, 2, 3, 4])
def test_scalar_phase_metrics_remain_equivalent(onset):
    track = np.arange(24).reshape(4, 3, 2).astype(float)
    alive = np.ones((4, 3), dtype=bool)
    initial = np.zeros((3, 2))
    expected_prefix = initial if onset == 0 else track[onset - 1]
    result = evaluation_metrics.phase_displacement(track, alive, initial, onset=onset)
    np.testing.assert_array_equal(result['potential'], np.linalg.norm(expected_prefix, axis=-1))
    np.testing.assert_array_equal(result['skill'], np.linalg.norm(track[-1] - expected_prefix, axis=-1))
    scalar = evaluation_metrics.state_coverage(track, alive, 1, onset=onset)
    vector = evaluation_metrics.state_coverage(track, alive, 1, onset=np.full(3, onset))
    assert scalar == vector


@dataclass
class _Config:
    environment: str = 'ant'
    num_envs: int = 3
    num_skills: int = 1
    skill_dim: int = 2
    seed: int = 0
    episode_length: int = 4
    potential_steps: int = 2
    skill_start_mode: str = 'two_phase'
    live_potential_only: bool = False
    branch_live_two_phase: bool = False
    potential_lane_fraction: float = 0.0
    random_potential_switch_max_steps: int = 3
    allow_reference_deviation: bool = True
    locomotion_observation_contract: str = 'test'


def test_evaluation_schedule_is_repeatable_and_independent_of_training_seed():
    config = _Config(random_potential_switch_max_steps=200)
    first = evaluation.evaluation_switch_steps(config, 64, 10_000)
    second = evaluation.evaluation_switch_steps(config, 64, 10_000)
    np.testing.assert_array_equal(first, second)
    training = evaluation.potential_switch_steps(jnp.arange(64), 2, 200, config.seed)
    assert np.any(np.asarray(first) != np.asarray(training))
    fixed = evaluation.evaluation_switch_steps(replace(config, random_potential_switch_max_steps=0), 64, 10_000)
    np.testing.assert_array_equal(fixed, np.full(64, 2))


def test_marginal_branch_horizon_is_not_an_evaluation_prefix():
    config = _Config(skill_start_mode='marginal_branch', random_potential_switch_max_steps=0)
    np.testing.assert_array_equal(evaluation.evaluation_switch_steps(config, 8, 10), np.zeros(8))
    np.testing.assert_array_equal(evaluation.evaluation_switch_steps(
        replace(config, live_potential_only=True), 8, 10), np.full(8, 2))
