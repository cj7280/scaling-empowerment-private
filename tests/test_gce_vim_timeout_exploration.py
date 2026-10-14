"""Timeout targets and training exploration use pre-reset states, not new episodes."""
import csv
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.baselines.collector import (
    EXPLORATION_DISPLACEMENT_FIELDS, episode_displacements,
)
from empowerment_warp.training.baselines.config import EnvironmentConfig, PPOConfig
from empowerment_warp.agents.ppo import (
    ControllerState, RolloutBatch, calculate_gae, create_ppo_train_state,
    empty_observation_norm, make_ppo_update, normalize_observations, squashed_log_prob,
)
from empowerment_warp.training.baselines.runner import append_csv
from empowerment_warp.core.rollout import (
    COVERAGE_TRACKS, _mark_coverage, coverage_counts, initial_coverage_grids,
)


def test_gae_timeout_terminal_and_reset_seams():
    # Three simultaneous lanes: timeout, true terminal, uninterrupted.
    values = jnp.full((3, 3), 2.)
    next_values = jnp.array([[10., jnp.nan, 10.], [20., 20., 20.], [30., 30., jnp.nan]])
    dones = jnp.array([[True, True, False], [False, False, False], [True, False, True]])
    terminal = jnp.array([[False, True, False], [False, False, False], [False, False, True]])
    rewards = jnp.ones((3, 3))
    advantages, returns = calculate_gae(rewards, dones, values, next_values, terminal,
                                        gamma=.9, gae_lambda=.8)
    expected = np.empty((3, 3))
    following = np.zeros(3)
    for t in (2, 1, 0):
        delta = 1 + .9*np.where(terminal[t], 0, next_values[t]) - 2
        following = delta + .9*.8*np.where(dones[t], 0, following)
        expected[t] = following
    np.testing.assert_allclose(advantages, expected, atol=1e-5)
    np.testing.assert_allclose(returns, expected+2, atol=1e-5)
    assert advantages[0, 0] == 8  # 1 + .9*10 - 2: timeout retains bootstrap
    assert advantages[0, 1] == -1  # genuine terminal ignores even a NaN next value
    assert advantages[-1, 0] == 26  # timeout at the rollout boundary too
    changed, _ = calculate_gae(rewards.at[1:].set(1000), dones, values, next_values, terminal,
                               gamma=.9, gae_lambda=.8)
    np.testing.assert_array_equal(advantages[0, :2], changed[0, :2])


@pytest.mark.parametrize('normalize', [False, True])
def test_real_ppo_uses_final_observation_not_reset_observation(normalize):
    spec = EnvironmentConfig('ant', 42, 8, 1., 400, 100, 100, .99, 42)
    config = PPOConfig(num_envs=4, rollout_steps=4, normalize_actor_observations=normalize)
    network, state = create_ppo_train_state(jax.random.PRNGKey(0), spec, config)
    obs = jnp.zeros((4, 4, 42))
    normal, values = network.apply({'params':state.params}, obs)
    raw = normal.sample(seed=jax.random.PRNGKey(1))
    terminal = jnp.zeros((4, 4), dtype=bool).at[1, 0].set(True)
    final_obs = jnp.ones_like(obs)*.7
    final_obs = final_obs.at[1, 0].set(jnp.nan)
    batch = RolloutBatch(obs, obs, raw, squashed_log_prob(normal, raw), values,
                         jnp.ones((4, 4), dtype=bool), final_obs, terminal)
    norm = empty_observation_norm(42)._replace(mean=jnp.ones(42)*.2, variance=jnp.ones(42)*4)
    # Deliberately different reset value; all transitions here reset.
    controller = ControllerState(None, jnp.ones((4, 42))*-7, norm)
    next_obs = jnp.where(terminal[..., None], 0., final_obs)
    if normalize:
        next_obs = normalize_observations(norm, next_obs)
    _, next_values = network.apply({'params':state.params}, next_obs)
    expected = jnp.ones((4, 4)) + .99*jnp.where(terminal, 0., next_values)
    _, metrics, _, returns, _ = make_ppo_update(network, spec, config)(
        state, batch, jnp.ones((4, 4)), controller, jax.random.PRNGKey(2))
    np.testing.assert_allclose(returns, expected, atol=1e-6)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))


def test_training_coverage_reuses_ablation_bins_and_is_monotonic():
    specs = COVERAGE_TRACKS['locomotion'][:1]
    grids = initial_coverage_grids(specs)
    phase = jnp.zeros(4, dtype=bool)
    first = (jnp.array([[0., 0.], [.9, .9], [1.1, 0.], [500., 500.]]),)
    grids = _mark_coverage(grids, specs, first, phase)
    counts = coverage_counts(grids, specs)
    assert counts['exploration/cells_visited_agent_xy_total'] == 2
    grids = _mark_coverage(grids, specs, (jnp.full((4, 2), -1.),), phase)
    assert coverage_counts(grids, specs)['exploration/cells_visited_agent_xy_total'] == 3


def test_displacement_definitions_match_ablation_and_csv_omits_missing_episodes(tmp_path):
    starts = (jnp.array([[10., 10.], [4., 5.]]), jnp.zeros((2, 2)))
    ends = (starts[0]+jnp.array([3., 4.]), jnp.full((2, 2), 2.))
    metrics = episode_displacements(SimpleNamespace(), None, starts, ends)
    np.testing.assert_allclose(metrics['episode_agent_disp_xy'], 5.)
    np.testing.assert_allclose(metrics['episode_ball_disp_xy'], np.sqrt(8))
    # Reuse initial object state and the same mean/max over three cubes.
    delta = jnp.array([[[3., 4., 0.], [0., 0., 2.], [0., 0., 0.]]])
    reached = SimpleNamespace(initial_object_pos=jnp.zeros_like(delta),
                               metrics=SimpleNamespace(obj_moved=jnp.array([1.])))
    positions = (delta[..., :2], delta[:, 0], delta[:, 1], delta[:, 2], jnp.zeros((1, 3)))
    metrics = episode_displacements(SimpleNamespace(_object_positions=True), reached, (), positions)
    np.testing.assert_allclose(metrics['episode_cube_disp_xyz'], 7/3)
    np.testing.assert_allclose(metrics['episode_cube_disp_max_xy'], 5.)
    path = tmp_path/'train.csv'
    append_csv(path, {'env_steps':1, 'exploration/completed_episodes':0},
               optional_fields=EXPLORATION_DISPLACEMENT_FIELDS)
    append_csv(path, {'env_steps':2, 'exploration/completed_episodes':2,
                      'exploration/episode_agent_disp_xy':5.},
               optional_fields=EXPLORATION_DISPLACEMENT_FIELDS)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['exploration/episode_agent_disp_xy'] == ''
    assert float(rows[1]['exploration/episode_agent_disp_xy']) == 5.
    assert all(None not in row for row in rows)
