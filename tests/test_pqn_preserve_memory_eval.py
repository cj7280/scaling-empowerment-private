"""Switching evaluations preserve each branch's observed pre-step memory."""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.pqn import branch_eval


GameState = namedtuple('GameState', 'player_position achievements')
State = namedtuple('State', 'game_state obs reward terminated truncated step_count rng')


class TinyGrid:
    def __init__(self, count):
        self.num_envs = count

    def reset(self, keys):
        n = self.num_envs
        position = jnp.zeros((n, 2), jnp.int32)
        return State(GameState(position, jnp.zeros((n, 1), bool)),
                     position.astype(jnp.float32), jnp.zeros(n), jnp.zeros(n, bool),
                     jnp.zeros(n, bool), jnp.zeros(n, jnp.int32), keys)

    def step(self, state, actions):
        position = state.game_state.player_position.at[:, 0].add(actions)
        return state._replace(game_state=state.game_state._replace(player_position=position),
                              obs=position.astype(jnp.float32), reward=actions.astype(jnp.float32),
                              step_count=state.step_count + 1)


class PackedAgent:
    """Distinct potential/effective carry slots and an observable previous action."""
    def __init__(self, preserve):
        self.config = {'skill_dim': 2, 'pqn_recurrent': True}
        if preserve is not None:
            self.config['pqn_preserve_memory_at_switch'] = preserve
        self.calls = []

    def initial_recurrent_state(self, count, dtype=jnp.float32):
        return jnp.zeros((count, 5), dtype)

    def sample_actions_recurrent_from_params(self, variables, obs, skills, anchors,
                                             recurrent, reset, key, epsilon, deterministic=False):
        state = jnp.where(reset[:, None], 0, recurrent)
        skill_on = jnp.linalg.norm(skills, axis=-1) > 0
        # Both independent memories and the actual previous action affect behavior.
        action = jnp.where(skill_on, jnp.sum(state, axis=-1).astype(jnp.int32), 1)
        next_state = state + jnp.asarray([1, 10, 100, 1000, 0], jnp.float32)
        next_state = next_state.at[:, -1].set(action)
        return jax.nn.one_hot(action, 2), action, next_state

    def sample_actions_recurrent(self, *, observations, skills, anchor_observations,
                                 recurrent_state, reset, seed, temperature, deterministic):
        _, actions, next_state = self.sample_actions_recurrent_from_params(
            {}, observations, skills, anchor_observations, recurrent_state, reset, seed, 0,
            deterministic=deterministic)
        self.calls.append((np.asarray(observations), np.asarray(skills),
                           np.asarray(recurrent_state), np.asarray(reset), np.asarray(next_state)))
        return actions[:, None].astype(jnp.float32), next_state


@pytest.mark.parametrize('preserve', [None, False, True])
def test_metric_switch_gathers_complete_pre_observation_memory(preserve):
    agent = PackedAgent(preserve)
    rollout = branch_eval.make_switching_rollout(
        TinyGrid(2), TinyGrid(4), agent, prefix_steps=2, episode_length=3, skills_per_context=2)
    arrays = rollout({}, jnp.asarray(0.0), jax.random.PRNGKey(1), jax.random.PRNGKey(2),
                     jnp.tile(jnp.eye(2), (2, 1)))
    # Two prefix calls produce [2,20,200,2000,1]; the switch observation has
    # not been consumed yet. All four branches must receive this exact carry.
    expected_action = 2223 if preserve else 0
    np.testing.assert_array_equal(arrays['cells'][-1, :, 1], 2 + expected_action)
    np.testing.assert_array_equal(arrays['episode_lengths'], [3] * 4)
    np.testing.assert_array_equal(arrays['valid'], np.ones((4, 4), bool))


class TinyScalarGrid:
    _episode_limit = 4

    def __init__(self):
        self._state = None

    def reset(self, *, seed):
        self._state = jnp.asarray(0)
        return np.asarray([0, 0], np.float32), {}

    def step(self, action):
        self._state += 1
        return np.asarray([int(self._state), 0], np.float32), 0.0, False, False, {}

    def render(self):
        return np.full((2, 2, 3), int(self._state), np.uint8)


@pytest.mark.parametrize('preserve', [None, False, True])
def test_media_branches_restore_prefix_memory_and_consume_switch_once(preserve):
    agent = PackedAgent(preserve)
    actor = branch_eval._make_recurrent_eval_actor(agent, jax.random.PRNGKey(3))
    env = TinyScalarGrid()
    rollouts = branch_eval.reference._collect_jax_game_branch_skill_rollouts(
        env, actor, {'benchmark_backend': 'craftax',
                     'skill_dim': 2, 'potential_steps': 2}, np.eye(2),
        count=2, warmup_steps=2, video_frame_skip=1, eval_temperature=1.0,
        eval_deterministic=True, supports_deterministic=True, seed=0)
    assert len(rollouts) == 3
    assert len(agent.calls) == 8  # Two prefix calls, then two calls per branch.
    prefix_carry = agent.calls[1][4]
    np.testing.assert_array_equal(prefix_carry, [[2, 20, 200, 2000, 1]])
    for index, call in enumerate(agent.calls[2:]):
        obs, skill, recurrent, reset, next_state = call
        np.testing.assert_array_equal(obs, [[2 + index % 2, 0]])
        if index % 2 == 0:
            np.testing.assert_array_equal(recurrent, prefix_carry)
            skill_on = np.any(skill)
            np.testing.assert_array_equal(reset, [skill_on and not preserve])
    assert actor.snapshot_recurrent_state()[0] is None  # Restore caller's entry state.
    actor.reset_recurrent_state()
    actor(np.asarray([0, 0]), skills=np.asarray([1, 0]), deterministic=True)
    np.testing.assert_array_equal(agent.calls[-1][2], np.zeros((1, 5)))
    np.testing.assert_array_equal(agent.calls[-1][3], [True])


@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('separate', [True])
def test_scalar_actor_matches_real_agent_switch_and_episode_reset(preserve, separate):
    from empowerment_warp.agents.see_pqn import PotentialPQNAgent, get_config

    config = get_config()
    config.update(pqn_hidden_size=4, pqn_num_layers=1, skill_dim=2, num_skills=2,
                  pqn_expected_num_envs=1, separate_skill0_q=separate,
                  pqn_preserve_memory_at_switch=preserve)
    agent = PotentialPQNAgent.create(0, jnp.zeros((1, 2)), jnp.zeros((1, 3)),
                                     config)
    actor = branch_eval._make_recurrent_eval_actor(agent, jax.random.PRNGKey(4))
    observation = jnp.asarray([1.0, 2.0])
    actor(observation, skills=jnp.zeros(2), anchor_observations=observation, deterministic=True)
    actor(observation + 1, skills=jnp.zeros(2), anchor_observations=observation + 1,
          deterministic=True)
    prefix_carry = actor.snapshot_recurrent_state()[0]
    switch = observation + 2
    _, expected = agent.sample_actions_recurrent(
        observations=switch[None], skills=jnp.asarray([[1.0, 0.0]]),
        anchor_observations=switch[None], recurrent_state=prefix_carry,
        reset=jnp.asarray([not preserve]), seed=jax.random.PRNGKey(9),
        temperature=1.0, deterministic=True)
    actor(switch, skills=jnp.asarray([1.0, 0.0]), anchor_observations=switch, deterministic=True)
    np.testing.assert_allclose(actor.snapshot_recurrent_state()[0], expected, atol=1e-7)
    actor.reset_recurrent_state()
    actor(switch, skills=jnp.asarray([1.0, 0.0]), anchor_observations=switch, deterministic=True)
    _, expected_reset = agent.sample_actions_recurrent(
        observations=switch[None], skills=jnp.asarray([[1.0, 0.0]]),
        anchor_observations=switch[None], recurrent_state=agent.initial_recurrent_state(1),
        reset=jnp.asarray([True]), seed=jax.random.PRNGKey(9),
        temperature=1.0, deterministic=True)
    np.testing.assert_allclose(actor.snapshot_recurrent_state()[0], expected_reset, atol=1e-7)
