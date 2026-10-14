"""Switch memory matches direct recurrent queries with the observed history."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.see_pqn import PotentialPQNAgent, get_config as potential_pqn_config
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig


def make_agent(preserve):
    cfg = dict(potential_pqn_config())
    cfg.update(pqn_hidden_size=8, pqn_num_layers=1, pqn_expected_num_envs=2,
               pqn_rollout_steps=4, pqn_num_minibatches=1, pqn_num_epochs=1,
               skill_dim=2, num_skills=2, potential_skill_samples=0,
               pqn_context_free_recurrence=False,
               pqn_preserve_memory_at_switch=preserve)
    return PotentialPQNAgent.create(7, jnp.zeros((1, 3)), jnp.zeros((1, 3)),
                                    cfg)


def test_replayed_values_match_queries_from_actual_pre_step_memory():
    agent = make_agent(True)
    variables = dict(params=agent.network.params, batch_stats=agent.batch_stats)
    obs = jax.random.normal(jax.random.PRNGKey(2), (4, 2, 3))
    anchors = jnp.broadcast_to(obs[0], obs.shape)
    skills = jnp.zeros((4, 2, 2)).at[1:].set(jnp.array([1., 0.]))
    # Include nonzero incoming chunk history and an actual episode reset.
    initial = agent.initial_recurrent_state(2).at[:, :-1].set(0.4).at[:, -1].set(2)
    reset = jnp.array([[False, False], [False, False], [True, False], [False, False]])
    state = initial
    actions, oracle = [], []
    for t in range(4):
        _, memory, previous = agent._unpack_state(state)
        values = []
        for z in jnp.eye(2):
            _, q = agent.network.apply_fn(
                variables, memory, agent._skill_q_inputs(obs[t], obs[t],
                    jnp.broadcast_to(z, (2, 2)))[None], reset[t][None],
                previous[None], name='effective_q', train=False)
            values.append(q[0].max(-1))
        oracle.append(jnp.maximum(jnp.mean(jnp.stack(values), axis=0), 0.))
        _, action, state = agent.sample_actions_recurrent_from_params(
            variables, obs[t], skills[t], anchors[t], state, reset[t],
            jax.random.PRNGKey(t), 0., deterministic=True)
        # Force a nonzero previous action, including immediately before reset.
        # This catches leakage from the preceding episode through that input.
        action = jnp.full_like(action, 2)
        state = state.at[:, -1].set(action)
        actions.append(action)
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    batch = dict(observations=flat(obs), anchor_observations=flat(anchors),
                 skills=flat(skills), actions=flat(jnp.stack(actions)),
                 rnn_resets=flat(reset))
    actual = jax.jit(lambda a: a.potential_values(batch['observations'], batch=batch,
                                                initial_state=initial))(agent)
    np.testing.assert_allclose(actual.reshape(4, 2), jnp.stack(oracle), rtol=1e-5, atol=1e-6)
    fresh = agent._potential_values(batch['observations']).reshape(4, 2)
    # An actual reset forgets history; the switch does not.
    np.testing.assert_allclose(actual.reshape(4, 2)[2, 0], fresh[2, 0], rtol=1e-5, atol=1e-6)
    assert not np.allclose(actual.reshape(4, 2)[1], fresh[1])
    reset_memory = make_agent(False)
    np.testing.assert_array_equal(reset_memory.potential_values(batch['observations'], batch=batch,
        initial_state=initial), reset_memory._potential_values(batch['observations']))
    with pytest.raises(ValueError, match='require the rollout'):
        agent.potential_values(batch['observations'])


def test_config_requires_unrelabeled_history():
    cfg = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
                               achieved_skill_relabeling=False,
                               pqn_preserve_memory_at_switch=True)
    assert cfg.agent_config()['pqn_preserve_memory_at_switch']
    with pytest.raises(ValueError, match='hindsight'):
        replace(cfg, achieved_skill_relabeling=True)
    with pytest.raises(ValueError, match='hindsight'):
        replace(cfg, sample_hindsight_anchors=True)


@pytest.mark.parametrize('preserve', [False, True])
def test_encoder_branches_copy_live_memory_and_episode_reset(preserve):
    from typing import NamedTuple
    from empowerment_warp.training.pqn.collector import make_pqn_encoder_branch_sweep
    class State(NamedTuple):
        obs: object
        reward: object
        terminated: object
        truncated: object
        step_count: object
        rng: object
    class Agent:
        config = {'pqn_preserve_memory_at_switch': preserve}
        def initial_recurrent_state(self, n):
            return jnp.zeros((n, 1))
        def sample_actions_recurrent_from_params(self, variables, obs, skills, anchor,
                memory, reset, key, epsilon, deterministic=False):
            memory = jnp.where(reset[:, None], 0., memory)
            return None, memory[:, 0].astype(jnp.int32), memory + 1
    class Env:
        def step(self, state, actions):
            return state._replace(obs=actions[:, None].astype(jnp.float32))
    state = State(jnp.zeros((2, 1)), jnp.zeros(2), jnp.zeros(2, bool),
                  jnp.zeros(2, bool), jnp.zeros(2), jax.random.split(jax.random.PRNGKey(0), 2))
    sweep = make_pqn_encoder_branch_sweep(Env(), Agent(), num_envs=2, skill_dim=2,
        horizon=3, skill_sampler=lambda k, n: jnp.ones((n, 2)), observation_fn=lambda x: x)
    incoming = jnp.array([[7.], [9.]])
    result, valid = sweep({}, state, 0., jax.random.PRNGKey(1), 1,
                          incoming, jnp.array([False, True]))
    expected = [[7, 0], [8, 1], [9, 2]] if preserve else [[0, 0], [1, 1], [2, 2]]
    np.testing.assert_array_equal(result['observations'].reshape(3, 2), expected)
    np.testing.assert_array_equal(incoming, [[7.], [9.]])
    assert int(valid) == 6


@pytest.mark.parametrize('preserve', [False, True])
def test_live_collector_only_retains_memory_at_switch(preserve):
    from empowerment_warp.envs.games import JaxGameBatchEnv
    from empowerment_warp.envs.wrappers import FlatObservationEnv
    from empowerment_warp.training.pqn.collector import (
        initialize_potential_pqn_collector, make_potential_pqn_chunk)
    env = FlatObservationEnv(JaxGameBatchEnv('craftax:classic-symbolic',
                                            num_envs=2, episode_length=4))
    cfg = dict(potential_pqn_config())
    cfg.update(pqn_hidden_size=8, pqn_num_layers=1, pqn_expected_num_envs=2,
               pqn_rollout_steps=6, pqn_num_minibatches=1, skill_dim=2,
               num_skills=2, pqn_preserve_memory_at_switch=preserve)
    agent = PotentialPQNAgent.create(0, jnp.zeros((1, env.observation_size)),
        jnp.zeros((1, env.action_size)), cfg)
    sampler = lambda key, n: jnp.broadcast_to(jnp.array([1., 0.]), (n, 2))
    carry = initialize_potential_pqn_collector(env, agent, jax.random.PRNGKey(1),
        num_envs=2, skill_dim=2, potential_steps=2, skill_sampler=sampler)
    collect = make_potential_pqn_chunk(env, agent, chunk_steps=6, num_envs=2,
        skill_dim=2, potential_steps=2, skill_sampler=sampler)
    _, batch, _ = collect(dict(params=agent.network.params, batch_stats=agent.batch_stats),
                          carry, jnp.array(0.))
    resets = np.asarray(batch['rnn_resets']).reshape(6, 2)
    np.testing.assert_array_equal(resets[2], [not preserve] * 2)
    np.testing.assert_array_equal(resets[4], [True] * 2)
    # The TD reward-phase cut is still present even though memory can continue.
    batch['pqn_rewards'] = jnp.zeros(12)
    prepared, _, _ = agent._prepare_rollout_batch(batch, num_envs=2, chunk_steps=6)
    np.testing.assert_array_equal(np.asarray(prepared['pqn_command_boundary']).reshape(6, 2)[1], [1, 1])
