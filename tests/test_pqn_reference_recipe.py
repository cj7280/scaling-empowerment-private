"""Reference reward reductions and the real Craftax reset/codec path."""

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.agents.see_pqn import PotentialPQNAgent
from empowerment_warp.agents.skill_classifier import SkillLearner
from empowerment_warp.training.pqn.archive import PQNEncoderArchive
from empowerment_warp.training.pqn.env import PQNEnv
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig


def test_potential_is_library_mean_not_best_skill():
    """The z=0 value is the MEAN over the Monte Carlo skill draws, not the best skill."""
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
        num_envs=4, skill_dim=4, num_skills=4, pqn_hidden_size=8, pqn_num_layers=1,
        discrete_skills=False, potential_skill_samples=4)
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, 7))
    agent = PotentialPQNAgent.create(0, obs[:1], jnp.zeros((1, 5)), config.agent_config())
    from empowerment_warp.utils.networks import RecurrentQNetwork
    carry = RecurrentQNetwork(action_dim=5, hidden_size=8).initialize_carry(obs.shape[0])
    key = jax.random.PRNGKey(2)
    draws = agent._probe_skills(obs.dtype, rows=obs.shape[0], key=key)
    assert draws.shape == (4, 3, 4)
    values = []
    for skills in draws:
        inputs = agent._skill_q_inputs(obs, obs, skills)
        _, q = agent.network.apply_fn(
            {'params': agent.network.params, 'batch_stats': agent.batch_stats},
            carry, inputs[None], jnp.ones((1, 3), bool), jnp.zeros((1, 3), jnp.int32),
            name='effective_q', train=False)
        values.append(q[0].max(axis=-1))
    values = jnp.stack(values)
    actual = agent._potential_values(obs, key)
    np.testing.assert_allclose(actual, jnp.maximum(values.mean(axis=0), 0.0), rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(np.asarray(actual - values.max(axis=0)))) > 1e-5
    assert config.separate_skill0_q and config.total_env_steps == 500_000_000


def test_craftax_codec_inventory_truncation_and_optimistic_reset():
    env = PQNEnv('craftax:classic-symbolic', num_envs=4, episode_length=1, reset_ratio=2)
    keys = jax.random.split(jax.random.PRNGKey(10), 4)
    state = jax.jit(env.reset)(keys)
    assert state.obs.shape == (4, 1345)
    archive = PQNEncoderArchive(num_envs=4, capacity=8, episode_length=1000, future_gamma=.95)
    archive_state = archive.initialize(4, 1345, 4)
    archive_state = jax.jit(archive.insert)(archive_state, dict(
        observations=state.obs, skills=jnp.eye(4), episode_ids=jnp.arange(4), timesteps=jnp.zeros(4)))
    decoded = archive.codec.decode_jax(archive_state.data.obs_map[:4], archive_state.data.obs_tail[:4])
    np.testing.assert_array_equal(decoded, state.obs)
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
                                  discrete_skills=False, num_skills=16, potential_skill_samples=8)
    learner = SkillLearner(config.skill_config(1345))
    np.testing.assert_array_equal(learner.encoder_observations(decoded), state.obs[:, 1323:1335])

    state = state._replace(game_state=state.game_state.replace(achievements=jnp.ones_like(state.game_state.achievements)))
    ended = jax.jit(env.step)(state, jnp.zeros(4, jnp.int32))
    assert np.asarray(ended.truncated).all()
    for _, name in env.achievement_columns:
        np.testing.assert_array_equal(ended.info[name], np.full(4, 100.))
    reset_keys = jax.random.split(jax.random.PRNGKey(12), 4)
    reset = jax.jit(env.reset_done)(ended, reset_keys)
    assert not np.asarray(reset.truncated | reset.terminated).any()
    assert np.asarray(reset.step_count == 0).all()
    # Derive the reference mapping and compare every backend state leaf.
    world_keys = jax.vmap(jax.random.split)(reset_keys)[:, 1]
    _, worlds = env.env._vector.reset(world_keys[:2])
    take = jnp.arange(4) // 2
    chosen = jax.random.choice(jax.random.fold_in(world_keys[0], 0xC0FFEE), 4,
                              shape=(2,), replace=False, p=jnp.ones(4))
    take = take.at[chosen].set(jnp.arange(2))
    expected_worlds = jax.tree_util.tree_map(lambda value: value[take], worlds)
    for expected, actual in zip(jax.tree_util.tree_leaves(expected_worlds), jax.tree_util.tree_leaves(reset.game_state)):
        np.testing.assert_array_equal(expected, actual)
