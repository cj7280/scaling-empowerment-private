"""Controller normalization preserves raw estimator data and terminal semantics."""
from dataclasses import asdict
from types import SimpleNamespace
from typing import NamedTuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.baselines.collector import initialize, make_controller_collector, make_evaluator
from empowerment_warp.training.baselines.config import EnvironmentConfig, PPOConfig
from empowerment_warp.agents.ppo import (
    calculate_gae, controller_learning_rate, create_ppo_train_state,
    make_ppo_update, normalize_observations,
)
from empowerment_warp.training.baselines.recipe import ControllerPPOConfig, Recipe
from empowerment_warp.checkpointing import CheckpointStore
from empowerment_warp.recipes import resolve


@struct.dataclass
class Data:
    qpos: object
    qvel: object
    ctrl: object
    mocap_pos: object
    mocap_quat: object

    def where(self, mask, replacement):
        return jax.tree.map(lambda a, b: jnp.where(mask, b, a), self, replacement)


class Metrics(NamedTuple):
    agent_x: object
    agent_y: object
    ball_x: object
    ball_y: object


class State(NamedTuple):
    data: object
    obs: object
    terminated: object
    truncated: object
    reward: object
    metrics: object
    clock: object


class TinyEnv:
    """Deterministic stateful test environment for the real collector/evaluator."""
    observation_size = 42
    action_size = 8
    soccer = False
    config = SimpleNamespace(num_envs=4)

    def reset(self, keys):
        obs = jnp.full((len(keys), 42), 3.)
        return State(Data(obs, obs, obs, obs, obs), obs, jnp.zeros(len(keys), bool),
                     jnp.zeros(len(keys), bool), jnp.zeros(len(keys)), Metrics(*(jnp.zeros(len(keys)),)*4),
                     jnp.zeros(len(keys), jnp.int32))

    def step(self, state, actions):
        obs = state.obs.at[:, :8].add(actions*.1).at[:, 8:].add(.1)
        clock = state.clock+1
        return state._replace(obs=obs, clock=clock, truncated=clock >= 2,
                              data=Data(obs, obs, obs, obs, obs))

    def where_done(self, state, reset):
        mask = state.terminated | state.truncated
        return jax.tree.map(lambda a, b: jnp.where(mask.reshape((len(mask),)+(1,)*(a.ndim-1)), b, a), state, reset)


def normalized_recipe():
    return Recipe('ant', 'gce_vector', 30, 10000, 'unused', num_envs=4,
                  rollout_steps=4, episode_steps=4, ppo=ControllerPPOConfig(
                      normalize_actor_observations=True,
                      anneal_learning_rate=True, learning_rate=3e-4, entropy_coefficient=.01))


# Overrides that turn latentgce.yaml into the H=30 controller with an exponentiated reward,
# an unbounded decoder and constant-LR PPO on raw observations.
H30_OVERRIDES = ['horizon=30', 'vector_gce_bound_action_decoder=false', 'exponentiate_empowerment=true',
                        'ppo.learning_rate=0.001', 'ppo.normalize_actor_observations=false',
                        'ppo.anneal_learning_rate=false']


@pytest.mark.parametrize('environment', ['ant', 'ant_soccer', 'builderbench_corner3'])
def test_recipe_plumbing_against_h30_controller(environment):
    path = f'configs/{environment}/latentgce.yaml'
    _, h30 = resolve(path, overrides=H30_OVERRIDES)
    _, new = resolve(path, overrides=['horizon=30'])
    assert h30.resolved()['ppo'] == asdict(PPOConfig(num_envs=256, rollout_steps=80))
    assert not h30.ppo.anneal_learning_rate and not h30.vector_gce_normalize_inputs
    assert new.ppo.normalize_actor_observations and not new.vector_gce_normalize_inputs
    assert 'normalize_rewards' not in new.resolved()['ppo']
    assert new.ppo.anneal_learning_rate and new.ppo.learning_rate == 3e-4
    assert new.ppo.entropy_coefficient == .01
    for key in ('estimator', 'outcome_indices', 'model_environment', 'effective_scale'):
        assert h30.resolved()[key] == new.resolved()[key]
    assert h30.exponentiate_empowerment and not h30.vector_gce_bound_action_decoder
    assert not new.exponentiate_empowerment and new.vector_gce_bound_action_decoder
    assert new.resolved()['reward'] == 'capacity_bits(current)'
    assert new.resolved()['effective_scale']['ppo_minibatches_per_update'] == 16
    assert new.resolved()['effective_scale']['ppo_samples_per_controller_transition'] == 4


@pytest.mark.parametrize('environment', ['ant', 'ant_soccer', 'builderbench_corner3'])
def test_h100_uses_raw_estimator_inputs_with_the_same_ppo(environment):
    metadata, recipe = resolve(f'configs/{environment}/latentgce.yaml')
    _, vim = resolve(f'configs/{environment}/vim.yaml')
    assert metadata['seeds'] == [0, 1, 2, 3, 4]
    assert recipe.horizon == 100
    assert not recipe.vector_gce_normalize_inputs
    assert recipe.resolved()['vector_gce_inputs'] == 'raw'
    assert not recipe.exponentiate_empowerment and recipe.vector_gce_bound_action_decoder
    assert recipe.resolved()['reward'] == 'capacity_bits(current)'
    assert recipe.ppo == vim.ppo
    assert recipe.ppo.normalize_actor_observations


@pytest.mark.parametrize('environment', ['ant', 'ant_soccer', 'builderbench_corner3'])
def test_vim_matches_selected_controller_without_changing_estimator(environment):
    metadata, vim = resolve(f'configs/{environment}/vim.yaml')
    _, gce = resolve(f'configs/{environment}/latentgce.yaml')
    assert metadata['seeds'] == [0, 1, 2]
    assert vim.ppo == gce.ppo
    assert vim.gamma == gce.gamma == .99
    assert vim.method == 'vim' and not vim.vector_gce_normalize_inputs
    assert vim.estimator.learning_rate == 3e-4
    assert vim.resolved()['reward'] == 'max(0, psi(reached)/beta)'
    assert 'normalize_rewards' not in vim.resolved()['ppo']


def test_real_collector_ppo_eval_and_checkpoint_frozen_moments(tmp_path):
    recipe = normalized_recipe()
    env = TinyEnv()
    spec = recipe.model_environment
    policy, train = create_ppo_train_state(jax.random.PRNGKey(0), spec, recipe.ppo,
                                          total_updates=recipe.updates)
    controller = initialize(env, jax.random.PRNGKey(1))
    norm = controller.observation_norm._replace(mean=jnp.ones(42), variance=jnp.ones(42)*4,
                                                count=jnp.asarray(10.))
    controller = controller._replace(observation_norm=norm)
    frozen = controller
    current, roots, batch, _, _ = make_controller_collector(env, policy, recipe)(
        train, controller, jax.random.PRNGKey(2))
    _, root_key = jax.random.split(jax.random.PRNGKey(2))
    times = jax.random.randint(root_key, (recipe.num_envs,), 0, recipe.rollout_steps)
    # Probe starts are actual time/lane entries, with their physical state copied.
    np.testing.assert_array_equal(roots.obs, batch.observations[times, jnp.arange(recipe.num_envs)])
    np.testing.assert_array_equal(roots.data.qpos, roots.obs)
    np.testing.assert_allclose(batch.policy_observations, normalize_observations(norm, batch.observations))
    assert not np.allclose(batch.policy_observations, batch.observations)
    np.testing.assert_array_equal(roots.obs[:, 8:]-roots.clock[:, None]*.1, 3.)
    distribution, expected_values = policy.apply({'params':train.params}, batch.policy_observations)
    np.testing.assert_allclose(batch.values, expected_values, atol=1e-6)
    assert current.observation_norm.count > norm.count
    assert batch.dones[1].all() and not batch.terminated.any()
    rewards = jnp.ones((4, 4))
    _, next_values = policy.apply({'params':train.params}, normalize_observations(norm, batch.terminal_observations))
    expected_advantages, expected_returns = calculate_gae(rewards, batch.dones, batch.values,
        next_values, batch.terminated, gamma=recipe.gamma, gae_lambda=recipe.ppo.gae_lambda)
    updated, metrics, advantages, returns, _ = make_ppo_update(policy, spec, recipe.ppo)(
        train, batch, rewards, frozen, jax.random.PRNGKey(3))
    np.testing.assert_allclose(advantages, expected_advantages, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(returns, expected_returns, rtol=1e-5, atol=1e-6)
    assert int(updated.step) == 16 and all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    # Evaluation uses the checkpoint moments without modifying them.
    evaluate = make_evaluator(env, policy, recipe, deterministic=True)
    before = evaluate(updated.params, current.observation_norm)
    evaluation_obs = normalize_observations(current.observation_norm, before[0][0])
    distribution, _ = policy.apply({'params':updated.params}, evaluation_obs)
    np.testing.assert_allclose(before[2][0], jnp.tanh(distribution.mean()), atol=1e-6)
    payload = {'ppo':updated, 'controller':current}
    store = CheckpointStore(tmp_path/'normalization_checkpoint')
    store.save(123, payload, recipe.resolved())
    restored = store.restore(payload)
    for a, b in zip(jax.tree.leaves(payload), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)
    after = evaluate(restored['ppo'].params, restored['controller'].observation_norm)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_array_equal(a, b)


def test_learning_rate_full_budget_and_disabled_behavior():
    recipe = normalized_recipe()
    total = recipe.updates*16
    np.testing.assert_allclose(controller_learning_rate(recipe.ppo, 0, total), 3e-4)
    np.testing.assert_allclose(controller_learning_rate(recipe.ppo, total/2, total), 1.5e-4)
    assert controller_learning_rate(recipe.ppo, total, total) == 0
    assert controller_learning_rate(ControllerPPOConfig(), total, total) == 1e-3


def test_resumed_config_without_reward_and_decoder_keys_keeps_their_defaults():
    from empowerment_warp.training.baselines.runner import behavior_config
    saved = normalized_recipe().resolved()
    explicit = behavior_config(saved)
    saved.pop('exponentiate_empowerment')
    saved.pop('vector_gce_bound_action_decoder')
    assert behavior_config(saved) == explicit
    assert behavior_config(dict(saved, exponentiate_empowerment=False)) != explicit
    assert behavior_config(dict(saved, vector_gce_bound_action_decoder=True)) != explicit
