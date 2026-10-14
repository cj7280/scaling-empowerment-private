"""Density, entropy-gradient and PPO regressions for the bounded controller."""
from dataclasses import replace

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.training.baselines.config import EnvironmentConfig, PPOConfig
from empowerment_warp.agents.ppo import (
    ControllerState, RolloutBatch, calculate_gae, create_ppo_train_state,
    empty_observation_norm, make_ppo_update, policy_action,
    squashed_entropy, squashed_log_prob,
)
from empowerment_warp.recipes import resolve


def test_squashed_density_matches_change_of_variables_and_ppo_ratio():
    old = distrax.MultivariateNormalDiag(jnp.array([0.2, -0.3]), jnp.array([0.8, 1.1]))
    new = distrax.MultivariateNormalDiag(jnp.array([0.4, -0.1]), jnp.array([0.9, 0.7]))
    raw = jnp.array([[0.3, -0.5], [2.0, -2.0]])
    # Independent float64 reference avoids cancellation in float32 log(1-tanh²).
    jacobian = np.sum(np.log1p(-np.tanh(np.asarray(raw, dtype=np.float64)) ** 2), axis=-1)
    np.testing.assert_allclose(squashed_log_prob(old, raw), old.log_prob(raw) - jacobian, atol=2e-6)
    np.testing.assert_allclose(
        squashed_log_prob(new, raw) - squashed_log_prob(old, raw),
        new.log_prob(raw) - old.log_prob(raw), atol=2e-6)
    saturated = jnp.array([[20.0, -20.0], [100.0, -100.0]])
    assert np.isfinite(squashed_log_prob(old, saturated)).all()
    assert np.isfinite(jax.grad(lambda x: squashed_log_prob(old, x).sum())(saturated)).all()


def test_squashed_entropy_and_gradient_penalize_excessive_noise():
    # Monte Carlo agrees with independent Gaussian quadrature at ordinary scale.
    key = jax.random.PRNGKey(91)
    shape = (16384, 2)
    def entropy(log_std):
        distribution = distrax.MultivariateNormalDiag(jnp.zeros(shape), jnp.full(shape, jnp.exp(log_std)))
        return squashed_entropy(distribution, key).mean()
    value, gradient = jax.jit(jax.value_and_grad(entropy))(jnp.array(0.0))
    nodes, weights = np.polynomial.hermite.hermgauss(100)
    raw = np.sqrt(2.0) * nodes
    log_jacobian = 2.0 * (np.log(2.0) - raw - np.logaddexp(0.0, -2.0 * raw))
    expected = 2 * (0.5 * np.log(2 * np.pi * np.e) + np.sum(weights * log_jacobian) / np.sqrt(np.pi))
    expected_grad = 2 * (1 - 2 * np.sum(weights * raw * np.tanh(raw)) / np.sqrt(np.pi))
    np.testing.assert_allclose(value, expected, atol=0.04)
    np.testing.assert_allclose(gradient, expected_grad, atol=0.05)
    high, high_gradient = jax.jit(jax.value_and_grad(entropy))(jnp.array(2.0))
    assert high < value and high_gradient < 0
    assert value < 2 * np.log(2) + 0.04


def test_real_ppo_update_and_eval_use_bounded_policy():
    spec = EnvironmentConfig('ant', 42, 8, 1., 400, 100, 100, .99, 42)
    config = PPOConfig(num_envs=4, rollout_steps=4)
    network, state = create_ppo_train_state(jax.random.PRNGKey(0), spec, config)
    observations = jnp.zeros((4, 4, 42))
    distribution, values = network.apply({'params': state.params}, observations)
    raw = distribution.sample(seed=jax.random.PRNGKey(1))
    batch = RolloutBatch(observations, observations, raw, squashed_log_prob(distribution, raw),
                         values, jnp.zeros((4, 4), dtype=bool), observations,
                         jnp.zeros((4, 4), dtype=bool))
    controller = ControllerState(None, observations[-1], empty_observation_norm(42))
    update = make_ppo_update(network, spec, config)
    updated, metrics, *_ = update(state, batch, jnp.zeros((4, 4)), controller, jax.random.PRNGKey(2))
    assert int(updated.step) == 16
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    assert metrics['entropy'] < 8 * np.log(2) + 0.5
    # Zero advantage isolates entropy: initially unit std is too wide after tanh.
    assert np.mean(updated.params['log_std']) < np.mean(state.params['log_std'])
    for deterministic in (False, True):
        action, entropy = policy_action(network, updated.params, observations[-1],
                                        jax.random.PRNGKey(3), deterministic=deterministic)
        assert np.isfinite(action).all() and np.isfinite(entropy).all()
        assert np.max(np.abs(action)) <= 1
        if deterministic:
            normal, _ = network.apply({'params': updated.params}, observations[-1])
            np.testing.assert_allclose(action, jnp.tanh(normal.mean()), atol=1e-6)


def test_gae_still_cuts_at_done():
    rewards = jnp.zeros((4, 1))
    dones = jnp.array([[False], [True], [False], [False]])
    def advantages(r):
        return calculate_gae(r, dones, rewards, rewards, dones, gamma=.99, gae_lambda=.95)[0]
    np.testing.assert_array_equal(advantages(rewards)[:2], advantages(rewards.at[2:].set(100.))[:2])


def test_production_and_smoke_configs_only_change_stop_and_path():
    _, recipe = resolve("configs/ant/vim.yaml", run_dir="unused", overrides=[
        "horizon=100", "num_envs=256", "rollout_steps=80", "episode_steps=400",
        "eval_every_updates=100", "checkpoint_keep=2", "total_env_steps=500000000",
    ])
    production = recipe.resolved()
    smoke = replace(recipe, max_updates=2).resolved()
    assert production['controller_distribution'] == 'tanh_normal'
    assert production['ppo']['entropy_coefficient'] == .01
    scale = production['effective_scale']
    assert scale['controller_steps_per_update'] == 20480
    assert scale['probe_steps_per_update'] == 25600
    assert scale['ppo_samples_per_controller_transition'] == 4
    smoke['max_updates'] = 0
    smoke['effective_scale']['smoke_max_updates'] = 0
    assert production == smoke
