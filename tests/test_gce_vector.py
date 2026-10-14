"""Learned vector GCE channel: loss, gradients, query and resolved recipe."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from empowerment_warp.training.baselines.config import EnvironmentConfig, EstimatorConfig
from empowerment_warp.agents.latent_gce import (
    VECTOR_GCE_METRICS, create_vector_gce_train_state, identity_moments,
    make_vector_gce_train_step, replay_moments, standardize,
    vector_capacity_from_observations, vector_losses,
)
from empowerment_warp.training.baselines.config import VECTOR_GCE_MIN_STD
from empowerment_warp.training.baselines.replay import empty_probe_buffer, insert_probe_batch
from empowerment_warp.training.baselines.recipe import Recipe
from empowerment_warp.training.baselines.replay import ProbeBatch


@pytest.mark.parametrize('target_dim', [42, 2])
def test_learned_vector_channel_and_loss(target_dim):
    spec = EnvironmentConfig('ant', 42, 8, 1., 400, 3, 3, .99, target_dim)
    cfg = EstimatorConfig(hidden_dim=16, hidden_layers=1, learning_rate=1e-4)
    network, state = create_vector_gce_train_state(
        jax.random.PRNGKey(0), spec, cfg, latent_dim=4)
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(1), 3)
    observations = jax.random.normal(key1, (4, 42))
    actions = jax.random.uniform(key2, (4, 3, 8), minval=-1, maxval=1)
    targets = jax.random.normal(key3, (4, target_dim))
    mask = jnp.array([1., 1., 0., 1.])
    rows = network.apply({'params': state.params}, observations, actions, targets)
    total, components = vector_losses(network, state.params, observations, actions, targets, mask)
    def mean(name):
        return jnp.sum(rows[name] * mask) / jnp.sum(mask)
    expected_regularization = (jnp.abs(1 - mean('context_squared'))
                               + jnp.abs(1 - mean('action_squared')))
    if target_dim != 42:
        expected_regularization += jnp.abs(1 - mean('target_squared'))
    expected = (mean('latent_prediction') + 500 * mean('observation_reconstruction')
                + 500 * mean('prediction_reconstruction')
                + 500 * mean('target_reconstruction')
                + 10 * mean('action_reconstruction') + expected_regularization)
    np.testing.assert_allclose(total, expected, rtol=1e-5)
    np.testing.assert_allclose(components[-1], expected_regularization, rtol=1e-5)
    params = state.params
    gradients = jax.grad(lambda p: vector_losses(
        network, p, observations, actions, targets, mask)[0])(params)
    for module in ('context_encoder', 'context_decoder',
                   'action_encoder', 'action_decoder', 'matrix_network'):
        assert float(optax.global_norm(gradients[module])) > 0, module
    if target_dim != 42:
        assert float(optax.global_norm(gradients['target_encoder'])) > 0
        assert float(optax.global_norm(gradients['target_decoder'])) > 0
    batch = ProbeBatch(observations, actions, jnp.arctanh(actions), targets)
    updated, metrics = jax.jit(make_vector_gce_train_step(network))(state, batch, mask)
    assert int(updated.step) == 1
    assert metrics.shape == (len(VECTOR_GCE_METRICS),)
    assert np.isfinite(np.asarray(metrics)).all()
    capacities, singular_values, _, _ = vector_capacity_from_observations(
        network, updated.params, observations)
    assert capacities.shape == (4,) and singular_values.shape == (4, 4)
    assert np.isfinite(np.asarray(capacities)).all()


def test_vector_recipe_is_a_distinct_arm_with_source_lr():
    vector = Recipe('ant', 'gce_vector', 30, 500_000_000, 'unused', num_envs=256)
    vim = Recipe('ant', 'vim', 30, 500_000_000, 'unused', num_envs=256)
    assert vector.resolved()['estimator']['learning_rate'] == 1e-4
    assert vim.resolved()['estimator']['learning_rate'] == 3e-4
    assert vector.resolved()['effective_scale'] == vim.resolved()['effective_scale']
    assert vector.resolved()['reward'] == 'exp(capacity_bits(current))'
    assert vector.resolved()['vector_gce_latent_dim'] == 32
    # Only the vector estimator and VIM are supported.
    with pytest.raises(ValueError, match="'gce_vector' or 'vim'"):
        Recipe('ant', 'gce', 30, 500_000_000, 'unused', num_envs=256)


def test_bounded_action_decoder_and_training_loss():
    spec = EnvironmentConfig('ant', 42, 8, 1., 400, 3, 3, .99, 42)
    cfg = EstimatorConfig(hidden_dim=16, hidden_layers=1, learning_rate=1e-4)
    model, state = create_vector_gce_train_state(
        jax.random.PRNGKey(5), spec, cfg, latent_dim=4, bound_action_decoder=True)
    # Force saturation so a linear decoder cannot accidentally satisfy the test.
    state.params['action_decoder']['Dense_2']['bias'] = jnp.full(24, 20.)
    latent = jnp.zeros((4, 4))
    decoded = jax.jit(lambda p: model.apply({'params': p}, latent, method=model.decode_action))(state.params)
    np.testing.assert_allclose(decoded, jnp.tanh(jnp.full((4, 24), 20.)))
    obs, actions = jnp.ones((4, 42)), jnp.zeros((4, 3, 8))
    batch = ProbeBatch(obs, actions, actions, obs)
    _, metrics = jax.jit(make_vector_gce_train_step(model))(state, batch, jnp.ones(4))
    assert np.isfinite(np.asarray(metrics)).all()
    np.testing.assert_allclose(metrics[VECTOR_GCE_METRICS.index('action_reconstruction_loss')], 1.)


def test_replay_moments_use_valid_rows_and_floor_constant_outcomes():
    spec = EnvironmentConfig('ant_soccer', 42, 8, 1., 400, 3, 3, .99, 2)
    buffer = empty_probe_buffer(spec, EstimatorConfig(buffer_size=8))
    empty = replay_moments(buffer)
    for a, b in zip(empty, identity_moments(spec)):
        np.testing.assert_array_equal(a, b)
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    observations = 20. + 3. * jax.random.normal(key1, (5, 42))
    actions = jax.random.uniform(key2, (5, 3, 8), minval=-1, maxval=1)
    # A ball that never moves: constant outcome across every stored probe.
    targets = jnp.tile(jnp.array([[10., 10.]]), (5, 1))
    buffer = insert_probe_batch(buffer, ProbeBatch(observations, actions, actions, targets),
                                jax.random.PRNGKey(1))
    moments = replay_moments(buffer)
    np.testing.assert_allclose(moments.observation_mean, observations.mean(0), rtol=1e-5)
    np.testing.assert_allclose(moments.observation_std,
                               np.maximum(observations.std(0), VECTOR_GCE_MIN_STD), rtol=1e-4)
    np.testing.assert_allclose(moments.target_mean, [10., 10.], rtol=1e-6)
    np.testing.assert_allclose(moments.target_std, [VECTOR_GCE_MIN_STD]*2, rtol=1e-6)
    standardized, standardized_targets = standardize(moments, observations, targets)
    np.testing.assert_allclose(standardized.mean(0), 0., atol=1e-4)
    np.testing.assert_allclose(standardized_targets, 0., atol=1e-6)


def test_input_normalization_is_explicit_and_vector_only():
    assert not Recipe('ant', 'gce_vector', 30, 500_000_000, 'unused').vector_gce_normalize_inputs
    resolved = Recipe('ant', 'gce_vector', 30, 500_000_000, 'unused',
                      vector_gce_normalize_inputs=True).resolved()
    assert resolved['vector_gce_inputs'].startswith('replay-standardized')
    with pytest.raises(ValueError, match='only to gce_vector'):
        Recipe('ant', 'vim', 30, 500_000_000, 'unused', vector_gce_normalize_inputs=True)


def test_shared_encoder_uses_pooled_start_and_outcome_coordinates():
    spec = EnvironmentConfig('ant', 42, 8, 1., 400, 3, 3, .99, 42)
    config = EstimatorConfig(hidden_dim=16, hidden_layers=1, buffer_size=8)
    observations = jnp.arange(4*42, dtype=jnp.float32).reshape((4, 42)) / 42
    targets = observations + 20.
    actions = jnp.full((4, 3, 8), .2)
    buffer = insert_probe_batch(empty_probe_buffer(spec, config),
        ProbeBatch(observations, actions, actions, targets), jax.random.PRNGKey(2))
    moments = replay_moments(buffer)
    pooled = jnp.concatenate((observations, targets))
    np.testing.assert_allclose(moments.observation_mean, pooled.mean(0), rtol=1e-6)
    np.testing.assert_allclose(moments.observation_std, pooled.std(0), rtol=1e-6)
    np.testing.assert_array_equal(moments.observation_mean, moments.target_mean)
    np.testing.assert_array_equal(moments.observation_std, moments.target_std)
    normalized, normalized_targets = standardize(moments, observations, targets)
    _, same_coordinates = standardize(moments, observations, observations)
    np.testing.assert_array_equal(normalized, same_coordinates)
    network, state = create_vector_gce_train_state(jax.random.PRNGKey(3), spec, config, latent_dim=4)
    batch = ProbeBatch(normalized, actions, actions, normalized_targets)
    state, metrics = jax.jit(make_vector_gce_train_step(network))(state, batch, jnp.ones(4))
    assert int(state.step) == 1 and np.isfinite(np.asarray(metrics)).all()
    capacity, *_ = vector_capacity_from_observations(network, state.params, normalized)
    assert np.isfinite(np.asarray(capacity)).all()
