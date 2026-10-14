"""Compiled loops preserve the former unrolled likelihood, sampling and replay."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.baselines.config import EnvironmentConfig, EstimatorConfig
from empowerment_warp.training.baselines.replay import (
    empty_probe_buffer, make_replay_epoch, train_replay,
)
from empowerment_warp.agents.vim import (
    VIM_METRICS,
    create_vim_train_states, make_vim_train_step,
)
from empowerment_warp.utils.networks import AutoregressiveGaussian, _tanh_log_prob


@pytest.fixture(autouse=True)
def full_precision_parity():
    # Keep TF32 kernel selection from obscuring the loop-equivalence check.
    # Production precision defaults are not changed.
    with jax.default_matmul_precision('highest'):
        yield


class UnrolledGaussian(AutoregressiveGaussian):
    """Pre-scan implementation, kept only as a numerical regression oracle."""

    def __call__(self, context, actions, pre_tanh_actions):
        prefix = jnp.zeros_like(actions)
        log_probabilities = []
        for step_index in range(self.horizon):
            distribution = self._distribution(context, prefix, step_index)
            log_probabilities.append(_tanh_log_prob(distribution, pre_tanh_actions[:, step_index, :]))
            prefix = prefix.at[:, step_index, :].set(actions[:, step_index, :])
        return jnp.sum(jnp.stack(log_probabilities, axis=1), axis=1)

    def sample(self, context, key):
        actions = jnp.zeros((len(context), self.horizon, self.action_dim), context.dtype)
        latent = jnp.zeros_like(actions)
        log_probabilities = []
        for step_index in range(self.horizon):
            key, sample_key = jax.random.split(key)
            distribution = self._distribution(context, actions, step_index)
            value = distribution.sample(seed=sample_key)
            log_probabilities.append(_tanh_log_prob(distribution, value))
            actions = actions.at[:, step_index, :].set(jnp.tanh(value))
            latent = latent.at[:, step_index, :].set(value)
        return actions, latent, jnp.sum(jnp.stack(log_probabilities, axis=1), axis=1)


def assert_tree_close(actual, expected, *, atol=3e-5):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=atol)


@pytest.mark.parametrize('initial_std', [None, .35])
def test_scan_matches_unrolled_values_gradients_and_rng(initial_std):
    models = [cls(5, 2, 8, 2, -5., 2., initial_std) for cls in
              (AutoregressiveGaussian, UnrolledGaussian)]
    key = jax.random.PRNGKey(21)
    context = jax.random.normal(key, (3, 4))
    latent = .4 * jax.random.normal(jax.random.PRNGKey(22), (3, 5, 2))
    actions = jnp.tanh(latent)
    variables = [model.init(key, context, actions, latent) for model in models]
    assert jax.tree.structure(variables[0]) == jax.tree.structure(variables[1])
    for actual, expected in zip(jax.tree.leaves(variables[0]), jax.tree.leaves(variables[1]), strict=True):
        np.testing.assert_array_equal(actual, expected)
    # Exercise nonzero source heads and dependence on the entire action prefix.
    params = jax.tree.map(lambda x: x + .05 * jax.random.normal(key, x.shape), variables[0]['params'])
    results = []
    samples = []
    sample_grads = []
    for model in models:
        loss = lambda p, c, a, u: model.apply({'params': p}, c, a, u).sum()
        results.append(jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))(
            params, context, actions, latent))
        sample = lambda p: model.apply({'params': p}, context, key, method=model.sample)
        samples.append(jax.jit(sample)(params))
        sample_grads.append(jax.jit(jax.grad(lambda p: sample(p)[0].sum()))(params))
    assert_tree_close(*results)
    assert_tree_close(*samples)
    assert_tree_close(*sample_grads)
    jaxpr = jax.make_jaxpr(lambda p: models[0].apply({'params': p}, context, actions, latent))(params)
    assert any(eqn.primitive.name == 'scan' for eqn in jaxpr.jaxpr.eqns)


@pytest.mark.parametrize('method', ['vim'])
@pytest.mark.parametrize('count', [3, 9])
def test_compiled_replay_matches_host_passes(method, count):
    spec = EnvironmentConfig('ant_soccer', 4, 2, 1., 8, 3, 3, .99, 2)
    cfg = EstimatorConfig(hidden_dim=8, scalar_hidden_dim=4, hidden_layers=1,
                          buffer_size=9, batch_size=4, replay_passes=2)
    key = jax.random.PRNGKey(7)
    obs = jax.random.normal(key, (9, 4))
    source, source_state, decoder, decoder_state = create_vim_train_states(key, key, spec, cfg)
    state = source_state, decoder_state
    update, names = make_vim_train_step(source, decoder), VIM_METRICS
    actions, latent, _ = source.apply({'params': source_state.params}, obs, key, method=source.sample)
    buffer = empty_probe_buffer(spec, cfg)._replace(
        observations=obs, actions=actions, pre_tanh_actions=latent,
        targets=obs[:, :2] + .1, count=jnp.asarray(count))
    epoch = make_replay_epoch(update, names, cfg)
    actual, metrics, actual_key = train_replay(epoch, state, buffer, key, cfg.replay_passes)
    expected, expected_key = state, key
    for _ in range(cfg.replay_passes):
        expected_key, epoch_key = jax.random.split(expected_key)
        expected, expected_metrics = epoch(expected, buffer, epoch_key)
    assert_tree_close(jax.tree.leaves(actual), jax.tree.leaves(expected))
    assert_tree_close({name: metrics[name] for name in expected_metrics}, expected_metrics)
    np.testing.assert_array_equal(actual_key, expected_key)
    assert int(metrics['gradient_minibatches']) == cfg.replay_passes * ((count + 3) // 4)
