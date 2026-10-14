"""Independent value, derivative, and learner checks for each environment's prior."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import hyp0f1, ive

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.utils.objectives import sphere4_log_partition, sphere_log_partition


@pytest.mark.parametrize('dimension', [2, 4, 9, 16])
def test_spherical_normalizer_values_gradients_and_zero_curvature(dimension):
    radius = np.unique(np.r_[0, np.logspace(-7, 6, 1000), 19.999, 20, 20.001]).astype(np.float32)
    delta = np.zeros((len(radius), dimension), dtype=np.float32)
    delta[:, 0] = radius
    value = jax.jit(lambda x: sphere_log_partition(x, 1.0))(jnp.asarray(delta))
    grad = jax.jit(jax.grad(lambda x: sphere_log_partition(x, 1.0).sum()))(jnp.asarray(delta))
    k = radius.astype(np.float64)
    order = dimension / 2 - 1
    expected = np.zeros_like(k)
    small = k < 1
    expected[small] = np.log(hyp0f1(dimension / 2, k[small]**2 / 4))
    large = ~small
    expected[large] = (k[large] + np.log(ive(order, k[large])) + math.lgamma(dimension / 2)
                       + order * (np.log(2) - np.log(k[large])))
    expected_grad = np.zeros_like(delta, dtype=np.float64)
    expected_grad[1:, 0] = ive(order + 1, k[1:]) / ive(order, k[1:])
    np.testing.assert_allclose(value, expected, atol=6e-6, rtol=5e-7)
    np.testing.assert_allclose(grad, expected_grad, atol=3e-6, rtol=2e-5)
    hessian = jax.hessian(lambda x: sphere_log_partition(x[None], .1)[0])(jnp.zeros(dimension))
    np.testing.assert_allclose(hessian, np.eye(dimension) * (100 / dimension), atol=1e-5)
    if dimension == 4:
        np.testing.assert_array_equal(value, jax.jit(lambda x: sphere4_log_partition(x, 1.0))(jnp.asarray(delta)))


@pytest.mark.parametrize('dimension', [2, 9, 16])
def test_uncapped_vmf_update_preserves_raw_reward_and_avoids_infonce(dimension, monkeypatch):
    from empowerment_warp.agents import skill_classifier as module

    def forbidden(*args, **kwargs):
        raise AssertionError('forward-vMF must not call symmetric InfoNCE')

    config = SkillConfig(method='potential', observation_dim=3, skill_dim=dimension,
                         discrete_skills=False, hidden_dims=(16, 16), phi_norm_cap=0,
                         continuous_reward_temperature=True, reward_encoder='live',
                         encoder_objective='vmf_forward')
    learner = SkillLearner(config)
    control = SkillLearner(replace(config, encoder_objective='symmetric_infonce'))
    state = learner.initialize(jax.random.PRNGKey(1))
    obs = jax.random.normal(jax.random.PRNGKey(2), (8, 3))
    future = jax.random.normal(jax.random.PRNGKey(3), (8, 3))
    skills = learner.sample_skills(jax.random.PRNGKey(4), 8)
    batch = SkillBatch(obs, future, future, obs, skills, jnp.ones(8, dtype=bool))
    key = jax.random.PRNGKey(5)
    np.testing.assert_array_equal(learner.intrinsic_reward(state, batch, key),
                                  control.intrinsic_reward(state, batch, key))
    # Patch the names where the caller (agents.skill_classifier) looks them up; the counted
    # positive control fails if that lookup moves and the forbidden patch goes stale.
    calls = []
    real_vmf = module.vmf_forward_loss
    def counted_vmf(*args, **kwargs):
        calls.append(1)
        return real_vmf(*args, **kwargs)
    monkeypatch.setattr(module, 'symmetric_infonce', forbidden)
    monkeypatch.setattr(module, 'vmf_forward_loss', counted_vmf)
    updated, metrics = learner.update(state, batch, key)
    assert len(calls) > 0
    assert np.isfinite(float(metrics.loss))
    np.testing.assert_allclose(metrics.mi_lower_bound, metrics.alignment - metrics.log_partition, atol=1e-6)
    assert any(np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(
        jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(updated.params)))
