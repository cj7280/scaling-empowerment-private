"""Check the analytic spherical likelihood independently of the contrastive code."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ive

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.utils.objectives import (
    in_batch_skill_accuracy, sphere4_log_partition, forward_infonce, vmf_forward_loss,
)


def test_sphere4_values_gradients_and_zero_curvature():
    kappa = np.array([0, 1e-7, 0.001, 0.02, 0.5, 0.999, 1, 1.001, 4, 20, 100, 1e3, 1e4, 1e5, 1e6], dtype=np.float32)
    delta = jnp.asarray(kappa[:, None] * np.array([[1, 0, 0, 0]], dtype=np.float32))
    expected = np.zeros_like(kappa, dtype=np.float64)
    nonzero = kappa[1:].astype(np.float64)
    expected[1:] = nonzero + np.log(2 * ive(1, nonzero) / nonzero)
    np.testing.assert_allclose(sphere4_log_partition(delta, 1), expected, atol=2e-6, rtol=2e-6)
    grad = jax.jit(jax.grad(lambda x: sphere4_log_partition(x, 1).sum()))(delta)
    expected_grad = np.zeros((len(kappa), 4))
    expected_grad[1:, 0] = ive(2, nonzero) / ive(1, nonzero)
    np.testing.assert_allclose(grad, expected_grad, atol=3e-6, rtol=1e-4)
    hessian = jax.hessian(lambda x: sphere4_log_partition(x[None], 0.1)[0])(jnp.zeros(4))
    np.testing.assert_allclose(hessian, 25 * np.eye(4), atol=1e-5)


def test_forward_masks_and_has_no_negative_sample_dependence():
    delta = jnp.array([[0.2, 0, 0, 0], [0, 0.4, 0, 0], [9, 8, 7, 6]], dtype=jnp.float32)
    skill = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=jnp.float32)
    valid = jnp.array([True, True, False])
    loss, metrics = vmf_forward_loss(delta, skill, temperature=0.1, valid=valid)
    single = [vmf_forward_loss(delta[i:i+1], skill[i:i+1], temperature=0.1)[0] for i in range(2)]
    np.testing.assert_allclose(loss, np.mean(single), atol=1e-6)
    np.testing.assert_allclose(loss, metrics['log_partition'] - metrics['alignment'])
    grad = jax.grad(lambda x: vmf_forward_loss(x, skill, temperature=0.1, valid=valid)[0])(delta)
    np.testing.assert_array_equal(grad[-1], np.zeros(4))


def test_real_encoder_update_and_raw_reward_parity(monkeypatch):
    config = SkillConfig(method='potential', observation_dim=3, skill_dim=4,
                         discrete_skills=False, hidden_dims=(16, 16),
                         continuous_reward_temperature=True, reward_encoder='live')
    old = SkillLearner(config)
    new = SkillLearner(replace(config, encoder_objective='vmf_forward'))
    state = new.initialize(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    future = obs + jax.random.normal(jax.random.PRNGKey(2), (8, 3))
    skills = jax.random.normal(jax.random.PRNGKey(3), (8, 4))
    skills = skills / jnp.linalg.norm(skills, axis=-1, keepdims=True)
    batch = SkillBatch(obs, future, future, obs, skills, jnp.ones(8, dtype=jnp.bool_))
    key = jax.random.PRNGKey(4)
    np.testing.assert_array_equal(new._intrinsic_impl(state, batch, key), old._intrinsic_impl(state, batch, key))
    def forbidden_infonce(*args, **kwargs):
        raise AssertionError('vMF update must not call forward InfoNCE')
    # Patch the names where the caller (agents.skill_classifier) looks them up; the counted
    # positive control fails if that lookup moves and the forbidden patch goes stale.
    from empowerment_warp.agents import skill_classifier as caller
    calls = []
    real_vmf = caller.vmf_forward_loss
    def counted_vmf(*args, **kwargs):
        calls.append(1)
        return real_vmf(*args, **kwargs)
    monkeypatch.setattr(caller, 'forward_infonce', forbidden_infonce)
    monkeypatch.setattr(caller, 'vmf_forward_loss', counted_vmf)
    next_state, metrics = new._update_jit(state, batch, key)
    jax.block_until_ready(next_state)
    assert len(calls) > 0
    assert np.isfinite(np.asarray(metrics)).all()
    np.testing.assert_allclose(metrics.mi_lower_bound, -metrics.loss, atol=1e-6)
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(state.params),
                                                     jax.tree_util.tree_leaves(next_state.params)))
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(next_state))


def test_vmf_rejects_unsupported_skill_geometry():
    with pytest.raises(ValueError, match='continuous 2/4/9/16-D commanded'):
        SkillConfig(method='potential', observation_dim=3, skill_dim=3,
                    discrete_skills=False, encoder_objective='vmf_forward')


@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('dimension', [2, 4])
def test_policy_anchor_and_delta_rewards_against_independent_scipy(normalized, dimension):
    config = SkillConfig(method='potential', observation_dim=3, skill_dim=dimension,
                         discrete_skills=False, hidden_dims=(16, 16),
                         continuous_reward_temperature=True, reward_encoder='live',
                         encoder_objective='vmf_forward',
                         continuous_reward_normalization='sphere' if normalized else 'none')
    anchor_learner = SkillLearner(config)
    delta_learner = SkillLearner(replace(config, potential_reward_reference='current'))
    state = anchor_learner.initialize(jax.random.PRNGKey(10))
    points = jax.random.normal(jax.random.PRNGKey(11), (6, 3))
    anchors = jnp.broadcast_to(points[0], (5, 3))
    direction = jnp.asarray([1., -1.] * (dimension // 2)) / jnp.sqrt(float(dimension))
    skills = jnp.broadcast_to(direction, (5, dimension))
    batch = SkillBatch(points[:-1], points[1:], points[1:], anchors, skills,
                       jnp.ones(5, dtype=jnp.bool_))
    displacement = np.asarray(anchor_learner.displacement(
        state.params, jnp.broadcast_to(points[0], points.shape), points))
    evidence = displacement @ np.asarray(skills[0]) / config.temperature
    if normalized:
        radius = np.linalg.norm(displacement, axis=-1).astype(np.float64) / config.temperature
        correction = np.zeros_like(radius)
        nonzero = radius > 0
        if dimension == 2:
            correction[nonzero] = radius[nonzero] + np.log(ive(0, radius[nonzero]))
        else:
            correction[nonzero] = radius[nonzero] + np.log(2 * ive(1, radius[nonzero]) / radius[nonzero])
        evidence = evidence - correction
    key = jax.random.PRNGKey(12)
    anchor_reward = np.asarray(anchor_learner.intrinsic_reward(state, batch, key))
    delta_reward = np.asarray(delta_learner.intrinsic_reward(state, batch, key))
    np.testing.assert_allclose(anchor_reward, evidence[1:], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(delta_reward, np.diff(evidence), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(delta_reward.sum(), evidence[-1], atol=2e-5)
    # Fixed horizon: discounted equivalence includes the terminal boundary term.
    gamma = 0.99
    weights = gamma ** np.arange(5)
    expected = (1 - gamma) * (weights @ anchor_reward) + gamma**5 * evidence[-1]
    np.testing.assert_allclose(weights @ delta_reward, expected, atol=2e-5)


def test_spherical_policy_reward_rejects_unscaled_logits():
    with pytest.raises(ValueError, match='with temperature'):
        SkillConfig(method='potential', observation_dim=3, skill_dim=4,
                    discrete_skills=False, continuous_reward_normalization='sphere')


def test_accuracy_only_matches_original_with_masking_and_ties():
    delta = jnp.array([[1., 0, 0, 0], [1., 0, 0, 0], [0, 1., 0, 0]])
    for valid in [jnp.array([True, True, True]), jnp.array([False, True, True]), jnp.zeros(3, bool)]:
        expected = forward_infonce(delta, delta, temperature=.1, valid=valid).accuracy
        actual = in_batch_skill_accuracy(delta, delta, temperature=.1, valid=valid)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dimension', [2, 4])
def test_annealed_reward_endpoints_and_midpoint(dimension):
    config = SkillConfig(method='potential', observation_dim=3, skill_dim=dimension,
                         discrete_skills=False, hidden_dims=(16,), phi_norm_cap=0,
                         continuous_reward_temperature=True, reward_encoder='live',
                         encoder_objective='vmf_forward')
    raw = SkillLearner(config)
    normalized = SkillLearner(replace(config, continuous_reward_normalization='sphere'))
    state = raw.initialize(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    future = jax.random.normal(jax.random.PRNGKey(2), (8, 3))
    skills = jnp.broadcast_to(jnp.eye(dimension)[0], (8, dimension))
    batch = SkillBatch(obs, future, future, obs, skills, jnp.ones(8, dtype=jnp.bool_))
    key = jax.random.PRNGKey(3)
    raw_reward = raw.intrinsic_reward(state, batch, key)
    full_reward = normalized.intrinsic_reward(state, batch, key)
    np.testing.assert_array_equal(normalized.intrinsic_reward(state, batch, key, 0.), raw_reward)
    np.testing.assert_array_equal(normalized.intrinsic_reward(state, batch, key, 1.), full_reward)
    np.testing.assert_allclose(normalized.intrinsic_reward(state, batch, key, .5),
                               .5 * (raw_reward + full_reward), atol=2e-6)
    # The policy coefficient never enters encoder optimization.
    raw_state, raw_metrics = raw.update(state, batch, key)
    norm_state, norm_metrics = normalized.update(state, batch, key)
    for a, b in zip(jax.tree_util.tree_leaves(raw_state), jax.tree_util.tree_leaves(norm_state)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(raw_metrics.mi_lower_bound, norm_metrics.mi_lower_bound)
