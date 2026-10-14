"""Forward-only classification must use skill candidates, with no reverse gradient."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.utils.objectives import forward_infonce


@pytest.mark.parametrize('dimension', [2, 4])
def test_forward_learner_matches_independent_row_cross_entropy_and_gradient(dimension):
    config = SkillConfig(method='potential', observation_dim=3, skill_dim=dimension,
                         discrete_skills=False, hidden_dims=(16, 16),
                         encoder_objective='forward_infonce', phi_noise_std=0,
                         phi_norm_cap=0, continuous_reward_temperature=True,
                         reward_encoder='live')
    learner = SkillLearner(config)
    state = learner.initialize(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    future = obs + jax.random.normal(jax.random.PRNGKey(2), (8, 3))
    z = jax.random.normal(jax.random.PRNGKey(3), (8, dimension))
    z /= jnp.linalg.norm(z, axis=-1, keepdims=True)
    valid = jnp.array([True] * 6 + [False] * 2)
    batch = SkillBatch(obs, future, future, obs, z, valid)
    key = jax.random.PRNGKey(4)

    def reference(params):
        delta = learner.displacement(params, obs, future)[:6]
        logits = delta @ z[:6].T / config.temperature
        return -jnp.diag(jax.nn.log_softmax(logits, axis=1)).mean()

    actual = lambda params: learner._loss(params, state.log_dual, state, batch, key,
                                         apply_phi_noise=False)[0]
    loss, metrics = learner._loss(state.params, state.log_dual, state, batch, key,
                                  apply_phi_noise=False)
    np.testing.assert_allclose(loss, reference(state.params), rtol=2e-6)
    for a, b in zip(jax.tree_util.tree_leaves(jax.grad(actual)(state.params)),
                    jax.tree_util.tree_leaves(jax.grad(reference)(state.params))):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.mi_lower_bound, jnp.log(6.) - loss, atol=1e-6)
    np.testing.assert_allclose(metrics.alignment - metrics.log_partition,
                               metrics.mi_lower_bound, atol=1e-6)
    vmf = SkillLearner(replace(config, encoder_objective='vmf_forward'))
    np.testing.assert_array_equal(learner._intrinsic_impl(state, batch, key),
                                  vmf._intrinsic_impl(state, batch, key))
    next_state, updated_metrics = learner._update_jit(state, batch, key)
    jax.block_until_ready(next_state)
    assert np.isfinite(np.asarray(updated_metrics)).all()
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(state.params),
                                                       jax.tree_util.tree_leaves(next_state.params)))


def test_forward_masks_invalid_candidates():
    d = jnp.array([[2., 0.], [1., .2], [9., 9.]])
    z = jnp.array([[1., 0.], [0., 1.], [1., 0.]])
    valid = jnp.array([True, True, False])
    fwd = forward_infonce(d, z, temperature=.3, valid=valid)
    np.testing.assert_allclose(fwd.loss, forward_infonce(d[:2], z[:2], temperature=.3).loss)
    grad = jax.grad(lambda x: forward_infonce(x, z, temperature=.3, valid=valid).loss)(d)
    np.testing.assert_array_equal(grad[-1], jnp.zeros(2))


def test_forward_rejects_unsupported_method():
    with pytest.raises(ValueError, match='unsupported skill method'):
        SkillConfig(method='metra', observation_dim=3, skill_dim=4,
                    discrete_skills=False, encoder_objective='forward_infonce')


def test_symmetric_continuous_potential_is_rejected():
    assert SkillConfig(method='potential', observation_dim=3, skill_dim=2,
                       discrete_skills=False).encoder_objective == 'forward_infonce'
    with pytest.raises(ValueError, match='symmetric continuous-skill CSF is not supported'):
        SkillConfig(method='potential', observation_dim=3, skill_dim=2,
                    discrete_skills=False, encoder_objective='symmetric_infonce')
