"""Exact pair-vMF likelihood and matching branch/live evaluation schedules."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.utils.objectives import sphere_log_partition
from empowerment_warp.training.sac.config import default_recipe
from empowerment_warp.training.sac.evaluation import evaluation_switch_steps


@pytest.mark.parametrize('dimension', [2, 9, 16])
def test_pair_vmf_reward_loss_and_active_parameter_gradients(dimension):
    config = SkillConfig(
        method='potential', observation_dim=3, skill_dim=dimension,
        discrete_skills=False, num_skills=0, hidden_dims=(8,),
        pair_form='pair_mlp', phi_norm_cap=0, phi_noise_std=0,
        encoder_objective='vmf_forward', continuous_reward_normalization='sphere',
        continuous_reward_temperature=True, reward_encoder='live',
    )
    learner = SkillLearner(config)
    state = learner.initialize(jax.random.PRNGKey(0))
    anchor = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    outcome = jax.random.normal(jax.random.PRNGKey(2), (8, 3))
    skills = learner.sample_skills(jax.random.PRNGKey(3), 8)
    batch = SkillBatch(anchor, outcome, outcome, anchor, skills, jnp.ones(8, bool))
    vector = learner.displacement(state.params, anchor, outcome)
    expected = (vector * skills).sum(-1) / config.temperature - sphere_log_partition(vector, config.temperature)
    np.testing.assert_allclose(learner.intrinsic_reward(state, batch, jax.random.PRNGKey(4)), expected, atol=1e-5)
    loss, _ = learner._loss(state.params, state.log_dual, state, batch,
                            jax.random.PRNGKey(5), apply_phi_noise=False)
    np.testing.assert_allclose(loss, -expected.mean(), atol=1e-5)
    grads = jax.grad(lambda p: learner._loss(p, state.log_dual, state, batch,
                     jax.random.PRNGKey(5), apply_phi_noise=False)[0])(state.params)
    assert sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(grads['pair'])) > 0
    # Existing pair decoder deliberately does not consume its per-state phi trunk.
    assert sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(grads['phi'])) == 0
    raw = SkillLearner(replace(config, continuous_reward_normalization='none'))
    np.testing.assert_allclose(raw.intrinsic_reward(state, batch, jax.random.PRNGKey(4)),
                               (vector * skills).sum(-1) / config.temperature, atol=1e-5)


def test_branch_encoder_preserves_requested_live_and_eval_prefix():
    marginal = default_recipe('ant', 'potential', skill_start_mode='marginal_branch',
                              allow_reference_deviation=True, representation_episodes=3334)
    selected = replace(marginal, branch_live_two_phase=True)
    assert marginal.resolved_live_potential_steps == 0
    assert selected.resolved_live_potential_steps == selected.potential_steps == 100
    np.testing.assert_array_equal(evaluation_switch_steps(marginal, 4, 0), np.zeros(4))
    np.testing.assert_array_equal(evaluation_switch_steps(selected, 4, 0), np.full(4, 100))
    assert selected.resolved_branch_horizon == marginal.resolved_branch_horizon == 300
    with pytest.raises(ValueError, match='encoder-only marginal branches'):
        replace(selected, skill_start_mode='two_phase')
