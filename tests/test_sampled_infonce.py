"""Sampled-partition ablation: independent formulas and real learner paths."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.utils.objectives import sampled_infonce_log_partition
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig


@pytest.mark.parametrize('rows', [1, 17, 257])
def test_partition_matches_explicit_candidates_and_gradients(rows):
    delta = jax.random.normal(jax.random.PRNGKey(0), (rows, 4))
    skills = jnp.broadcast_to(jnp.eye(4)[0], delta.shape)
    negatives = jax.random.normal(jax.random.PRNGKey(1), (15, 4))
    negatives /= jnp.linalg.norm(negatives, axis=-1, keepdims=True)
    logits = np.concatenate([np.sum(delta * skills, axis=-1)[:, None], delta @ negatives.T], axis=1) / .1
    actual = jax.jit(lambda d: sampled_infonce_log_partition(d, skills, negatives, .1))(delta)
    np.testing.assert_allclose(actual, logsumexp(logits, axis=-1) - np.log(16), atol=1e-5)
    actual_grad = jax.grad(lambda d: sampled_infonce_log_partition(d, skills, negatives, .1).sum())(delta)
    expected_grad = jax.grad(lambda d: (jax.nn.logsumexp(jnp.concatenate([
        jnp.sum(d * skills, axis=-1)[:, None], d @ negatives.T], axis=1) / .1, axis=-1) - jnp.log(16.)).sum())(delta)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('dim', [2, 4, 9, 16])
def test_real_reward_anneal_and_encoder_never_call_closed_form(monkeypatch, dim):
    def forbidden(*args, **kwargs):
        raise AssertionError('sampled ablation called the closed-form partition')
    # Patch the names where the caller (agents.skill_classifier) looks them up; the counted
    # positive control fails if that lookup moves and the forbidden patches go stale.
    from empowerment_warp.agents import skill_classifier as caller
    calls = []
    real_sampled = caller.sampled_infonce_log_partition
    def counted_sampled(*args, **kwargs):
        calls.append(1)
        return real_sampled(*args, **kwargs)
    monkeypatch.setattr(caller, 'sphere_log_partition', forbidden)
    monkeypatch.setattr(caller, 'vmf_forward_loss', forbidden)
    monkeypatch.setattr(caller, 'sampled_infonce_log_partition', counted_sampled)
    cfg = SkillConfig(method='potential', observation_dim=3, skill_dim=dim,
        discrete_skills=False, hidden_dims=(8,), phi_norm_cap=0,
        continuous_reward_temperature=True, reward_encoder='live',
        encoder_objective='forward_infonce', continuous_reward_normalization='sampled_infonce')
    learner = SkillLearner(cfg)
    state = learner.initialize(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (17, 3))
    future = jax.random.normal(jax.random.PRNGKey(2), (17, 3))
    skills = learner.sample_skills(jax.random.PRNGKey(3), 17)
    batch = SkillBatch(obs, future, future, obs, skills, jnp.ones(17, bool))
    key = jax.random.PRNGKey(4)
    raw = SkillLearner(replace(cfg, continuous_reward_normalization='none')).intrinsic_reward(state, batch, key)
    full = learner.intrinsic_reward(state, batch, key, 1.)
    np.testing.assert_array_equal(learner.intrinsic_reward(state, batch, key, 0.), raw)
    np.testing.assert_allclose(learner.intrinsic_reward(state, batch, key, .5), (raw + full) / 2, atol=1e-5)
    assert np.max(full) <= np.log(1024) + 1e-5
    assert not np.array_equal(full, learner.intrinsic_reward(state, batch, jax.random.PRNGKey(5)))
    assert len(calls) > 0
    updated, info = learner.update(state, batch, key)
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(updated))
    assert float(info.mi_lower_bound) <= np.log(17) + 1e-5
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(updated.params)))


def test_pqn_real_frozen_reward_uses_sampled_partition():
    from empowerment_warp.agents.see_pqn import PotentialPQNAgent
    from empowerment_warp.agents.see_pqn import make_frozen_reward_fn
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
        num_envs=4, rollout_steps=4, discrete_skills=False, skill_dim=16, potential_skill_samples=2,
        pqn_hidden_size=8, pqn_num_layers=1, encoder_hidden_dims=(8,), encoder_obs_mode='full',
        encoder_objective='forward_infonce', continuous_reward_normalization='sampled_infonce', phi_norm_cap=0,
        reward_partition_anneal_start_steps=600_000_000, reward_partition_anneal_end_steps=1_600_000_000)
    learner = SkillLearner(config.skill_config(observation_dim=3))
    state = learner.initialize(jax.random.PRNGKey(0))
    agent = PotentialPQNAgent.create(0, jnp.zeros((1,3)), jnp.zeros((1,4)), config.agent_config())
    obs = jax.random.normal(jax.random.PRNGKey(1), (17,3))
    skills = learner.sample_skills(jax.random.PRNGKey(2),17).at[0].set(0.)
    fn = make_frozen_reward_fn(learner, config.discount, 1e-6)
    values = [fn(agent,state,obs,obs+.5,obs,skills,jnp.zeros(17),jax.random.PRNGKey(3),partition_weight=jnp.asarray(w))[0] for w in (0.,.5,1.)]
    np.testing.assert_allclose(values[1], (values[0]+values[2])/2, atol=1e-6)
    assert np.any(np.asarray(values[0][1:] - values[2][1:]) != 0)
    assert values[0][0] == values[2][0]
    for step, weight in [(0,0),(600_000_000,0),(1_100_000_000,.5),(1_600_000_000,1)]:
        assert float(config.reward_partition_weight(step)) == weight
