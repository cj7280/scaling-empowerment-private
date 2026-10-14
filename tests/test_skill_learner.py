from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner, sample_skills


def _batch(config: SkillConfig, batch_size: int = 16) -> SkillBatch:
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, config.observation_dim))
    return SkillBatch(
        observations=obs,
        next_observations=obs + 0.1 * jax.random.normal(jax.random.PRNGKey(1), obs.shape),
        future_observations=obs + 0.5 * jax.random.normal(jax.random.PRNGKey(2), obs.shape),
        anchor_observations=jax.random.normal(jax.random.PRNGKey(3), obs.shape),
        skills=sample_skills(config, jax.random.PRNGKey(4), batch_size),
        future_valid=jnp.ones((batch_size,), dtype=jnp.bool_),
    )


@pytest.mark.parametrize(
    "method,discrete",
    [
        ("potential", False),
    ],
)
def test_skill_learner_update_and_reward(method: str, discrete: bool) -> None:
    config = SkillConfig(
        method=method,
        observation_dim=5,
        skill_dim=4 if discrete else 3,
        discrete_skills=discrete,
        num_skills=4 if discrete else 0,
        discrete_skill_library="random",
        hidden_dims=(16,),
        layer_norm=False,
    )
    learner = SkillLearner(config)
    state = learner.initialize(jax.random.PRNGKey(5))
    batch = _batch(config)
    state, metrics = learner.update(state, batch, jax.random.PRNGKey(6))
    rewards = learner.intrinsic_reward(state, batch, jax.random.PRNGKey(7))
    assert int(state.step) == 1
    assert rewards.shape == (16,)
    assert np.isfinite(np.asarray(rewards)).all()
    assert all(np.isfinite(float(value)) for value in metrics)


def test_clean_evaluation_disables_phi_noise_on_the_same_batch() -> None:
    config = SkillConfig(
        method="potential",
        observation_dim=5,
        skill_dim=2,
        discrete_skills=False,
        num_skills=0,
        hidden_dims=(16,),
        layer_norm=False,
        phi_noise_std=1.0,
    )
    learner = SkillLearner(config)
    state = learner.initialize(jax.random.PRNGKey(20))
    batch = learner.preprocess_batch(_batch(config, batch_size=64))

    clean_a = learner.evaluate_preprocessed(
        state,
        batch,
        jax.random.PRNGKey(21),
        apply_phi_noise=False,
    )
    clean_b = learner.evaluate_preprocessed(
        state,
        batch,
        jax.random.PRNGKey(22),
        apply_phi_noise=False,
    )
    augmented_a = learner.evaluate_preprocessed(
        state,
        batch,
        jax.random.PRNGKey(21),
    )
    augmented_b = learner.evaluate_preprocessed(
        state,
        batch,
        jax.random.PRNGKey(22),
    )

    np.testing.assert_array_equal(clean_a.loss, clean_b.loss)
    np.testing.assert_array_equal(clean_a.accuracy, clean_b.accuracy)
    assert not np.isclose(float(augmented_a.loss), float(augmented_b.loss))


def test_delta_reward_reference_telescopes_to_the_anchor_form() -> None:
    """`current` is a per-step delta whose RETURN equals the anchor form.

    sum_t <phi(s_t+1) - phi(s_t), z> = <phi(s_T) - phi(s_0), z>, so walking a
    trajectory and summing the delta reward must reproduce a single anchor-referenced
    reward taken from s_0 to s_T. Only true for the RAW inner product.
    """

    def make(reference: str) -> SkillLearner:
        return SkillLearner(
            SkillConfig(
                method="potential",
                observation_dim=4,
                skill_dim=2,
                discrete_skills=False,
                num_skills=0,
                hidden_dims=(16,),
                layer_norm=False,
                potential_reward_reference=reference,
            )
        )

    delta_learner, anchor_learner = make("current"), make("anchor")
    config = delta_learner.config
    state = delta_learner.initialize(jax.random.PRNGKey(21))
    steps = 5
    traj = jax.random.normal(jax.random.PRNGKey(22), (steps + 1, 3, config.observation_dim))
    skills = sample_skills(config, jax.random.PRNGKey(23), 3)

    def batch(obs, nxt, anchor):
        return SkillBatch(
            observations=obs, next_observations=nxt, future_observations=nxt,
            anchor_observations=anchor, skills=skills,
            future_valid=jnp.ones((3,), dtype=jnp.bool_),
        )

    per_step = sum(
        np.asarray(delta_learner.intrinsic_reward(state, batch(traj[t], traj[t + 1], traj[0]),
                                                  jax.random.PRNGKey(24)))
        for t in range(steps)
    )
    endpoint = np.asarray(
        anchor_learner.intrinsic_reward(state, batch(traj[0], traj[steps], traj[0]),
                                        jax.random.PRNGKey(24))
    )
    np.testing.assert_allclose(per_step, endpoint, rtol=1e-4, atol=1e-4)


def test_delta_reward_reference_changes_the_reward() -> None:
    """Guard against the flag being inert."""

    def reward(reference: str) -> float:
        config = SkillConfig(
            method="potential", observation_dim=4, skill_dim=2, discrete_skills=False,
            num_skills=0, hidden_dims=(16,), layer_norm=False,
            potential_reward_reference=reference,
        )
        learner = SkillLearner(config)
        state = learner.initialize(jax.random.PRNGKey(25))
        obs = jax.random.normal(jax.random.PRNGKey(26), (8, config.observation_dim))
        b = SkillBatch(
            observations=obs,
            next_observations=obs + 0.5,
            future_observations=obs + 0.5,
            anchor_observations=obs - 3.0,  # far from `observations`
            skills=sample_skills(config, jax.random.PRNGKey(27), 8),
            future_valid=jnp.ones((8,), dtype=jnp.bool_),
        )
        return float(jnp.mean(learner.intrinsic_reward(state, b, jax.random.PRNGKey(28))))

    assert abs(reward("current") - reward("anchor")) > 1e-4
