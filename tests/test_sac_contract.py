from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.see_sac import ContinuousSAC, SACConfig
from empowerment_warp.core import Transition


def _batch(*, discrete: bool, batch_size: int = 16) -> Transition:
    key = jax.random.PRNGKey(7)
    obs = jax.random.normal(key, (batch_size, 5))
    actions = (
        jax.random.randint(key, (batch_size,), 0, 4)
        if discrete
        else jax.random.uniform(key, (batch_size, 2), minval=-1.0, maxval=1.0)
    )
    skills = jax.random.normal(jax.random.PRNGKey(9), (batch_size, 3))
    return Transition(
        observations=obs,
        anchor_observations=obs - 0.2,
        actions=actions,
        rewards=jax.random.normal(key, (batch_size,)),
        masks=jnp.ones((batch_size,), dtype=jnp.float32),
        next_observations=obs + 0.1 * jax.random.normal(jax.random.PRNGKey(8), obs.shape),
        next_anchor_observations=obs - 0.2,
        dones=jnp.zeros((batch_size,), dtype=jnp.bool_),
        skills=skills,
        next_skills=skills,
        loss_weights=jnp.ones((batch_size,), dtype=jnp.float32),
        episode_ids=jnp.zeros((batch_size,), dtype=jnp.int32),
        timesteps=jnp.arange(batch_size, dtype=jnp.int32),
    )


def _tree_delta(before, after) -> float:
    leaves = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), before, after)
    return float(sum(jax.tree_util.tree_leaves(leaves)))


def _assert_tree_allclose(left, right) -> None:
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=0.0, atol=0.0)


def test_continuous_sac_update_and_actions() -> None:
    learner = ContinuousSAC(
        SACConfig(5, 2, 3, hidden_dims=(32, 32), actor_layer_norm=False, critic_layer_norm=False, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True)
    )
    assert learner.target_entropy == 1.0
    state = learner.initialize(jax.random.PRNGKey(0))
    updated, metrics = learner.update(state, _batch(discrete=False), jax.random.PRNGKey(1))
    assert _tree_delta(state.actor_params, updated.actor_params) > 0.0
    assert _tree_delta(state.critic_params, updated.critic_params) > 0.0
    assert all(np.isfinite(float(value)) for value in metrics)
    actions = learner.sample_actions(
        updated,
        jnp.zeros((4, 5)),
        jnp.zeros((4, 5)),
        jnp.zeros((4, 3)),
        jax.random.split(jax.random.PRNGKey(2), 4),
    )
    assert actions.shape == (4, 2)
    assert np.all(np.abs(np.asarray(actions)) <= 1.0)


def test_continuous_sac_rejects_unreachable_entropy_target() -> None:
    with pytest.raises(ValueError, match="exceeds the maximum"):
        ContinuousSAC(SACConfig(5, 2, 3, target_entropy=2.0, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True))


def test_empowerment_sac_backup_is_normalized_and_has_no_entropy_term() -> None:
    learner = ContinuousSAC(
        SACConfig(
            5,
            2,
            3,
            hidden_dims=(16,),
            actor_layer_norm=False,
            critic_layer_norm=False,
            discount=0.99,
            initial_alpha=100.0,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        )
    )
    state = learner.initialize(jax.random.PRNGKey(10))
    batch = _batch(discrete=False).replace(
        rewards=jnp.ones((16,), dtype=jnp.float32),
        masks=jnp.zeros((16,), dtype=jnp.float32),
    )
    _, metrics = learner.update(state, batch, jax.random.PRNGKey(11))
    assert float(metrics.target_q_mean) == pytest.approx(0.01, abs=1e-6)


def test_actor_uses_pre_update_critic_snapshot() -> None:
    common = dict(
        observation_dim=5,
        action_dim=2,
        skill_dim=3,
        hidden_dims=(16,),
        actor_layer_norm=False,
        critic_layer_norm=False,
    )
    slow_critic = ContinuousSAC(SACConfig(**common, critic_lr=1e-8, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True))
    fast_critic = ContinuousSAC(SACConfig(**common, critic_lr=1e-1, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True))
    slow_state = slow_critic.initialize(jax.random.PRNGKey(12))
    fast_state = fast_critic.initialize(jax.random.PRNGKey(12))
    batch = _batch(discrete=False)
    key = jax.random.PRNGKey(13)

    slow_updated, _ = slow_critic.update(slow_state, batch, key)
    fast_updated, _ = fast_critic.update(fast_state, batch, key)

    assert _tree_delta(slow_updated.critic_params, fast_updated.critic_params) > 0.0
    _assert_tree_allclose(slow_updated.actor_params, fast_updated.actor_params)


def test_skill_zero_heads_and_delayed_value_replacement_are_live() -> None:
    learner = ContinuousSAC(
        SACConfig(
            5,
            2,
            3,
            hidden_dims=(16,),
            actor_layer_norm=False,
            critic_layer_norm=False,
            replace_skill_zero_reward_with_random_value=True,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            potential_steps=4,
        ),
    )
    state = learner.initialize(jax.random.PRNGKey(14))
    base = _batch(discrete=False, batch_size=8)
    skills = base.skills.at[:4].set(0.0)
    batch = base.replace(
        skills=skills,
        next_skills=skills,
        timesteps=jnp.arange(8, dtype=jnp.int32) % 4,
    )
    updated, metrics = learner.update(state, batch, jax.random.PRNGKey(15))
    assert _tree_delta(state.skill0_actor_params, updated.skill0_actor_params) > 0.0
    assert _tree_delta(state.skill0_critic_params, updated.skill0_critic_params) > 0.0
    assert float(metrics.potential_weight / metrics.replay_weight) == pytest.approx(0.5)
    assert np.isfinite(float(metrics.potential_reward_total))


def test_potential_diagnostics_use_the_separate_critic_and_masked_rows() -> None:
    from empowerment_warp.training.sac.logging import summarize_potential_metrics

    learner = ContinuousSAC(SACConfig(
        5, 2, 3, hidden_dims=(4,), actor_layer_norm=False, critic_layer_norm=False,
        discount=0.5, replace_skill_zero_reward_with_random_value=True,
        separate_skill0_actor=True, separate_skill0_critic=True, potential_steps=2,
        skill0_value_num_skills=3,
    ))
    state = learner.initialize(jax.random.PRNGKey(80))

    def constant(params, q1, q2):
        params = jax.tree_util.tree_map(jnp.zeros_like, params)
        for name, value in (('q1', q1), ('q2', q2)):
            params[name]['Dense_1']['bias'] = jnp.full_like(params[name]['Dense_1']['bias'], value)
        return params

    state = state.replace(
        critic_params=constant(state.critic_params, 100, 200),
        target_critic_params=constant(state.target_critic_params, 8, 10),
        skill0_critic_params=constant(state.skill0_critic_params, 1, 3),
        target_skill0_critic_params=constant(state.target_skill0_critic_params, 4, 6),
    )
    batch = _batch(discrete=False, batch_size=4)
    skills = batch.skills.at[:3].set(0)
    batch = batch.replace(
        skills=skills, next_skills=skills, timesteps=jnp.array([0, 1, 1, 0]),
        loss_weights=jnp.array([1., 3., 0., 2.]), masks=jnp.array([1., 0., 1., 1.]),
    )
    _, metrics = learner.update(state, batch, jax.random.PRNGKey(81))
    report = summarize_potential_metrics(metrics)
    # Skill value 8; delayed rewards [2,4]; targets [.5*2+.5*4,.5*4]=[3,2].
    assert report['potential/skill_value_mean'] == pytest.approx(8)
    assert report['potential/reward_mean'] == pytest.approx(3.5)
    assert report['potential/replay_fraction'] == pytest.approx(4/6)
    assert report['potential/critic_q_mean'] == pytest.approx(1)
    assert report['potential/critic_target_mean'] == pytest.approx(2.25)
    # Mean twin squared errors: [(4+0)/2,(1+1)/2]=[2,1], weights [1,3].
    assert report['potential/critic_td_rmse'] == pytest.approx(np.sqrt(1.25))

    skill_only = batch.replace(skills=jnp.ones((4, 3)), next_skills=jnp.ones((4, 3)))
    _, inactive = learner.update(state, skill_only, jax.random.PRNGKey(82))
    assert summarize_potential_metrics(inactive) == {}


def test_random_switch_uses_each_episode_boundary_for_potential_reward_discount() -> None:
    from empowerment_warp.core.rollout import potential_switch_steps
    from empowerment_warp.training.sac.logging import summarize_potential_metrics

    learner = ContinuousSAC(SACConfig(
        5, 2, 3, hidden_dims=(4,), actor_layer_norm=False, critic_layer_norm=False,
        discount=.5, replace_skill_zero_reward_with_random_value=True,
        separate_skill0_actor=True, separate_skill0_critic=True, potential_steps=2,
        random_potential_switch_max_steps=4, potential_switch_seed=19,
    ))
    state = learner.initialize(jax.random.PRNGKey(90))
    targets = jax.tree_util.tree_map(jnp.zeros_like, state.target_critic_params)
    for name in ('q1', 'q2'):
        targets[name]['Dense_1']['bias'] = jnp.full_like(targets[name]['Dense_1']['bias'], 8.)
    state = state.replace(target_critic_params=targets)
    ids = jnp.arange(32, dtype=jnp.int32)
    boundaries = potential_switch_steps(ids, 2, 4, 19)
    skills = jnp.where((boundaries > 0)[:, None], jnp.zeros((32, 3)), jnp.ones((32, 3)))
    batch = _batch(discrete=False, batch_size=32).replace(
        episode_ids=ids, timesteps=jnp.zeros(32, dtype=jnp.int32),
        skills=skills, next_skills=skills,
    )
    _, metrics = learner.update(state, batch, jax.random.PRNGKey(91))
    report = summarize_potential_metrics(metrics)
    actual_boundaries = np.asarray(boundaries)[np.asarray(boundaries) > 0]
    assert len(np.unique(actual_boundaries)) > 1
    expected = np.mean(8 * .5 ** actual_boundaries)
    assert report['potential/reward_mean'] == pytest.approx(expected)
    assert abs(expected - 8 * .5 ** 2) > .01  # catches a fixed-100-style delay


@pytest.mark.parametrize('value, expected', [(-8., 0.), (0., 0.), (8., 8.)])
def test_potential_reward_floors_state_value_before_discount(value, expected):
    learner = ContinuousSAC(SACConfig(
        5, 2, 3, hidden_dims=(4,), actor_layer_norm=False, critic_layer_norm=False,
        discount=.5, replace_skill_zero_reward_with_random_value=True,
        separate_skill0_actor=True, separate_skill0_critic=True, potential_steps=2,
        skill0_value_num_skills=4,
    ))
    state = learner.initialize(jax.random.PRNGKey(90))
    targets = jax.tree_util.tree_map(jnp.zeros_like, state.target_critic_params)
    for name in ('q1', 'q2'):
        targets[name]['Dense_1']['bias'] = jnp.full_like(targets[name]['Dense_1']['bias'], value)
    state = state.replace(target_critic_params=targets)
    batch = _batch(discrete=False, batch_size=8).replace(
        skills=jnp.zeros((8, 3)), next_skills=jnp.zeros((8, 3)),
        timesteps=jnp.zeros(8, dtype=jnp.int32),
    )
    _, metrics = learner.update(state, batch, jax.random.PRNGKey(91))
    assert float(metrics.potential_skill_value_total / metrics.potential_weight) == pytest.approx(expected)
    assert float(metrics.potential_reward_total / metrics.potential_weight) == pytest.approx(.25 * expected)
