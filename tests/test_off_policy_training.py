from __future__ import annotations

import math

from typing import NamedTuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillConfig, SkillLearner
from empowerment_warp.agents.see_sac import ContinuousSAC, SACConfig
from empowerment_warp.core.replay import (
    GEOMETRIC_ANCHOR_REPLAY_MODE,
    MARGINAL_BRANCH_SKILL_START_MODE,
)
from empowerment_warp.core.rollout import potential_switch_steps
from empowerment_warp.training.sac.config import OffPolicyConfig
from empowerment_warp.training.sac.runner import OffPolicyTrainer
from empowerment_warp.utils.sampling import (
    marginal_representation_archive_batch,
    representation_archive_batch,
    policy_hindsight_batch,
)


@struct.dataclass
class _State:
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array


class _Config(NamedTuple):
    num_envs: int


class _Env:
    config = _Config(num_envs=4)

    def reset(self, keys):
        values = jax.vmap(lambda key: jax.random.uniform(key, (3,), minval=-0.1, maxval=0.1))(keys)
        return _State(
            obs=values,
            reward=jnp.zeros((4,), dtype=jnp.float32),
            terminated=jnp.zeros((4,), dtype=jnp.bool_),
            truncated=jnp.zeros((4,), dtype=jnp.bool_),
            step_count=jnp.zeros((4,), dtype=jnp.int32),
        )

    def step(self, state, actions):
        step_count = state.step_count + 1
        delta = jnp.pad(actions, ((0, 0), (0, 1)))
        return _State(
            obs=state.obs + 0.05 * delta,
            reward=jnp.zeros((4,), dtype=jnp.float32),
            terminated=jnp.zeros((4,), dtype=jnp.bool_),
            truncated=step_count >= 4,
            step_count=step_count,
        )

    @staticmethod
    def where(mask, state, replacement):
        return jax.tree_util.tree_map(
            lambda current, new: jnp.where(
                mask.reshape(mask.shape + (1,) * (current.ndim - mask.ndim)),
                new,
                current,
            ),
            state,
            replacement,
        )

    @classmethod
    def where_done(cls, state, reset_state):
        return cls.where(state.terminated | state.truncated, state, reset_state)


def _tree_delta(left, right) -> float:
    deltas = jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), left, right)
    return float(sum(jax.tree_util.tree_leaves(deltas)))


def _assert_tree_allclose(left, right) -> None:
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=0.0, atol=0.0)


def _top_level_scan_lengths(closed_jaxpr) -> tuple[int, ...]:
    return tuple(
        equation.params["length"]
        for equation in closed_jaxpr.jaxpr.eqns
        if equation.primitive.name == "scan"
    )


def test_full_compiled_segment_updates_skill_and_sac() -> None:
    env = _Env()
    sac = ContinuousSAC(
        SACConfig(
            3,
            2,
            2,
            hidden_dims=(16,),
            actor_layer_norm=False,
            critic_layer_norm=False,
            target_entropy=1.0,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        )
    )
    skill = SkillLearner(
        SkillConfig(
            method="potential",
            observation_dim=2,
            observation_indices=(0, 2),
            skill_dim=2,
            discrete_skills=False,
            hidden_dims=(16,),
            layer_norm=False,
            phi_norm_cap=1.0,
        )
    )
    trainer = OffPolicyTrainer(
        env,
        sac,
        skill,
        OffPolicyConfig(
            rollout_steps=2,
            updates_per_chunk=2,
            batch_size=4,
            replay_capacity=32,
            episode_length=4,
            representation_episodes=4,
            max_future_steps=3,
        ),
    )
    state = trainer.initialize(jax.random.PRNGKey(0))
    state, metrics = trainer.run(state)
    # no episode has ended yet: the bag is empty and no update runs
    assert int(state.representation_replay.size) == 0
    assert int(state.sac.step) == 0
    assert np.all(np.asarray(metrics.update_applied) == 0.0)
    state, metrics = trainer.run(state)
    assert int(state.env_steps) == 16
    assert int(state.replay.size) == 16
    assert state.replay.data.observations.shape == (32, 3)
    # bag: (representation_episodes, episode_length, view); staging: (num_envs, episode_length, view)
    assert state.representation_replay.episodes.observations.shape == (4, 4, 2)
    assert state.representation_replay.staging.observations.shape == (4, 4, 2)
    assert not hasattr(state.representation_replay.episodes, "actions")
    assert int(state.representation_replay.size) == 4
    assert int(state.sac.step) == 2
    assert int(state.skill.step) == 2
    assert np.all(np.asarray(metrics.update_applied) == 1.0)
    assert np.isfinite(np.asarray(metrics.sac.critic_loss)).all()
    assert np.isfinite(np.asarray(metrics.skill.loss)).all()


def test_optimizer_steps_interleave_sac_and_encoder_updates() -> None:
    def run_with_encoder_lr(learning_rate: float):
        env = _Env()
        sac = ContinuousSAC(
            SACConfig(
                3,
                2,
                2,
                hidden_dims=(16,),
                actor_layer_norm=False,
                critic_layer_norm=False,
                target_entropy=1.0,
                separate_skill0_actor=True,
                separate_skill0_critic=True,
                replace_skill_zero_reward_with_random_value=True,
            )
        )
        skill = SkillLearner(
            SkillConfig(
                method="potential",
                observation_dim=3,
                skill_dim=2,
                discrete_skills=False,
                hidden_dims=(16,),
                learning_rate=learning_rate,
                layer_norm=False,
                phi_norm_cap=1.0,
            )
        )
        trainer = OffPolicyTrainer(
            env,
            sac,
            skill,
            OffPolicyConfig(
                rollout_steps=2,
                updates_per_chunk=2,
                batch_size=4,
                replay_capacity=32,
                episode_length=4,
                representation_episodes=4,
                max_future_steps=3,
            ),
        )
        state = trainer.initialize(jax.random.PRNGKey(30))
        state, _ = trainer.run(state)  # fills the bag; no update yet
        updated, metrics = trainer.run(state)
        assert np.all(np.asarray(metrics.update_applied) == 1.0)
        return updated

    slow_encoder = run_with_encoder_lr(1e-8)
    fast_encoder = run_with_encoder_lr(1e-1)

    assert _tree_delta(slow_encoder.skill.params, fast_encoder.skill.params) > 0.0
    _assert_tree_allclose(slow_encoder.sac.actor_params, fast_encoder.sac.actor_params)
    # The first optimizer step advances both learners from one shared snapshot.
    # Its updated target encoder therefore affects the second critic target.
    assert _tree_delta(slow_encoder.sac.critic_params, fast_encoder.sac.critic_params) > 0.0








def test_potential_phase_bag_holds_completed_episodes_with_in_episode_futures() -> None:
    env = _Env()
    sac = ContinuousSAC(
        SACConfig(
            3,
            2,
            2,
            hidden_dims=(8,),
            actor_layer_norm=False,
            critic_layer_norm=False,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        )
    )
    skill = SkillLearner(
        SkillConfig(
            method="potential",
            observation_dim=3,
            skill_dim=2,
            discrete_skills=False,
            hidden_dims=(8,),
            layer_norm=False,
        )
    )
    trainer = OffPolicyTrainer(
        env,
        sac,
        skill,
        OffPolicyConfig(
            rollout_steps=2,
            updates_per_chunk=1,
            batch_size=4,
            replay_capacity=16,
            episode_length=4,
            representation_episodes=4,
            max_future_steps=4,
            potential_steps=1,
        ),
    )
    state = trainer.initialize(jax.random.PRNGKey(40))
    traced = jax.make_jaxpr(trainer._run_impl)(state)
    gated_update_scans = [
        tuple(_top_level_scan_lengths(branch) for branch in equation.params["branches"])
        for equation in traced.jaxpr.eqns
        if equation.primitive.name == "cond"
        and sorted(len(_top_level_scan_lengths(branch)) for branch in equation.params["branches"])
        == [0, 1]
    ]
    assert gated_update_scans == [((), (1,))]

    # chunk 1 = steps 0 and 1: nothing has ended, the bag is empty.
    state, first_metrics = trainer.run(state)
    assert int(state.replay.size) == 8
    assert int(state.representation_replay.size) == 0
    assert float(first_metrics.update_applied[0]) == 0.0
    # chunk 2 = steps 2 and 3: every lane ends its 4-step episode with 3 skill-on rows.
    state, second_metrics = trainer.run(state)
    assert int(state.representation_replay.size) == 4
    assert int(state.representation_replay.active_rows) == 12
    assert np.all(np.asarray(state.representation_replay.lengths) == 4)
    assert float(second_metrics.update_applied[0]) == 1.0
    assert float(second_metrics.archive.valid_fraction[0]) == 1.0

    archive = state.representation_replay
    slots, rows, row_valid = trainer.representation_replay.sample_active(
        archive, jax.random.PRNGKey(41), 4
    )
    assert np.all(np.asarray(row_valid))
    skills = np.asarray(archive.episodes.skills)[np.asarray(slots), np.asarray(rows)]
    assert np.all(np.linalg.norm(skills, axis=-1) > 1e-6)
    assert np.all(np.asarray(rows) >= 1)  # skill-off row 0 is never sampled
    batch, _ = representation_archive_batch(
        trainer.agent.config,
        trainer.agent.representation_replay,
        archive, slots, rows, row_valid, jax.random.PRNGKey(42)
    )
    observations = np.asarray(archive.episodes.observations)
    for row, (slot, position) in enumerate(zip(np.asarray(slots), np.asarray(rows))):
        future = np.asarray(batch.future_observations)[row]
        later = [observations[slot, j] for j in range(position, 4)]
        assert any(np.allclose(future, candidate) for candidate in later)
        assert bool(np.asarray(batch.future_valid)[row])


def _cadence_trainer(encoder_updates_per_chunk: int) -> OffPolicyTrainer:
    env = _Env()
    sac = ContinuousSAC(
        SACConfig(
            3,
            2,
            2,
            hidden_dims=(16,),
            actor_layer_norm=False,
            critic_layer_norm=False,
            target_entropy=1.0,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        )
    )
    skill = SkillLearner(
        SkillConfig(
            method="potential",
            observation_dim=2,
            observation_indices=(0, 2),
            skill_dim=2,
            discrete_skills=False,
            hidden_dims=(16,),
            layer_norm=False,
            phi_norm_cap=1.0,
        )
    )
    return OffPolicyTrainer(
        env,
        sac,
        skill,
        OffPolicyConfig(
            rollout_steps=2,
            updates_per_chunk=4,
            batch_size=4,
            replay_capacity=32,
            episode_length=4,
            representation_episodes=4,
            max_future_steps=3,
            encoder_updates_per_chunk=encoder_updates_per_chunk,
        ),
    )


def test_partition_schedule_boundaries():
    config = OffPolicyConfig(reward_partition_anneal_start_steps=25_000_000,
                             reward_partition_anneal_end_steps=100_000_000)
    steps = jnp.array([0, 25_000_000, 62_500_000, 100_000_000, 500_000_000])
    np.testing.assert_array_equal(jax.jit(config.reward_partition_weight)(steps), [0, 0, .5, 1, 1])
    with pytest.raises(ValueError, match='partition annealing'):
        OffPolicyConfig(reward_partition_anneal_start_steps=100, reward_partition_anneal_end_steps=50)


@pytest.mark.parametrize('dimension', [2, 4])
def test_compiled_partition_anneal_matches_controls_and_resumed_clock(dimension):
    from dataclasses import replace
    from flax import serialization

    sac = ContinuousSAC(SACConfig(3, 2, dimension, hidden_dims=(16,),
                                 actor_layer_norm=False, critic_layer_norm=False, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True))
    skill_config = SkillConfig(method='potential', observation_dim=3, skill_dim=dimension,
                               discrete_skills=False, hidden_dims=(16,), layer_norm=False,
                               phi_norm_cap=0, continuous_reward_temperature=True,
                               encoder_objective='vmf_forward', reward_encoder='live')
    config = OffPolicyConfig(rollout_steps=2, updates_per_chunk=2, batch_size=4,
                              replay_capacity=32, episode_length=4,
                              representation_episodes=4, max_future_steps=3)
    raw = OffPolicyTrainer(_Env(), sac, SkillLearner(skill_config), config)
    normalized_skill = SkillLearner(replace(skill_config, continuous_reward_normalization='sphere'))
    normalized = OffPolicyTrainer(_Env(), sac, normalized_skill, config)
    annealed = OffPolicyTrainer(_Env(), sac, normalized_skill,
                                replace(config, reward_partition_anneal_start_steps=25_000_000,
                                        reward_partition_anneal_end_steps=100_000_000))
    state = raw.initialize(jax.random.PRNGKey(0))
    state, _ = raw.run(state)
    state, _ = raw.run(state)
    assert int(state.sac.step) > 0
    for step, control in [(25_000_000, raw), (100_000_000, normalized)]:
        # Each real collector chunk adds eight transitions before its replay updates.
        checkpoint_state = state.replace(env_steps=jnp.asarray(step - 8, jnp.int32))
        restored = serialization.from_bytes(checkpoint_state, serialization.to_bytes(checkpoint_state))
        actual, metrics = annealed.run(restored)
        expected, _ = control.run(restored)
        assert int(actual.env_steps) == step
        assert np.all(np.asarray(metrics.update_applied) == 1)
        # A dynamic coefficient can alter XLA fusion versus a constant control;
        # compare float32 update results within rounding, not bit identity.
        for a, b in zip(jax.tree_util.tree_leaves((actual.sac, actual.skill)),
                        jax.tree_util.tree_leaves((expected.sac, expected.skill)), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-8)
    middle, metrics = annealed.run(state.replace(env_steps=jnp.asarray(62_499_992, jnp.int32)))
    assert int(middle.env_steps) == 62_500_000
    assert np.isfinite(np.asarray(metrics.sac.policy_reward_skill_on)).all()


def test_encoder_cadence_decouples_encoder_from_sac_replay_ratio() -> None:
    """The encoder's gradient-step count is what governs memorisation.

    SAC keeps its replay ratio either way; only the encoder's step count moves.
    """

    for encoder_updates, expected_encoder_steps in ((0, 4), (4, 4), (2, 2), (1, 1)):
        trainer = _cadence_trainer(encoder_updates)
        assert trainer.config.resolved_encoder_updates_per_chunk == expected_encoder_steps
        assert (
            trainer.config.encoder_update_stride
            == 4 // expected_encoder_steps
        )
        state = trainer.initialize(jax.random.PRNGKey(3))
        for _ in range(6):
            state, _ = trainer.run(state)
        sac_steps = int(state.sac.step)
        assert sac_steps > 0, "the test must reach the updating regime"
        # SAC is untouched by the knob; the encoder scales with it.
        assert sac_steps == 4 * (sac_steps // 4)
        assert int(state.skill.step) == sac_steps // 4 * expected_encoder_steps


def test_encoder_cadence_must_divide_updates_per_chunk() -> None:
    with pytest.raises(ValueError, match="must divide updates_per_chunk"):
        OffPolicyConfig(updates_per_chunk=4, encoder_updates_per_chunk=3)
    with pytest.raises(ValueError, match="cannot exceed updates_per_chunk"):
        OffPolicyConfig(updates_per_chunk=4, encoder_updates_per_chunk=8)


def _branch_replay_trainer(to_policy_replay: bool) -> OffPolicyTrainer:
    env = _Env()
    skill = SkillLearner(
        SkillConfig(
            method="potential",
            observation_dim=3,
            skill_dim=2,
            discrete_skills=False,
            num_skills=0,
            hidden_dims=(8,),
            layer_norm=False,
        )
    )
    sac = ContinuousSAC(
        SACConfig(
            3,
            2,
            2,
            hidden_dims=(8,),
            actor_layer_norm=False,
            critic_layer_norm=False,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        ),
    )
    return OffPolicyTrainer(
        env,
        sac,
        skill,
        OffPolicyConfig(
            rollout_steps=2,
            updates_per_chunk=1,
            batch_size=4,
            replay_capacity=256,
            episode_length=4,
            representation_episodes=6,
            max_future_steps=4,
            potential_steps=2,
            stagger_reset_groups=2,
            skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
            branch_transitions_to_policy_replay=to_policy_replay,
        ),
    )


def test_branch_transitions_reach_policy_replay_with_fixed_anchors() -> None:
    """Branch rollouts are real transitions under a commanded skill.

    Live lanes hold one skill for a whole episode, so only branch rows carry a
    same-anchor/different-skill contrast, and only they have the fixed anchor
    the potential reward assumes. This checks completed branches reach policy
    replay, cannot be paired with live rows by hindsight, and keep the
    vector-step-major layout the future lookup depends on.
    """

    off = _branch_replay_trainer(False)
    on = _branch_replay_trainer(True)
    horizon = on.config.branch_horizon

    state_off = off.initialize(jax.random.PRNGKey(91))
    state_on = on.initialize(jax.random.PRNGKey(91))
    emitted = 0
    for _ in range(12):
        size_before = int(state_on.replay.size)
        state_off, _ = off.run(state_off)
        state_on, metrics_on = on.run(state_on)
        grown = int(state_on.replay.size) - size_before
        # `applied` also counts the one-shot bootstrap, which stays
        # encoder-only; the staggered collector is what feeds policy replay.
        if float(metrics_on.branch.policy_rows_alive_fraction) > 0.0:
            emitted += 1
            # Live rows plus one completion batch of branch rows.
            assert grown > on.config.rollout_steps * on.num_envs
        else:
            assert grown == on.config.rollout_steps * on.num_envs
    assert emitted > 0, "no branch completed; the test never exercised the path"

    # The flag is the only difference, so the surplus is exactly branch rows.
    surplus = int(state_on.replay.size) - int(state_off.replay.size)
    assert surplus > 0
    assert surplus % (horizon * on.branch_completion_batch_size) == 0

    data = state_on.replay.data
    ids = np.asarray(data.episode_ids)[: int(state_on.replay.size)]
    # Negative ids cannot collide with the collector's non-negative counters, so
    # a hindsight future can never cross from a branch into live experience.
    assert np.any(ids < 0), "no branch rows found in replay"
    assert np.any(ids >= 0), "live rows should still be present"

    branch_rows = np.nonzero(ids < 0)[0]
    weights = np.asarray(data.loss_weights)[branch_rows]
    # Padding rows are inserted but inert, so the replay keeps its layout.
    assert np.all((weights == 0.0) | (weights == 1.0))
    assert weights.sum() > 0


def test_branch_policy_replay_requires_marginal_branch() -> None:
    with pytest.raises(ValueError, match="requires marginal_branch"):
        OffPolicyConfig(branch_transitions_to_policy_replay=True)


def test_fano_bound_is_zero_at_chance_and_positive_above_it() -> None:
    """The CE bound goes vacuous under overconfidence; Fano cannot.

    `log K - CE` can be tens of nats negative at chance accuracy, which reads as
    "strongly negative information" when the truth is "no information, plus a
    miscalibrated critic".
    """

    from empowerment_warp.training.sac.logging import _branch_fano_mi_lower_bound

    K = 30
    assert float(_branch_fano_mi_lower_bound(jnp.asarray(1.0 / K), K)) == pytest.approx(
        0.0, abs=1e-4
    )
    # Never negative, however bad the classifier.
    assert float(_branch_fano_mi_lower_bound(jnp.asarray(0.0), K)) >= 0.0
    # Monotone in accuracy, and matches the closed form off chance.
    below, above = (
        float(_branch_fano_mi_lower_bound(jnp.asarray(0.05), K)),
        float(_branch_fano_mi_lower_bound(jnp.asarray(0.13), K)),
    )
    assert 0.0 <= below < above
    assert above == pytest.approx(0.0855, abs=2e-3)
    # A perfect classifier recovers the full log K.
    assert float(_branch_fano_mi_lower_bound(jnp.asarray(1.0), K)) == pytest.approx(
        math.log(K), abs=1e-3
    )


def test_heldout_reward_and_accuracy_score_the_same_displacement() -> None:
    """Both held-out numbers must use the anchor->future displacement.

    The marginal archive sets `next_observations` to outcome 0, so a batch
    passed through unchanged makes the reward a ONE-STEP delta while the
    accuracy is the full displacement. The two then measure different geometries
    and cannot be compared -- which produced a spurious "the reward signal is
    1.8% of achievable" reading.
    """

    trainer = _branch_replay_trainer(True)
    state = trainer.initialize(jax.random.PRNGKey(31))
    archive = state.representation_replay
    slots = jnp.zeros((4,), dtype=jnp.int32)
    rows = jnp.zeros((4,), dtype=jnp.int32)
    valid = jnp.ones((4,), dtype=jnp.bool_)
    batch, _ = marginal_representation_archive_batch(
        trainer.agent.config,
        trainer.agent.representation_replay,
        archive, slots, rows, valid, jax.random.PRNGKey(32)
    )
    # The archive itself must align them, so every consumer of this batch scores
    # the reward on the same displacement the loss uses.
    np.testing.assert_array_equal(
        np.asarray(batch.next_observations), np.asarray(batch.future_observations)
    )


def _live_potential_trainer(live_potential_only: bool) -> OffPolicyTrainer:
    env = _Env()
    skill = SkillLearner(
        SkillConfig(
            method="potential",
            observation_dim=3,
            skill_dim=2,
            discrete_skills=False,
            num_skills=0,
            hidden_dims=(8,),
            layer_norm=False,
        )
    )
    sac = ContinuousSAC(
        SACConfig(3, 2, 2, hidden_dims=(8,), actor_layer_norm=False, critic_layer_norm=False, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True),
    )
    return OffPolicyTrainer(
        env, sac, skill,
        OffPolicyConfig(
            rollout_steps=2, updates_per_chunk=1, batch_size=4, replay_capacity=256,
            episode_length=4, representation_episodes=6, max_future_steps=4,
            potential_steps=2, stagger_reset_groups=2,
            skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
            branch_transitions_to_policy_replay=True,
            sample_hindsight_anchors=True,
            live_potential_only=live_potential_only,
        ),
    )


def test_live_potential_only_makes_skill_rows_exactly_the_branch_rows() -> None:
    """The whole point: skill-on rows must all be branch rows.

    `skill0_mask` routes zero-skill rows to the skill-0 actor and skill-on rows
    to the empowerment reward. If live lanes never activate a skill, the only
    rows carrying that reward are branch rows -- whose anchor is the fixed
    branch anchor the encoder was trained on, so the reward is never scored on a
    geometry the encoder has not seen.
    """

    off = _live_potential_trainer(False)
    on = _live_potential_trainer(True)
    assert off.config.collector_potential_steps == 0
    assert on.config.collector_potential_steps > on.config.episode_length
    # Decoupled from the branch horizon, which still comes from potential_steps.
    assert on.config.branch_horizon == off.config.branch_horizon

    state_off = off.initialize(jax.random.PRNGKey(84))
    state_on = on.initialize(jax.random.PRNGKey(84))
    assert np.any(np.linalg.norm(np.asarray(state_off.collector.skills), axis=-1) > 0.0)
    np.testing.assert_allclose(np.asarray(state_on.collector.skills), 0.0)

    for _ in range(6):
        state_off, _ = off.run(state_off)
        state_on, _ = on.run(state_on)

    ids = np.asarray(state_on.replay.data.episode_ids)[: int(state_on.replay.size)]
    norms = np.linalg.norm(
        np.asarray(state_on.replay.data.skills)[: int(state_on.replay.size)], axis=-1
    )
    live, branch = ids >= 0, ids < 0
    assert live.any() and branch.any(), "need both row kinds to make the claim"
    np.testing.assert_allclose(norms[live], 0.0, atol=1e-6)
    assert (norms[branch] > 1e-6).any(), "branch rows must carry active skills"

    # The control still mixes skill-on rows into the live stream.
    off_ids = np.asarray(state_off.replay.data.episode_ids)[: int(state_off.replay.size)]
    off_norms = np.linalg.norm(
        np.asarray(state_off.replay.data.skills)[: int(state_off.replay.size)], axis=-1
    )
    assert (off_norms[off_ids >= 0] > 1e-6).any()


def test_branch_reward_batch_preserves_the_fixed_archive_anchor() -> None:
    trainer = _live_potential_trainer(True)
    state = trainer.initialize(jax.random.PRNGKey(85))
    for _ in range(6):
        state, _ = trainer.run(state)

    size = int(state.replay.size)
    episode_ids = np.asarray(state.replay.data.episode_ids)[:size]
    timesteps = np.asarray(state.replay.data.timesteps)[:size]
    weights = np.asarray(state.replay.data.loss_weights)[:size]
    branch_indices = np.flatnonzero(
        (episode_ids < 0) & (timesteps > 0) & (weights > 0.0)
    )
    assert branch_indices.size > 0
    indices = jnp.asarray(branch_indices, dtype=jnp.int32)
    stored = trainer.replay.get(state.replay, indices)

    replay_batch, reward_batch, _, _, _ = policy_hindsight_batch(
        trainer.agent.config,
        trainer.replay,
        state.replay,
        indices,
        jax.random.PRNGKey(86),
        hindsight=True,
    )

    np.testing.assert_allclose(
        np.asarray(reward_batch.anchor_observations),
        np.asarray(stored.anchor_observations),
    )
    np.testing.assert_allclose(
        np.asarray(replay_batch.anchor_observations),
        np.asarray(stored.anchor_observations),
    )
    assert not np.allclose(
        np.asarray(stored.observations),
        np.asarray(stored.anchor_observations),
    )


def test_live_potential_only_requires_marginal_branch() -> None:
    with pytest.raises(ValueError, match="live_potential_only requires marginal_branch"):
        OffPolicyConfig(live_potential_only=True)




def _segment_end_trainer(outcome: str, samples: int = 8) -> OffPolicyTrainer:
    env = _Env()
    sac = ContinuousSAC(
        SACConfig(3, 2, 2, hidden_dims=(8,), actor_layer_norm=False, critic_layer_norm=False, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True)
    )
    skill = SkillLearner(
        SkillConfig(
            method="potential", observation_dim=3, skill_dim=2, discrete_skills=False,
            hidden_dims=(8,), layer_norm=False,
        )
    )
    return OffPolicyTrainer(
        env, sac, skill,
        OffPolicyConfig(
            rollout_steps=4, updates_per_chunk=1, batch_size=4, replay_capacity=32,
            episode_length=4, representation_episodes=4, max_future_steps=4,
            sample_hindsight_anchors=True, achieved_skill_relabel_outcome=outcome,
            achieved_skill_relabel_samples=samples,
        ),
    )


def _random_switch_trainer(replay_mode=GEOMETRIC_ANCHOR_REPLAY_MODE, replay_capacity=32) -> OffPolicyTrainer:
    return OffPolicyTrainer(
        _Env(),
        ContinuousSAC(SACConfig(
            3, 2, 4, hidden_dims=(8,), actor_layer_norm=False, critic_layer_norm=False,
            potential_steps=1, random_potential_switch_max_steps=3, potential_switch_seed=31,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        )),
        SkillLearner(SkillConfig(
            method="potential", observation_dim=3, skill_dim=4, discrete_skills=False,
            hidden_dims=(8,), layer_norm=False, encoder_objective="vmf_forward", phi_norm_cap=0.0,
            continuous_reward_normalization="none", achieved_skill_relabeling=True,
        )),
        OffPolicyConfig(
            rollout_steps=4, updates_per_chunk=1, batch_size=4, replay_capacity=replay_capacity,
            episode_length=4, representation_episodes=4, max_future_steps=4,
            sample_hindsight_anchors=True, potential_steps=1,
            random_potential_switch_max_steps=3, potential_switch_seed=31,
            potential_replay_mode=replay_mode, future_gamma=0.9999,
        ),
    )


def test_random_potential_switch_compiled_update_and_archive() -> None:
    trainer = _random_switch_trainer()
    initial = trainer.initialize(jax.random.PRNGKey(30))
    state, metrics = trainer.run(initial)
    assert int(state.sac.step) == 1 and int(state.skill.step) == 1
    assert float(metrics.update_applied[0]) == 1.0
    assert np.isfinite(np.asarray(metrics.sac.critic_loss)).all()
    assert np.isfinite(np.asarray(metrics.skill.loss)).all()
    assert _tree_delta(initial.sac.actor_params, state.sac.actor_params) > 0.0
    archive = state.representation_replay
    boundaries = potential_switch_steps(archive.episodes.episode_ids[:, 0], 1, 3, 31)
    assert int(archive.active_rows) == int(jnp.sum(4 - boundaries))
    slots, rows, valid = trainer.representation_replay.sample_active(
        archive, jax.random.PRNGKey(32), 16, candidates_per_row=8,
    )
    sampled = trainer.representation_replay.get(archive, slots, rows)
    assert np.all(np.asarray(sampled.timesteps)[valid] >= np.asarray(
        potential_switch_steps(sampled.episode_ids, 1, 3, 31)
    )[valid])
    # Second compiled segment crosses the next reset and replaces archive contents.
    state, metrics = trainer.run(state)
    assert int(state.sac.step) == 2 and int(state.skill.step) == 2
    assert float(metrics.update_applied[0]) == 1.0
    archive = state.representation_replay
    boundaries = potential_switch_steps(archive.episodes.episode_ids[:, 0], 1, 3, 31)
    assert int(archive.active_rows) == int(jnp.sum(4 - boundaries))


@pytest.mark.parametrize("replay_mode", [GEOMETRIC_ANCHOR_REPLAY_MODE])
def test_random_potential_switch_hindsight_respects_boundary_after_wrap(replay_mode) -> None:
    trainer = _random_switch_trainer(replay_mode, replay_capacity=24)
    state = trainer.initialize(jax.random.PRNGKey(30))
    collector, replay = state.collector, state.replay
    # Six vector steps fit: wrapping overwrites the first half of the older episode.
    # Observations encode time and episode ID so cross-phase/episode anchors are visible.
    for _ in range(3):
        collector, transitions, _ = trainer._rollout(state.sac, collector)
        t = transitions.timesteps.astype(jnp.float32)
        ids = transitions.episode_ids.astype(jnp.float32)
        boundary = potential_switch_steps(transitions.episode_ids, 1, 3, 31)
        observations = jnp.stack((t, ids, jnp.zeros_like(t)), axis=-1)
        anchors = observations.at[:, 0].set(jnp.minimum(t, boundary))
        transitions = transitions.replace(
            observations=observations, anchor_observations=anchors,
            next_observations=observations.at[:, 0].add(1),
            next_anchor_observations=anchors.at[:, 0].set(jnp.minimum(t + 1, boundary)),
        )
        replay = trainer.replay.insert(replay, transitions)
    assert int(replay.size) == 24
    indices = jnp.tile(jnp.arange(24, dtype=jnp.int32), 4)
    batch, _, _, _, _ = policy_hindsight_batch(
        trainer.agent.config,
        trainer.replay, replay, indices, jax.random.PRNGKey(33), hindsight=True,
    )
    active = np.linalg.norm(np.asarray(batch.skills), axis=-1) > 1e-6
    boundaries = np.asarray(potential_switch_steps(batch.episode_ids, 1, 3, 31))
    anchor_t = np.asarray(batch.anchor_observations[:, 0])
    t = np.asarray(batch.timesteps)
    assert active.any() and (~active).any()
    assert np.all(anchor_t[active] >= boundaries[active])
    assert np.all(anchor_t <= t)
    np.testing.assert_array_equal(batch.anchor_observations[:, 1], batch.episode_ids)
    if replay_mode == GEOMETRIC_ANCHOR_REPLAY_MODE:
        # The cap is activation or the oldest surviving same-episode row, whichever is later.
        stored_ids = np.asarray(replay.data.episode_ids)
        stored_t = np.asarray(replay.data.timesteps)
        first_available = np.asarray([
            stored_t[stored_ids == episode_id].min() for episode_id in np.asarray(batch.episode_ids)
        ])
        expected = np.maximum(boundaries, first_available)
        assert np.mean(anchor_t[active] == expected[active]) > 0.9
    else:
        assert np.any(anchor_t[active] < t[active]), "the test must exercise strict-prior HER"


def test_segment_end_relabel_outcome_is_the_episode_final_state() -> None:
    """`achieved_skill_relabel_outcome="segment_end"` relabels a skill-on replay row to the
    skill achieved from its (re-sampled) anchor to the END of the skill segment -- the same
    episode's final `next_observation` -- instead of to the row's own state. "current" (the base)
    must be byte-identical to the pre-knob behaviour, including the relabel outcome."""

    current = _segment_end_trainer("current")
    segment = _segment_end_trainer("segment_end")
    state = current.initialize(jax.random.PRNGKey(20))
    _, transitions, _ = current._rollout(state.sac, state.collector)
    replay_state = current.replay.insert(state.replay, transitions)
    # rows 4..7 = timestep 1 of four lanes (episode_length 4, so the episode end is timestep 3)
    indices = jnp.arange(4, 8, dtype=jnp.int32)
    _, _, allowed_c, outcomes_c, _ = policy_hindsight_batch(
        current.agent.config,
        current.replay, replay_state, indices, jax.random.PRNGKey(21), hindsight=True
    )
    _, _, allowed_s, outcomes_s, metrics_s = policy_hindsight_batch(
        segment.agent.config,
        segment.replay, replay_state, indices, jax.random.PRNGKey(21), hindsight=True
    )
    np.testing.assert_array_equal(np.asarray(allowed_c), np.asarray(allowed_s))
    # base: the outcome is the row's own observation
    np.testing.assert_allclose(np.asarray(outcomes_c), np.asarray(transitions.observations[4:8]))
    # segment_end: the outcome is the episode's final next_observation (rows 12..15 = timestep 3)
    np.testing.assert_allclose(np.asarray(outcomes_s), np.asarray(transitions.next_observations[12:16]))
    assert float(metrics_s.segment_end_available_fraction) == 1.0


def test_discounted_relabel_outcome_samples_same_episode_futures_and_aggregates() -> None:
    """`achieved_skill_relabel_outcome="discounted"` supplies M futures per row from the same
    episode, each strictly after the row and no later than the episode end, and the relabel
    aggregates them in representation space so the label is the discounted-return greedy skill.
    Every future must be a real row of the row's own episode; the availability metric is 1."""

    trainer = _segment_end_trainer("discounted", samples=4)
    state = trainer.initialize(jax.random.PRNGKey(20))
    _, transitions, _ = trainer._rollout(state.sac, state.collector)
    replay_state = trainer.replay.insert(state.replay, transitions)
    indices = jnp.arange(4, 8, dtype=jnp.int32)  # timestep 1 of four lanes; futures at t=2,3
    replay_batch, _, allowed, outcomes, metrics = policy_hindsight_batch(
        trainer.agent.config,
        trainer.replay, replay_state, indices, jax.random.PRNGKey(21), hindsight=True
    )
    outcomes = np.asarray(outcomes)
    assert outcomes.shape == (4, 4, 3)  # rows, samples, obs
    nexts = np.asarray(transitions.next_observations).reshape(4, 4, 3)  # (timestep, lane, obs)
    for lane in range(4):
        for sample in range(4):
            candidates = nexts[2:4, lane]  # next_observations of timesteps 2 and 3 of this lane
            assert any(np.allclose(outcomes[lane, sample], c) for c in candidates), (lane, sample)
    assert float(metrics.segment_end_available_fraction) == 1.0
    # the relabel accepts the sampled axis and produces one skill per row
    relabeled, _ = trainer.skill.relabel_achieved_skills(
        state.skill, transitions_slice(transitions, 4, 8),
        allowed, outcome_observations=jnp.asarray(outcomes),
    )
    assert np.asarray(relabeled.skills).shape == (4, 2)


def transitions_slice(transitions, start, stop):
    return jax.tree_util.tree_map(lambda v: v[start:stop], transitions)


@pytest.mark.parametrize("objective", ["forward_infonce", "vmf_forward"])
@pytest.mark.parametrize("branch_live_two_phase", [False, True])
def test_marginal_branch_runs_with_continuous_skills(objective, branch_live_two_phase) -> None:
    """`skill_start_mode=marginal_branch` with continuous
    skills. There is no library, so every relaunched branch lane draws a fresh random unit
    skill; the archive fills and refreshes on the same schedule as the discrete collector, the
    holdout is scored, and the class-count metrics degrade to row counts (Fano reads 0)."""
    env = _Env()
    skill = SkillLearner(
        SkillConfig(method="potential", observation_dim=3, skill_dim=2, discrete_skills=False,
                    num_skills=0, hidden_dims=(8,), layer_norm=False,
                    encoder_objective=objective, phi_norm_cap=0,
                    continuous_reward_temperature=True, reward_encoder="live",
                    continuous_reward_normalization="sphere" if objective == "vmf_forward" else "none")
    )
    sac = ContinuousSAC(SACConfig(3, 2, 2, hidden_dims=(8,), actor_layer_norm=False, critic_layer_norm=False, separate_skill0_actor=True, separate_skill0_critic=True, replace_skill_zero_reward_with_random_value=True))
    trainer = OffPolicyTrainer(
        env, sac, skill,
        OffPolicyConfig(rollout_steps=2, updates_per_chunk=1, batch_size=4, replay_capacity=32,
                        episode_length=4, representation_episodes=6, max_future_steps=4, potential_steps=2,
                        stagger_reset_groups=2, skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
                        branch_live_two_phase=branch_live_two_phase),
    )
    assert trainer.config.collector_potential_steps == (2 if branch_live_two_phase else 0)
    state = trainer.initialize(jax.random.PRNGKey(61))
    for _ in range(4):
        state, metrics = trainer.run(state)
    assert int(state.representation_replay.size) == 6
    assert float(metrics.branch.applied) == 1.0 and float(metrics.branch.collector_active) == 1.0
    assert state.staggered_branch is not None
    skills = np.asarray(state.staggered_branch.skills)
    assert np.allclose(np.linalg.norm(skills, axis=-1), 1.0, atol=1e-5), "branch skills are unit vectors"
    assert np.isfinite(float(metrics.branch.holdout_loss))
    if objective == "vmf_forward":
        assert float(metrics.branch.holdout_mi_lower_bound) == pytest.approx(
            -float(metrics.branch.holdout_loss), abs=1e-6
        )
        assert float(metrics.branch.holdout_intrinsic_reward) == pytest.approx(
            -float(metrics.branch.holdout_loss_clean), abs=1e-6
        )
    assert float(metrics.branch.holdout_mi_fano) == 0.0
    assert float(metrics.branch.target_count_min) == float(metrics.branch.target_count_max) == 4.0
    # steady state: one phase completes per segment; relaunched lanes get NEW random skills
    before = np.asarray(state.staggered_branch.skills)
    state, _ = trainer.run(state)
    state, steady = trainer.run(state)
    after = np.asarray(state.staggered_branch.skills)
    assert float(steady.branch.inserted_groups) == 2.0
    assert np.allclose(np.linalg.norm(after, axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(before, after), "completed lanes were relaunched with fresh skills"
    from empowerment_warp.training.sac.logging import HoldoutCycleAccumulator
    accumulator = HoldoutCycleAccumulator(1, 0)
    report = accumulator.add(steady)
    assert report is not None
    assert report["skill/mi_lower_bound_heldout"] == pytest.approx(
        float(steady.branch.holdout_mi_lower_bound)
    )
    with pytest.raises(ValueError, match="targets_per_anchor == 1"):
        OffPolicyTrainer(
            env, sac, skill,
            OffPolicyConfig(rollout_steps=2, updates_per_chunk=1, batch_size=4, replay_capacity=32,
                            episode_length=4, representation_episodes=6, max_future_steps=4, potential_steps=2,
                            stagger_reset_groups=2, skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
                            branch_targets_per_anchor=2),
        )
