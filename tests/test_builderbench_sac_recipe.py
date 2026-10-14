from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.agents.see_sac import ContinuousSAC, SACConfig
from empowerment_warp.core import Transition
from empowerment_warp.core.replay import (
    GEOMETRIC_ANCHOR_REPLAY_MODE,
)
from empowerment_warp.training.sac.config import (
    BUILDERBENCH_INCUMBENT_SAC,
    BUILDERBENCH_REFERENCE_COMMIT,
    ExperimentConfig,
    PRODUCTION_NUM_ENVS,
    PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO,
    STANDING_POLICY_REPLAY_CAPACITY,
    STANDING_REPRESENTATION_EPISODES,
    default_recipe,
)
from empowerment_warp.envs.builderbench import BuilderBenchWarpEnv




def test_episode_bag_is_independent_of_vector_width() -> None:
    for num_envs in (256, 2048):
        config = default_recipe("builderbench_corner3", "potential", num_envs=num_envs)
        assert config.representation_episodes == 3_334
        assert config.replay_capacity == 10_000_000
        assert config.reference_mismatches() == {}
        assert config.policy_update_sample_ratio == 4
    nonstaggered = default_recipe(
        "builderbench_corner3", "potential", num_envs=2048, staggered_resets=False,
        allow_reference_deviation=True,
    )
    assert nonstaggered.representation_episodes >= nonstaggered.num_envs
    with pytest.raises(ValueError, match="finish an episode on the same step"):
        default_recipe(
            "builderbench_corner3", "potential", num_envs=4096, staggered_resets=False,
            allow_reference_deviation=True,
        )


def test_config_without_replay_mode_loads_geometric_semantics() -> None:
    payload = default_recipe("builderbench_corner3", "potential").to_dict()
    payload.pop("potential_replay_mode")
    payload["allow_reference_deviation"] = False
    loaded = ExperimentConfig.from_dict(payload)
    assert loaded.potential_replay_mode == "legacy_geometric"
    assert loaded.allow_reference_deviation
    # The reference expectation is legacy_geometric, so this config registers no
    # replay-mode mismatch.
    assert "potential_replay_mode" not in loaded.reference_mismatches()


def test_builderbench_encoder_view_matches_block_only_layout() -> None:
    indices = BuilderBenchWarpEnv.encoder_observation_indices("object_pos")
    assert indices == (10, 11, 12, 13, 14, 15, 16, 17, 18)
    learner = SkillLearner(
        SkillConfig(
            method="potential",
            observation_dim=9,
            observation_indices=indices,
            skill_dim=3,
            discrete_skills=False,
            hidden_dims=(8,),
            layer_norm=False,
        )
    )
    observations = jnp.arange(2 * 51, dtype=jnp.float32).reshape(2, 51)
    projected = learner.encoder_observations(observations)
    np.testing.assert_array_equal(
        np.asarray(projected),
        np.asarray(observations[:, indices]),
    )


def test_warp_default_has_global_std_and_skill_mean_symmetry_break() -> None:
    learner = ContinuousSAC(
        SACConfig(
            observation_dim=5,
            action_dim=2,
            skill_dim=3,
            hidden_dims=(16,),
            actor_layer_norm=False,
            critic_layer_norm=True,
            actor_final_scale=0.01,
            skill_mean_init_std=0.2,
            state_dependent_std=False,
            use_anchor_in_policy=True,
            use_anchor_in_critic=True,
            target_entropy=-12.5,
            separate_skill0_actor=True,
            separate_skill0_critic=True,
            replace_skill_zero_reward_with_random_value=True,
        )
    )
    state = learner.initialize(jax.random.PRNGKey(0))
    observations = jnp.zeros((2, 5), dtype=jnp.float32)
    anchors = jnp.stack((jnp.zeros((5,)), jnp.ones((5,))), axis=0)
    skills = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    means, log_std = learner.actor.apply(
        {"params": state.actor_params}, observations, anchors, skills
    )
    assert not np.allclose(np.asarray(means[0]), np.asarray(means[1]))
    np.testing.assert_allclose(np.asarray(log_std), 0.0)
    assert "log_stds" in state.actor_params
    assert "skill_mean_adapter" in state.actor_params


def test_builderbench_default_is_the_frozen_recipe() -> None:
    config = default_recipe("builderbench_corner3", "potential")
    assert config.recipe_id == BUILDERBENCH_INCUMBENT_SAC
    manifest = config.reference_manifest()
    assert manifest["contract_fields_match"]
    assert not manifest["exact_reference_parity"]
    assert manifest["parity_with_declared_deviations"]
    assert manifest["reference_commit"] == BUILDERBENCH_REFERENCE_COMMIT
    assert set(manifest["intentional_deviations"]) == {
        "skill_mean_symmetry_break",
        "encoder_temporal_reference",
        "representation_archive_layout",
        "representation_archive_capacity",
        "policy_replay_capacity",
        "staggered_resets",
        "policy_update_sample_ratio",
        "potential_replay_semantics",
    }
    assert config.skill_mean_init_std == 0.2
    assert config.encoder_obs_mode == "object_pos"
    assert config.episode_length == 400
    assert config.potential_steps == 100
    assert config.replay_capacity == STANDING_POLICY_REPLAY_CAPACITY == 10_000_000
    assert config.representation_episodes == STANDING_REPRESENTATION_EPISODES == 3_334
    assert config.representation_episodes * (config.episode_length - config.potential_steps) == 1_000_200
    assert config.staggered_resets is True
    assert config.reward_encoder == "slow"
    assert config.actor_final_scale == 0.01
    assert config.phi_norm_cap == 0.0
    assert config.achieved_skill_relabel_min_prob == pytest.approx(2.0 / 30.0)
    assert config.achieved_skill_relabel_weight_mode == "effective_number"
    assert config.replace_skill_zero_reward_with_random_value
    assert config.separate_skill0_actor
    assert config.separate_skill0_critic
    # Potential rows replay with geometric anchors.
    assert config.potential_replay_mode == GEOMETRIC_ANCHOR_REPLAY_MODE
    assert manifest["potential_replay_mode"] == GEOMETRIC_ANCHOR_REPLAY_MODE
    assert manifest["intentional_deviations"]["skill_mean_symmetry_break"] == {
        "reference": 0.0,
        "warp_default": 0.2,
        "reason": "a per-skill action-mean symmetry break",
    }
    assert config.num_envs == PRODUCTION_NUM_ENVS == 2_048
    assert config.updates_per_chunk == 256
    assert config.policy_update_sample_ratio == PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO == 4
    with pytest.raises(ValueError, match="representation_episodes=64"):
        default_recipe("builderbench_corner3", "potential", representation_episodes=64)
