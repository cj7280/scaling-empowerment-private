from __future__ import annotations

import math
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillConfig
from empowerment_warp.core.replay import (
    MARGINAL_BRANCH_SKILL_START_MODE,
    GEOMETRIC_ANCHOR_REPLAY_MODE,
    TWO_PHASE_SKILL_START_MODE,
)
from empowerment_warp.training.sac.config import (
    ANTSOCCER_INCUMBENT_SAC,
    ANT_STANDARD_42_STATE_SAC,
    PRODUCTION_NUM_ENVS,
    PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO,
    default_recipe,
)


def test_antsoccer_closer_ball_recipe_is_guarded_and_resolved() -> None:
    config = default_recipe("ant_soccer", "potential", num_envs=2048)
    assert config.recipe_id == ANTSOCCER_INCUMBENT_SAC
    assert config.antsoccer_preset == "closer_ball"
    assert config.encoder_obs_mode == "ball_xy"
    assert config.phi_noise_std == 0.1
    assert config.skill_start_mode == TWO_PHASE_SKILL_START_MODE
    assert config.discrete_skill_library == "auto"
    assert config.policy_update_sample_ratio == PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO == 4
    assert config.updates_per_chunk == 64
    # Potential rows replay with geometric anchors.
    assert config.potential_replay_mode == GEOMETRIC_ANCHOR_REPLAY_MODE
    assert config.reference_mismatches() == {}
    assert config.reference_manifest()["intentional_deviations"][
        "encoder_training_noise"
    ]["warp_default"] == 0.1

    with pytest.raises(ValueError, match="AntSoccer SAC drift"):
        default_recipe("ant_soccer", "potential", encoder_obs_mode="full")


def test_ant_comparison_standard_42_value_state_recipe_is_guarded() -> None:
    config = default_recipe("ant", "potential")
    assert config.recipe_id == ANT_STANDARD_42_STATE_SAC
    assert config.antsoccer_preset == "standard"
    assert config.encoder_obs_mode == "full"
    assert config.phi_noise_std == 0.05
    assert config.skill_start_mode == TWO_PHASE_SKILL_START_MODE
    assert config.num_envs == PRODUCTION_NUM_ENVS == 2_048
    assert config.policy_update_sample_ratio == PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO == 4
    assert config.updates_per_chunk == 64
    # Potential rows replay with geometric anchors.
    assert config.potential_replay_mode == GEOMETRIC_ANCHOR_REPLAY_MODE
    assert config.reference_mismatches() == {}
    assert "encoder_training_noise" not in config.reference_manifest()[
        "intentional_deviations"
    ]

    with pytest.raises(ValueError, match="standard-reset 42-state Ant SAC drift"):
        default_recipe("ant", "potential", phi_noise_std=0.0)


def test_locomotion_r4_ratio_is_invariant_to_vector_width() -> None:
    for environment in ("ant", "ant_soccer"):
        for num_envs, expected_updates in ((256, 8), (2_048, 64)):
            config = default_recipe(environment, "potential", num_envs=num_envs)
            assert config.updates_per_chunk == expected_updates
            assert config.policy_update_sample_ratio == 4
            assert config.reference_mismatches() == {}


@pytest.mark.parametrize("environment", ("ant", "ant_soccer"))
def test_marginal_branch_recipe_resolves_scale_and_requires_deviation(
    environment: str,
) -> None:
    with pytest.raises(ValueError, match="drift detected"):
        default_recipe(
            environment,
            "potential",
            skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
        )
    config = default_recipe(
        environment,
        "potential",
        skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
        allow_reference_deviation=True,
    )
    assert config.resolved_live_potential_steps == 0
    assert config.resolved_branch_horizon == 300
    assert config.branch_simulator_to_live_ratio == 0.75
    assert config.resolved_branch_phase_groups == 50
    assert config.resolved_branch_steps_per_segment == 6
    assert config.resolved_branch_completions_per_segment == 41
    assert 48 * 41 + 2 * 40 == 2_048
    first_update_env_steps = (
        math.ceil(config.representation_episodes / config.num_envs)
        * config.episode_length
        * config.num_envs
    )
    assert first_update_env_steps == 1_638_400
    manifest = config.reference_manifest()
    assert manifest["mismatches"]["skill_start_mode"] == {
        "actual": MARGINAL_BRANCH_SKILL_START_MODE,
        "expected": TWO_PHASE_SKILL_START_MODE,
    }
    assert manifest["branch_interval_live_vector_steps"] == 400
    assert manifest["branch_phase_groups"] == 50
    assert manifest["branch_steps_per_segment"] == 6
    assert manifest["branch_completions_per_segment"] == 41
    assert manifest["representation_archive_unit"] == "marginal_anchor_branch"
    assert manifest["representation_active_row_capacity"] == 1_000_200
    assert manifest["branch_targets_per_anchor"] == 1


def test_standard_ball_only_pair_stays_rejected_and_past_center_is_accepted() -> None:
    # "standard" is shared with the base Ant env and stays fixed. The past-center ball
    # lives in its own preset, which lifts the dead-task premise.
    with pytest.raises(ValueError, match="known dead task"):
        default_recipe(
            "ant_soccer",
            "potential",
            antsoccer_preset="standard",
            encoder_obs_mode="ball_xy",
            allow_reference_deviation=True,
        )
    config = default_recipe(
        "ant_soccer",
        "potential",
        antsoccer_preset="past_center_ball",
        encoder_obs_mode="ball_xy",
        allow_reference_deviation=True,
    )
    assert config.antsoccer_preset == "past_center_ball"
    assert config.encoder_obs_mode == "ball_xy"


def test_branch_policy_replay_is_guarded_at_the_recipe_level() -> None:
    """The trainer-level guard fires too late to protect a launcher.

    `OffPolicyConfig` only validates when the trainer is constructed, which is
    after the environment is built; a launcher that asks for branch rows without
    marginal_branch must fail while resolving the recipe instead.
    """

    with pytest.raises(ValueError, match="requires skill_start_mode=marginal_branch"):
        default_recipe(
            "ant",
            "potential",
            num_envs=PRODUCTION_NUM_ENVS,
            branch_transitions_to_policy_replay=True,
            allow_reference_deviation=True,
        )

    config = default_recipe(
        "ant",
        "potential",
        num_envs=PRODUCTION_NUM_ENVS,
        skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
        branch_transitions_to_policy_replay=True,
        allow_reference_deviation=True,
    )
    assert config.reference_manifest()["branch_transitions_to_policy_replay"] is True
