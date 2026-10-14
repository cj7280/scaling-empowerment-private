"""Directed information: I(Z^T -> S^T) = sum_t I(z; s_{t+1} | s_t).

The branch collector must be able to run a UNIT horizon, so the M skills in a
menu share their conditioning state s_t exactly (they only do so at the
anchor). That needs `branch_phase_groups` decoupled from the live collector's
reset staggering.
"""

from __future__ import annotations

import pytest

from empowerment_warp.core.replay import MARGINAL_BRANCH_SKILL_START_MODE
from empowerment_warp.training.sac.config import OffPolicyConfig


def _branch_config(**overrides) -> OffPolicyConfig:
    base = dict(
        skill_start_mode=MARGINAL_BRANCH_SKILL_START_MODE,
        episode_length=400,
        rollout_steps=8,
        potential_steps=100,
        stagger_reset_groups=50,
        representation_episodes=4096,
    )
    base.update(overrides)
    return OffPolicyConfig(**base)


def test_branch_phase_groups_defaults_to_the_live_derived_value() -> None:
    """Default behaviour is byte-for-byte what it was before the knob existed."""

    config = _branch_config()
    assert config.branch_phase_groups == 400 // 8 == 50
    assert config.branch_horizon == 300
    assert config.branch_steps_per_segment == 6


def test_branch_phase_groups_override_enables_a_unit_horizon() -> None:
    """H=1 is the shape exact 1-step conditional MI needs, and it was unreachable.

    `branch_phase_groups` was pinned to `episode_length // rollout_steps` = 50, so
    `branch_horizon % branch_phase_groups` could never be satisfied for H=1.
    """

    config = _branch_config(potential_steps=399, branch_phase_groups_override=1)
    assert config.branch_horizon == 1
    assert config.branch_phase_groups == 1
    assert config.branch_steps_per_segment == 1
    # Every lane completes its branch every segment, instead of num_envs/50.
    # Live reset staggering is untouched: still ceil(2048/50) lanes per group.
    assert config.max_simultaneous_completions(2048) == 41


def test_unit_horizon_is_rejected_without_the_override() -> None:
    """The divisibility guard sat after a `raise` and had never executed."""

    with pytest.raises(ValueError, match="divisible"):
        _branch_config(potential_steps=399)


def test_branch_horizon_must_be_positive() -> None:
    """An existing guard already covers a fully-consumed episode."""

    with pytest.raises(ValueError, match="at least one skill-on step"):
        _branch_config(potential_steps=400, branch_phase_groups_override=1)
