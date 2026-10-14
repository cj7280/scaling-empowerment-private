from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from empowerment_warp.training.sac.logging import (
    HoldoutCycleAccumulator,
    summarize_holdout_event,
)


class _Branch(NamedTuple):
    holdout_cycle_eligible: jnp.ndarray
    holdout_sampled_rows: jnp.ndarray
    holdout_trajectories: jnp.ndarray
    holdout_physical_anchors: jnp.ndarray
    holdout_accuracy: jnp.ndarray
    holdout_loss: jnp.ndarray
    holdout_mi_lower_bound: jnp.ndarray
    holdout_mi_fano: jnp.ndarray
    holdout_intrinsic_reward: jnp.ndarray
    holdout_accuracy_clean: jnp.ndarray
    holdout_loss_clean: jnp.ndarray
    holdout_mi_lower_bound_clean: jnp.ndarray
    holdout_mi_fano_clean: jnp.ndarray


class _Metrics(NamedTuple):
    branch: _Branch


def _event(*, rows: float, accuracy: float, loss: float) -> _Metrics:
    return _Metrics(
        _Branch(
            holdout_cycle_eligible=jnp.asarray(1.0),
            holdout_sampled_rows=jnp.asarray(rows),
            holdout_trajectories=jnp.asarray(rows / 2.0),
            holdout_physical_anchors=jnp.asarray(1.0),
            holdout_accuracy=jnp.asarray(accuracy),
            holdout_loss=jnp.asarray(loss),
            holdout_mi_lower_bound=jnp.asarray(math.log(4.0) - loss),
            holdout_mi_fano=jnp.asarray(0.0),
            holdout_intrinsic_reward=jnp.asarray(0.5 * loss),
            holdout_accuracy_clean=jnp.asarray(accuracy + 0.1),
            holdout_loss_clean=jnp.asarray(loss - 0.25),
            holdout_mi_lower_bound_clean=jnp.asarray(
                math.log(4.0) - (loss - 0.25)
            ),
            holdout_mi_fano_clean=jnp.asarray(0.0),
        )
    )


def test_holdout_cycle_uses_all_events_and_weights_by_sampled_rows() -> None:
    accumulator = HoldoutCycleAccumulator(required_events=3, num_skills=4)
    assert accumulator.add(_event(rows=2.0, accuracy=0.5, loss=1.0)) is None
    assert accumulator.add(_event(rows=6.0, accuracy=0.25, loss=3.0)) is None
    summary = accumulator.add(_event(rows=2.0, accuracy=1.0, loss=2.0))
    assert summary is not None

    assert summary["skill/heldout_phase_events"] == 3.0
    assert summary["skill/heldout_sampled_rows"] == 10.0
    assert summary["skill/heldout_trajectories"] == 5.0
    assert summary["skill/heldout_physical_anchors"] == 3.0
    assert summary["skill/accuracy_heldout"] == pytest.approx(0.45)
    assert summary["skill/loss_heldout"] == pytest.approx(2.4)
    assert summary["skill/mi_lower_bound_heldout"] == pytest.approx(
        math.log(4.0) - 2.4
    )
    assert summary["skill/accuracy_heldout_clean"] == pytest.approx(0.55)
    assert summary["skill/loss_heldout_clean"] == pytest.approx(2.15)

    assert accumulator.events == 0
    assert accumulator.sampled_rows == 0.0


def test_single_phase_holdout_is_explicit_and_never_zero_filled() -> None:
    event = _event(rows=8.0, accuracy=0.25, loss=2.0)
    summary = summarize_holdout_event(event)
    assert summary["skill/accuracy_heldout_event"] == pytest.approx(0.25)
    assert summary["skill/heldout_sampled_rows_event"] == 8.0
    assert "skill/accuracy_heldout" not in summary

    inactive = _Metrics(
        event.branch._replace(holdout_cycle_eligible=jnp.asarray(0.0))
    )
    assert summarize_holdout_event(inactive) == {}


def test_vmf_holdout_uses_decoder_bound_without_discrete_offset_or_fano() -> None:
    accumulator = HoldoutCycleAccumulator(required_events=2, num_skills=0)
    for rows, loss in [(2.0, -1.0), (6.0, 3.0)]:
        event = _event(rows=rows, accuracy=0.25, loss=loss)
        event = _Metrics(event.branch._replace(
            holdout_mi_lower_bound=jnp.asarray(-loss),
            holdout_mi_lower_bound_clean=jnp.asarray(-loss + 0.25),
        ))
        summary = accumulator.add(event)
    assert summary is not None
    assert summary["skill/mi_lower_bound_heldout"] == pytest.approx(-2.0)
    assert summary["skill/mi_lower_bound_heldout_clean"] == pytest.approx(-1.75)
    assert summary["skill/mi_lower_bound_heldout"] == -summary["skill/loss_heldout"]
    assert not any("fano" in key for key in summary)
    assert not any("fano" in key for key in summarize_holdout_event(event))
    assert accumulator.mi_sum == accumulator.clean_mi_sum == 0.0


def test_potential_diagnostics_weight_updates_by_actual_phase_rows() -> None:
    from empowerment_warp.agents.see_sac import SACMetrics
    from empowerment_warp.training.sac.logging import summarize_potential_metrics

    # Three updates: no potential rows, one unit of weight, then three.
    metrics = SACMetrics(*(jnp.zeros(3) for _ in SACMetrics._fields))._replace(
        replay_weight=jnp.array([8., 8., 8.]),
        potential_weight=jnp.array([0., 1., 3.]),
        potential_skill_value_weight=jnp.array([0., 1., 3.]),
        potential_skill_value_total=jnp.array([0., 2., 30.]),
        potential_reward_total=jnp.array([0., 1., 15.]),
        potential_q_total=jnp.array([0., 3., 21.]),
        potential_target_total=jnp.array([0., 5., 27.]),
        potential_td_squared_total=jnp.array([0., 4., 12.]),
    )
    report = summarize_potential_metrics(metrics)
    assert report == pytest.approx({
        'potential/replay_fraction': 4/24,
        'potential/skill_value_mean': 8,
        'potential/reward_mean': 4,
        'potential/critic_q_mean': 6,
        'potential/critic_target_mean': 8,
        'potential/critic_td_rmse': 2,
    })
    assert summarize_potential_metrics(metrics._replace(potential_weight=jnp.zeros(3))) == {}



def test_potential_end_metrics_average_over_switch_events():
    """switch_* sums are divided by the number of switch events, not by completed episodes."""
    from empowerment_warp.training.sac.logging import _summarize_exploration
    totals = {"completed_episodes": 4.0, "agent_disp_xy_sum": 8.0, "ball_disp_xy_sum": 0.0,
              "switch_count": 5.0, "switch_center_dist_sum": 25.0, "switch_agent_ball_dist_sum": 2.5}
    summary = _summarize_exploration(totals, "ant_soccer")
    assert summary["exploration/episode_agent_disp_xy"] == 2.0
    assert summary["exploration/potential_end_center_distance"] == 5.0
    assert summary["exploration/potential_end_agent_ball_distance"] == 0.5
    assert "exploration/potential_end_center_distance" not in _summarize_exploration({"completed_episodes": 4.0, "switch_count": 0.0}, "ant")
