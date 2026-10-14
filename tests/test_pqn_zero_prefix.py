"""Skill-only PQN evaluation starts commanded branches at the reset state."""

from typing import NamedTuple

import numpy as np

from empowerment_warp.training.pqn import branch_protocol as protocol


class _MediaState(NamedTuple):
    achievements: np.ndarray
    step: np.ndarray


class _MediaEnv:
    _episode_limit = 3
    _achievement_key_by_col = ()

    def __init__(self):
        self._state = None
        self._episode_return = np.array(0.0)
        self._episode_length = np.array(0)

    def reset(self, *, seed):
        del seed
        self._state = _MediaState(np.zeros(1, np.float32), np.array(0))
        self._episode_return = np.array(0.0)
        self._episode_length = np.array(0)
        return np.zeros(3, np.float32), {}

    def step(self, action):
        del action
        step = int(self._state.step) + 1
        self._state = _MediaState(np.zeros(1, np.float32), np.array(step))
        self._episode_return = np.array(float(step))
        self._episode_length = np.array(step)
        return np.full(3, step, np.float32), 1.0, step >= self._episode_limit, False, {}

    def current_state(self):
        return self._state

    def render(self):
        return np.full((4, 4, 3), int(self._state.step), np.uint8)


class _RecordingActor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, skills, deterministic=False, **kwargs):
        self.calls.append((np.array(skills, copy=True), deterministic))
        return np.array([1.0, 0.0], np.float32)


def _media_rollouts(potential_steps):
    actor = _RecordingActor()
    rollouts = protocol._collect_jax_game_branch_skill_rollouts(
        _MediaEnv(),
        actor,
        {"benchmark_backend": "craftax", "skill_dim": 2,
         "potential_steps": potential_steps, "max_episode_steps": 3},
        np.eye(2, dtype=np.float32),
        count=2,
        warmup_steps=potential_steps,
        video_frame_skip=1,
        eval_temperature=1.0,
        eval_deterministic=True,
        supports_deterministic=True,
        seed=9,
    )
    return actor, rollouts


def test_zero_prefix_media_has_only_commanded_skills_from_reset():
    actor, rollouts = _media_rollouts(0)

    assert len(rollouts) == 2
    assert not any(rollout["is_potential_continuation"] for rollout in rollouts)
    assert all(rollout["branch_prefix_steps"] == 0 for rollout in rollouts)
    assert all(rollout["activation_frame"] == -1 for rollout in rollouts)
    assert all(np.linalg.norm(skill) > 0.0 and deterministic for skill, deterministic in actor.calls)


def test_positive_prefix_media_keeps_potential_prefix_and_continuation():
    actor, rollouts = _media_rollouts(2)

    assert len(rollouts) == 3
    assert rollouts[0]["is_potential_continuation"]
    assert all(rollout["branch_prefix_steps"] == 2 for rollout in rollouts)
    assert all(rollout["activation_frame"] >= 0 for rollout in rollouts)
    np.testing.assert_array_equal(actor.calls[0][0], np.zeros(2, np.float32))
    np.testing.assert_array_equal(actor.calls[1][0], np.zeros(2, np.float32))
    assert all(deterministic for _, deterministic in actor.calls)
