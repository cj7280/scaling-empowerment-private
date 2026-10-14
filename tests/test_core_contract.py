from __future__ import annotations

from typing import NamedTuple

from flax import serialization, struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.core import ReplayBuffer, Transition, initialize_collector, make_rollout_chunk
from empowerment_warp.core.replay import (
    BranchTrajectory,
    BranchTrajectoryArchive,
    EpisodeArchive,
    RepresentationTransition,
)
from empowerment_warp.core.rollout import (
    MarginalSource,
    StaggeredBranchState,
    branch_anchor_layout,
    make_branch_trajectory_rollout,
    make_staggered_branch_trajectory_rollout,
    potential_switch_steps,
    staggered_reset_offsets,
)


def _transition(batch_size: int = 4) -> Transition:
    x = jnp.arange(batch_size, dtype=jnp.float32)
    return Transition(
        observations=jnp.stack((x, x + 1), axis=-1),
        anchor_observations=jnp.stack((x, x + 1), axis=-1),
        actions=x[:, None],
        rewards=x,
        masks=jnp.ones_like(x),
        next_observations=jnp.stack((x + 1, x + 2), axis=-1),
        next_anchor_observations=jnp.stack((x, x + 1), axis=-1),
        dones=jnp.zeros((batch_size,), dtype=jnp.bool_),
        skills=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        next_skills=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        loss_weights=jnp.ones((batch_size,), dtype=jnp.float32),
        episode_ids=jnp.arange(batch_size, dtype=jnp.int32),
        timesteps=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def test_sac_rejects_unsupported_environments_and_algorithms() -> None:
    from dataclasses import replace
    from empowerment_warp.training.sac.config import default_recipe
    config = default_recipe("builderbench_corner3", "potential")
    for environment in ("craftax:classic-symbolic", "shape_crush:symbolic"):
        with pytest.raises(ValueError, match="unknown environment"):
            replace(config, environment=environment)
    for method in ("diayn", "metra", "dads", "csf"):
        with pytest.raises(ValueError, match="unknown algorithm"):
            replace(config, algorithm=method)


def test_replay_insert_wrap_and_sample() -> None:
    replay = ReplayBuffer(capacity=6)
    state = replay.initialize(_transition())
    insert = jax.jit(replay.insert)
    state = insert(state, _transition())
    state = insert(state, _transition())
    assert int(state.size) == 6
    assert int(state.cursor) == 2
    sample = jax.jit(lambda value, key: replay.sample(value, key, 3))(
        state,
        jax.random.PRNGKey(0),
    )
    assert sample.observations.shape == (3, 2)
    assert np.isfinite(np.asarray(sample.observations)).all()


def _archive_rows(step: int, num_envs: int, episode_length: int) -> RepresentationTransition:
    envs = jnp.arange(num_envs, dtype=jnp.int32)
    timestep = step % episode_length
    value = (envs[:, None] * 1000 + step).astype(jnp.float32) * jnp.ones((1, 2))
    return RepresentationTransition(
        observations=value,
        anchor_observations=value,
        next_observations=value + 1.0,
        skills=jnp.ones((num_envs, 1), dtype=jnp.float32),
        episode_ids=envs + num_envs * (step // episode_length),
        timesteps=jnp.full((num_envs,), timestep, dtype=jnp.int32),
        dones=jnp.full((num_envs,), timestep == episode_length - 1),
    )


def test_episode_archive_keeps_the_most_recent_completed_episodes() -> None:
    archive = EpisodeArchive(num_envs=3, num_episodes=4, episode_length=4)
    state = archive.initialize(_archive_rows(0, 3, 4))
    insert = jax.jit(archive.insert)
    for step in range(0, 8, 2):
        chunk = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, b), axis=0),
            _archive_rows(step, 3, 4),
            _archive_rows(step + 1, 3, 4),
        )
        state = insert(state, chunk)
        if step + 1 < 3:
            assert int(state.size) == 0  # nothing complete before step 3
    # six episodes ended (3 lanes x 2); the bag keeps the last four, FIFO.
    assert int(state.size) == 4 and int(state.cursor) == 2 and int(state.active_rows) == 16
    assert np.all(np.asarray(state.lengths) == 4)
    stored_steps = np.asarray(state.episodes.observations)[:, :, 0] % 1000
    for slot in range(4):
        np.testing.assert_array_equal(stored_steps[slot], stored_steps[slot, 0] + np.arange(4))
    first_steps = sorted(int(v) for v in stored_steps[:, 0])
    assert first_steps == [0, 4, 4, 4]
    slots, rows, valid = jax.jit(lambda s, k: archive.sample_active(s, k, 6))(state, jax.random.PRNGKey(1))
    assert np.all(np.asarray(valid))
    assert np.all(np.asarray(rows) < 4) and np.all(np.asarray(slots) < 4)


def test_branch_archive_prefix_insert_masks_padding_across_wrap() -> None:
    archive = BranchTrajectoryArchive(num_groups=5, horizon=2)
    state = archive.initialize(
        jnp.zeros((4, 1), dtype=jnp.float32),
        jnp.zeros((4, 1), dtype=jnp.float32),
    )

    def branch(offset: int) -> BranchTrajectory:
        anchors = jnp.arange(offset, offset + 4, dtype=jnp.float32)[:, None]
        return BranchTrajectory(
            anchors=anchors,
            skills=anchors + 100.0,
            outcomes=jnp.broadcast_to(anchors[None], (2, 4, 1)),
            valid=jnp.ones((2, 4), dtype=jnp.bool_),
        )

    state = jax.jit(archive.insert)(state, branch(0))
    assert int(state.fresh_group_count) == 4
    unchanged = jax.jit(archive.insert_prefix)(
        state,
        branch(20),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert int(unchanged.fresh_group_count) == 4
    state = jax.jit(archive.insert_prefix)(
        state,
        branch(10),
        jnp.asarray(3, dtype=jnp.int32),
    )

    assert int(state.cursor) == 2
    assert int(state.size) == 5
    assert int(state.active_rows) == 10
    assert int(state.fresh_group_count) == 3
    np.testing.assert_array_equal(
        np.asarray(state.anchors[:, 0]),
        [11.0, 12.0, 2.0, 3.0, 10.0],
    )
    assert np.all(np.asarray(state.lengths) == 2)


def test_staggered_offsets_form_equal_reset_groups() -> None:
    offsets = np.asarray(staggered_reset_offsets(400, 8, 4))
    np.testing.assert_array_equal(offsets, [0, 0, 100, 100, 200, 200, 300, 300])
    np.testing.assert_array_equal(np.asarray(staggered_reset_offsets(400, 8, 0)), 0)


def test_staggered_offsets_balance_production_width_remainder() -> None:
    offsets = np.asarray(staggered_reset_offsets(400, 2_048, 50))
    unique_offsets, counts = np.unique(offsets, return_counts=True)
    np.testing.assert_array_equal(unique_offsets, np.arange(50) * 8)
    assert counts.min() == 40
    assert counts.max() == 41
    assert counts.sum() == 2_048


class _BranchConfig(NamedTuple):
    num_envs: int


class _BranchEnv:
    config = _BranchConfig(num_envs=4)

    @staticmethod
    def step(state, actions):
        return state.replace(
            obs=state.obs + actions,
            reward=jnp.zeros((4,), dtype=jnp.float32),
            step_count=state.step_count + 1,
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

    @staticmethod
    def gather(state, lane_indices):
        return jax.tree_util.tree_map(lambda value: value[lane_indices], state)


def _branch_source(value: float) -> MarginalSource:
    return MarginalSource(
        env_state=_FakeState(
            obs=jnp.full((4, 2), value, dtype=jnp.float32),
            reward=jnp.zeros((4,), dtype=jnp.float32),
            terminated=jnp.zeros((4,), dtype=jnp.bool_),
            truncated=jnp.zeros((4,), dtype=jnp.bool_),
            step_count=jnp.full((4,), 7, dtype=jnp.int32),
        ),
        skills=jnp.full((4, 2), value, dtype=jnp.float32),
    )


def test_staggered_branch_uses_current_actor_and_rejects_startup_stubs() -> None:
    skill_library = jnp.eye(2, dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del anchors, skills, keys
        return jnp.ones_like(obs) * params

    initialize, run = make_staggered_branch_trajectory_rollout(
        _BranchEnv(),
        actions,
        lambda obs: obs,
        horizon=4,
        chunk_steps=2,
        phase_groups=2,
        skill_sampler=lambda key, count: jax.random.normal(key, (count, 2)),
    )
    template = StaggeredBranchState(
        env_state=_branch_source(-1.0).env_state,
        policy_anchors=jnp.zeros((4, 2), dtype=jnp.float32),
        encoder_anchors=jnp.zeros((4, 2), dtype=jnp.float32),
        source_skills=jnp.zeros((4, 2), dtype=jnp.float32),
        skills=jnp.zeros((4, 2), dtype=jnp.float32),
        outcomes=jnp.zeros((4, 4, 2), dtype=jnp.float32),
        valid=jnp.zeros((4, 4), dtype=jnp.bool_),
        progress=jnp.zeros((4,), dtype=jnp.int32),
        genuine=jnp.zeros((4,), dtype=jnp.bool_),
        alive=jnp.zeros((4,), dtype=jnp.bool_),
        active=jnp.asarray(False),
        skill_shift=jnp.asarray(0, dtype=jnp.int32),
        policy_outcomes=jnp.zeros((4, 4, 2), dtype=jnp.float32),
        actions=jnp.zeros((4, 4, 2), dtype=jnp.float32),
        terminated=jnp.zeros((4, 4), dtype=jnp.bool_),
        truncated=jnp.zeros((4, 4), dtype=jnp.bool_),
        alive_rows=jnp.zeros((4, 4), dtype=jnp.bool_),
        emit_index=jnp.asarray(0, dtype=jnp.int32),
        anchor_xy=jnp.zeros((4, 2), dtype=jnp.float32),
    )
    target_skills = skill_library[jnp.asarray([0, 1, 0, 1])]
    state = initialize(template, _branch_source(0.0), target_skills)
    np.testing.assert_array_equal(state.progress, [0, 0, 2, 2])
    np.testing.assert_array_equal(state.genuine, [True, True, False, False])

    # The offset group finishes first, but its missing prefix makes it invalid.
    state, _, _, completed, discarded, _, _, _, _ = run(
        jnp.asarray(1.0), state, _branch_source(10.0), jax.random.PRNGKey(20)
    )
    assert int(completed) == 0
    assert int(discarded) == 2
    np.testing.assert_array_equal(state.progress, [2, 2, 0, 0])
    np.testing.assert_array_equal(state.genuine, [True, True, True, True])

    # Group zero's trajectory joins two actor versions: +1 for its first two
    # steps and +2 for its final two. Its ordered outcomes must reflect both.
    state, branch, _, completed, discarded, _, _, _, _ = run(
        jnp.asarray(2.0), state, _branch_source(20.0), jax.random.PRNGKey(21)
    )
    assert int(completed) == 2
    assert int(discarded) == 0
    np.testing.assert_allclose(branch.anchors, 0.0)
    np.testing.assert_allclose(branch.outcomes[:, 0, 0], [1.0, 2.0, 4.0, 6.0])
    np.testing.assert_array_equal(branch.skills, target_skills[:2])

    # The reset group started from the then-current marginal source (10), not
    # from the original initialization source or the source of this call.
    state, branch, _, completed, _, _, _, _, _ = run(
        jnp.asarray(3.0), state, _branch_source(30.0), jax.random.PRNGKey(22)
    )
    assert int(completed) == 2
    np.testing.assert_allclose(branch.anchors, 10.0)
    np.testing.assert_allclose(branch.outcomes[:, 0, 0], [12.0, 14.0, 17.0, 20.0])


@struct.dataclass
class _FakeState:
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array


class _FakeConfig(NamedTuple):
    num_envs: int


class _FakeEnv:
    config = _FakeConfig(num_envs=3)

    def reset(self, keys):
        del keys
        return _FakeState(
            obs=jnp.zeros((3, 2), dtype=jnp.float32),
            reward=jnp.zeros((3,), dtype=jnp.float32),
            terminated=jnp.zeros((3,), dtype=jnp.bool_),
            truncated=jnp.zeros((3,), dtype=jnp.bool_),
            step_count=jnp.zeros((3,), dtype=jnp.int32),
        )

    def step(self, state, actions):
        step_count = state.step_count + 1
        return _FakeState(
            obs=state.obs + actions,
            reward=jnp.sum(actions, axis=-1),
            terminated=jnp.zeros((3,), dtype=jnp.bool_),
            truncated=step_count >= 2,
            step_count=step_count,
        )

    @staticmethod
    def where_done(state, reset_state):
        done = state.terminated | state.truncated
        return jax.tree_util.tree_map(
            lambda current, reset: jnp.where(
                done.reshape(done.shape + (1,) * (current.ndim - done.ndim)),
                reset,
                current,
            ),
            state,
            reset_state,
        )


def test_rollout_chunk_stays_scannable_across_resets() -> None:
    env = _FakeEnv()

    def sample_skills(key, count):
        return jax.random.normal(key, (count, 1))

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, keys
        return jnp.ones_like(obs) * skills

    collector = initialize_collector(env, jax.random.PRNGKey(0), sample_skills)
    run = make_rollout_chunk(env, actions, sample_skills, chunk_steps=5)
    collector, transitions, exploration = run({}, collector)
    assert transitions.observations.shape == (15, 2)
    assert int(collector.next_episode_id) == 9
    assert np.all(np.asarray(collector.env_state.step_count) == 1)
    assert float(exploration.completed_episodes) == 6.0


def test_rollout_uses_zero_skill_prefix_then_anchors_activation_state() -> None:
    env = _FakeEnv()

    def sample_skills(key, count):
        del key
        return jnp.ones((count, 1), dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, keys
        return jnp.ones_like(obs) * skills

    collector = initialize_collector(
        env,
        jax.random.PRNGKey(4),
        sample_skills,
        potential_steps=1,
    )
    run = make_rollout_chunk(
        env,
        actions,
        sample_skills,
        chunk_steps=2,
        potential_steps=1,
    )
    _, transitions, _ = run({}, collector)
    first, second = np.asarray(transitions.skills).reshape(2, 3, 1)
    np.testing.assert_array_equal(first, 0.0)
    np.testing.assert_array_equal(second, 1.0)
    next_skills = np.asarray(transitions.next_skills).reshape(2, 3, 1)
    np.testing.assert_array_equal(next_skills[0], 1.0)
    np.testing.assert_allclose(
        np.asarray(transitions.next_anchor_observations).reshape(2, 3, 2)[0],
        np.asarray(transitions.next_observations).reshape(2, 3, 2)[0],
    )


def test_random_potential_switch_is_episode_keyed_and_inclusive() -> None:
    ids = jnp.arange(4096, dtype=jnp.int32)
    draws = potential_switch_steps(ids, 100, 200, 17)
    values = np.asarray(draws)
    assert values.min() == 0 and values.max() == 200
    assert abs(values.mean() - 100.0) < 5.0
    np.testing.assert_array_equal(
        potential_switch_steps(ids[::-1], 100, 200, 17), draws[::-1]
    )
    assert int(potential_switch_steps(jnp.asarray(13), 100, 200, 17)) == values[13]
    assert not np.array_equal(values, potential_switch_steps(ids, 100, 200, 18))
    np.testing.assert_array_equal(potential_switch_steps(ids, 100, 0, 17), 100)


def test_random_potential_switch_rollout_reset_anchor_and_resume() -> None:
    env = _FakeEnv()

    def sample_skills(key, count):
        del key
        return jnp.ones((count, 1), dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, skills, keys
        return jnp.ones_like(obs)

    kwargs = dict(potential_steps=1, random_potential_switch_max_steps=1, potential_switch_seed=19)
    initial = initialize_collector(env, jax.random.PRNGKey(23), sample_skills, **kwargs)
    full = make_rollout_chunk(env, actions, sample_skills, chunk_steps=8, **kwargs)
    full_state, rows, _ = full({}, initial)
    first = make_rollout_chunk(env, actions, sample_skills, chunk_steps=3, **kwargs)
    rest = make_rollout_chunk(env, actions, sample_skills, chunk_steps=5, **kwargs)
    partial, first_rows, _ = first({}, initial)
    restored = serialization.from_bytes(partial, serialization.to_bytes(partial))
    resumed, last_rows, _ = rest({}, restored)
    joined = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b)), first_rows, last_rows)
    for expected, actual in zip(jax.tree_util.tree_leaves((full_state, rows)),
                                jax.tree_util.tree_leaves((resumed, joined)), strict=True):
        np.testing.assert_array_equal(expected, actual)

    boundaries = np.asarray(potential_switch_steps(rows.episode_ids, 1, 1, 19))
    timesteps = np.asarray(rows.timesteps)
    assert set(boundaries) == {0, 1}
    np.testing.assert_array_equal(np.asarray(rows.skills[:, 0]) > 0, timesteps >= boundaries)
    np.testing.assert_array_equal(np.asarray(rows.next_skills[:, 0]) > 0, timesteps + 1 >= boundaries)
    np.testing.assert_array_equal(rows.anchor_observations[:, 0], np.minimum(timesteps, boundaries))
    np.testing.assert_array_equal(rows.next_anchor_observations[:, 0], np.minimum(timesteps + 1, boundaries))
    # New IDs after resets determine new draws; no schedule/RNG fields need checkpointing.
    assert len(np.unique(np.asarray(rows.episode_ids))) == 12


def test_random_potential_switch_staggered_start_and_fixed_equivalence() -> None:
    env = _FakeEnv()

    def sample_skills(key, count):
        return jax.random.normal(key, (count, 1))

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, skills, keys
        return jnp.ones_like(obs)

    key = jax.random.PRNGKey(24)
    fixed = initialize_collector(env, key, sample_skills, potential_steps=1)
    explicit = initialize_collector(env, key, sample_skills, potential_steps=1,
                                    random_potential_switch_max_steps=0, potential_switch_seed=77)
    run = make_rollout_chunk(env, actions, sample_skills, chunk_steps=5, potential_steps=1)
    explicit_run = make_rollout_chunk(env, actions, sample_skills, chunk_steps=5, potential_steps=1,
                                     random_potential_switch_max_steps=0, potential_switch_seed=77)
    for expected, actual in zip(jax.tree_util.tree_leaves(run({}, fixed)),
                                jax.tree_util.tree_leaves(explicit_run({}, explicit)), strict=True):
        np.testing.assert_array_equal(expected, actual)
    staggered = initialize_collector(
        env, key, sample_skills, potential_steps=1, episode_length=2, stagger_groups=2,
        random_potential_switch_max_steps=1, potential_switch_seed=77,
    )
    expected_active = staggered.env_state.step_count >= potential_switch_steps(staggered.episode_ids, 1, 1, 77)
    np.testing.assert_array_equal(jnp.linalg.norm(staggered.skills, axis=-1) > 0, expected_active)
    np.testing.assert_array_equal(staggered.episode_genuine, staggered.env_state.step_count == 0)


def test_potential_phase_anchor_tracks_the_current_state() -> None:
    """The SAC potential reward is E_z[min_i Q_i(s, anchor, z, a)] evaluated on skill-off
    rows (`sac.py` `_random_skill_value`), and under the pinned `legacy_geometric` replay
    it passes `batch.anchor_observations` straight from the buffer. That is only the
    empowerment OF s -- the PQN form E_z max_a Q(s, s, z, a) -- because the collector
    resets the anchor every step while the skill is off (`anchor_reset = ... | ~next_skill_on`).
    Nothing else asserts that invariant, so a change to the anchor-reset rule would
    silently redefine the potential. Skill-ON rows must keep the anchor frozen at activation.
    """

    num_envs, episode_length, potential_steps = 3, 10, 4

    class _LongEpisodeEnv:
        config = _FakeConfig(num_envs=num_envs)

        def reset(self, keys):
            del keys
            return _FakeState(
                obs=jnp.zeros((num_envs, 2), dtype=jnp.float32),
                reward=jnp.zeros((num_envs,), dtype=jnp.float32),
                terminated=jnp.zeros((num_envs,), dtype=jnp.bool_),
                truncated=jnp.zeros((num_envs,), dtype=jnp.bool_),
                step_count=jnp.zeros((num_envs,), dtype=jnp.int32),
            )

        def step(self, state, actions):
            step_count = state.step_count + 1
            return _FakeState(
                obs=state.obs + actions,
                reward=jnp.sum(actions, axis=-1),
                terminated=jnp.zeros((num_envs,), dtype=jnp.bool_),
                truncated=step_count >= episode_length,
                step_count=step_count,
            )

        @staticmethod
        def where_done(state, reset_state):
            done = state.terminated | state.truncated
            return jax.tree_util.tree_map(
                lambda current, reset: jnp.where(
                    done.reshape(done.shape + (1,) * (current.ndim - done.ndim)),
                    reset,
                    current,
                ),
                state,
                reset_state,
            )

    env = _LongEpisodeEnv()

    def sample_skills(key, count):
        return jax.random.normal(key, (count, 1)) + 3.0

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, skills, keys
        return jnp.ones_like(obs) * 0.25

    collector = initialize_collector(
        env, jax.random.PRNGKey(0), sample_skills, potential_steps=potential_steps
    )
    run = make_rollout_chunk(
        env, actions, sample_skills, chunk_steps=20, potential_steps=potential_steps
    )
    _, transitions, _ = run({}, collector)

    observations = np.asarray(transitions.observations).reshape(-1, 2)
    anchors = np.asarray(transitions.anchor_observations).reshape(-1, 2)
    skills = np.asarray(transitions.skills).reshape(observations.shape[0], -1)
    skill_off = np.linalg.norm(skills, axis=-1) <= 1e-6

    assert skill_off.any() and (~skill_off).any()
    np.testing.assert_allclose(observations[skill_off], anchors[skill_off])
    assert np.abs(observations[~skill_off] - anchors[~skill_off]).max() > 0.0


class _MotionMetrics(NamedTuple):
    agent_x: jax.Array
    agent_y: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array


@struct.dataclass
class _MotionState:
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array
    metrics: _MotionMetrics


class _MotionEnv:
    config = _FakeConfig(num_envs=2)

    def reset(self, keys):
        del keys
        zeros = jnp.zeros((2,), dtype=jnp.float32)
        return _MotionState(
            obs=jnp.zeros((2, 1), dtype=jnp.float32),
            reward=zeros,
            terminated=jnp.zeros((2,), dtype=jnp.bool_),
            truncated=jnp.zeros((2,), dtype=jnp.bool_),
            step_count=jnp.zeros((2,), dtype=jnp.int32),
            metrics=_MotionMetrics(zeros, zeros, zeros, zeros),
        )

    def step(self, state, actions):
        del actions
        step_count = state.step_count + 1
        return state.replace(
            obs=state.obs + 1.0,
            truncated=step_count >= 2,
            step_count=step_count,
            metrics=_MotionMetrics(
                state.metrics.agent_x + 1.0,
                state.metrics.agent_y,
                state.metrics.ball_x + 2.0,
                state.metrics.ball_y,
            ),
        )

    @staticmethod
    def where_done(state, reset_state):
        done = state.terminated | state.truncated
        return jax.tree_util.tree_map(
            lambda current, reset: jnp.where(
                done.reshape(done.shape + (1,) * (current.ndim - done.ndim)),
                reset,
                current,
            ),
            state,
            reset_state,
        )


def test_rollout_reports_reference_agent_and_ball_displacement() -> None:
    env = _MotionEnv()

    def sample_skills(key, count):
        del key
        return jnp.ones((count, 1), dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, skills, keys
        return jnp.zeros_like(obs)

    collector = initialize_collector(env, jax.random.PRNGKey(5), sample_skills)
    run = make_rollout_chunk(env, actions, sample_skills, chunk_steps=2)
    _, _, exploration = run({}, collector)
    assert float(exploration.completed_episodes) == 2.0
    assert float(exploration.agent_disp_xy_sum) == 4.0
    assert float(exploration.ball_disp_xy_sum) == 8.0


def test_rollout_accumulates_phase_split_state_coverage() -> None:
    """The collector carry holds a cumulative visited-cell grid per track, split by the phase
    (potential z=0 vs skill) the lane was in when it visited the cell; runner logs the counts."""
    from empowerment_warp.core.rollout import coverage_counts, coverage_track_specs

    env = _MotionEnv()   # agent x advances 1 m/step, ball x 2 m/step; episodes of 2 steps

    def sample_skills(key, count):
        del key
        return jnp.ones((count, 1), dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, skills, keys
        return jnp.zeros_like(obs)

    # potential_steps=1: step 1 of every episode runs z=0, step 2 runs the skill.
    collector = initialize_collector(env, jax.random.PRNGKey(5), sample_skills, potential_steps=1)
    specs = coverage_track_specs(env, collector.env_state)
    assert [name for name, _, _ in specs] == ["agent_xy", "ball_xy"]
    assert collector.coverage_grids.shape == (2, 2, 100 * 100)
    run = make_rollout_chunk(env, actions, sample_skills, chunk_steps=2, potential_steps=1)
    y0, yb = float(collector.env_state.metrics.agent_y[0]), float(collector.env_state.metrics.ball_y[0])
    collector, _, exploration = run({}, collector)
    # Potential-end distances: both lanes switch to a skill after step 1, where the agent is at (1, y0)
    # and the ball at (2, yb); the sums are over the two switch events.
    assert float(exploration.switch_count) == 2.0
    assert float(exploration.switch_center_dist_sum) == pytest.approx(2.0 * float(np.hypot(1.0 - 10.0, y0 - 10.0)), rel=1e-5)
    assert float(exploration.switch_agent_ball_dist_sum) == pytest.approx(2.0 * float(np.hypot(1.0 - 2.0, y0 - yb)), rel=1e-5)
    assert float(exploration.switch_gripper_cube_dist_sum) == 0.0     # no objects in this env
    counts = coverage_counts(collector.coverage_grids, specs)
    # Positions after step 1 (x=1, potential phase) and step 2 (x=2, skill phase), both lanes identical.
    assert counts["exploration/cells_visited_agent_xy_potential_phase"] == 1.0
    assert counts["exploration/cells_visited_agent_xy_skill_phase"] == 1.0
    assert counts["exploration/cells_visited_agent_xy_total"] == 2.0
    assert counts["exploration/cells_visited_ball_xy_total"] == 2.0   # ball at x=2 then x=4
    # Cumulative: a second chunk revisits the same cells and adds nothing.
    collector, _, _ = run({}, collector)
    assert coverage_counts(collector.coverage_grids, specs)["exploration/cells_visited_agent_xy_total"] == 2.0
    # Envs without world-frame positions carry no grid and log nothing.
    plain = initialize_collector(_FakeEnv(), jax.random.PRNGKey(0), lambda k, n: jax.random.normal(k, (n, 1)))
    assert plain.coverage_grids is None
    assert coverage_counts(None, ()) == {}


@struct.dataclass
class _ObjectData:
    object_pos: jax.Array


class _ObjectMetrics(NamedTuple):
    obj_moved: jax.Array


@struct.dataclass
class _ObjectState:
    data: _ObjectData
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array
    initial_object_pos: jax.Array
    metrics: _ObjectMetrics


class _ObjectEnv:
    config = _FakeConfig(num_envs=2)

    def reset(self, keys):
        del keys
        object_pos = jnp.zeros((2, 3, 3), dtype=jnp.float32)
        return _ObjectState(
            data=_ObjectData(object_pos),
            obs=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.zeros((2,), dtype=jnp.float32),
            terminated=jnp.zeros((2,), dtype=jnp.bool_),
            truncated=jnp.zeros((2,), dtype=jnp.bool_),
            step_count=jnp.zeros((2,), dtype=jnp.int32),
            initial_object_pos=object_pos,
            metrics=_ObjectMetrics(jnp.zeros((2,), dtype=jnp.float32)),
        )

    def step(self, state, actions):
        del actions
        step_count = state.step_count + 1
        delta = jnp.zeros_like(state.data.object_pos).at[:, 0, 0].set(0.02)
        object_pos = state.data.object_pos + delta
        moved = jnp.any(
            jnp.linalg.norm(object_pos - state.initial_object_pos, axis=-1) > 0.01,
            axis=-1,
        )
        return state.replace(
            data=_ObjectData(object_pos),
            truncated=step_count >= 2,
            step_count=step_count,
            metrics=_ObjectMetrics(moved.astype(jnp.float32)),
        )

    @staticmethod
    def _object_positions(data):
        return data.object_pos

    @staticmethod
    def where_done(state, reset_state):
        done = state.terminated | state.truncated
        return jax.tree_util.tree_map(
            lambda current, reset: jnp.where(
                done.reshape(done.shape + (1,) * (current.ndim - done.ndim)),
                reset,
                current,
            ),
            state,
            reset_state,
        )


def test_rollout_reports_reference_builderbench_object_metrics() -> None:
    env = _ObjectEnv()

    def sample_skills(key, count):
        del key
        return jnp.ones((count, 1), dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, skills, keys
        return jnp.zeros_like(obs)

    collector = initialize_collector(env, jax.random.PRNGKey(6), sample_skills)
    run = make_rollout_chunk(env, actions, sample_skills, chunk_steps=2)
    collector, _, exploration = run({}, collector)
    from empowerment_warp.core.rollout import coverage_counts, coverage_track_specs

    counts = coverage_counts(collector.coverage_grids, coverage_track_specs(env, collector.env_state))
    for name in ("cube_xy", "block_0_xyz", "block_1_xyz", "block_2_xyz", "gripper_xyz"):
        assert counts[f"exploration/cells_visited_{name}_total"] == 1
    assert float(exploration.completed_episodes) == 2.0
    assert float(exploration.obj_moved_sum) == 2.0
    assert float(exploration.cube_disp_xyz_sum) == pytest.approx(2.0 * 0.04 / 3.0)
    assert float(exploration.cube_disp_max_xyz_sum) == pytest.approx(2.0 * 0.04)
    assert float(exploration.cube_disp_xy_sum) == pytest.approx(2.0 * 0.04 / 3.0)
    assert float(exploration.cube_disp_max_xy_sum) == pytest.approx(2.0 * 0.04)


def test_bootstrap_branch_shares_the_complete_start_state() -> None:
    def actions(params, obs, anchors, skills, keys):
        del anchors, skills, keys
        return jnp.ones_like(obs) * params

    def skills(key, count):
        del key
        return jnp.eye(2, dtype=jnp.float32)[jnp.arange(count) % 2]

    run = make_branch_trajectory_rollout(
        _BranchEnv(),
        actions,
        skills,
        lambda obs: obs,
        horizon=1,
        targets_per_anchor=2,
    )
    source = _branch_source(0.0)
    distinct = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    source = source.replace(env_state=source.env_state.replace(obs=distinct))
    branch = run(jnp.asarray(1.0), source, jax.random.PRNGKey(91))

    np.testing.assert_array_equal(branch.anchors[0], branch.anchors[1])
    np.testing.assert_array_equal(branch.anchors[2], branch.anchors[3])
    np.testing.assert_array_equal(branch.outcomes[0, 0], branch.outcomes[0, 1])
    np.testing.assert_array_equal(branch.outcomes[0, 2], branch.outcomes[0, 3])


def test_branch_anchor_blocks_do_not_cross_completion_phases() -> None:
    anchors, block_indices, positions, _ = branch_anchor_layout(2_048, 50, 10)
    lanes = np.arange(2_048)
    phases = lanes * 50 // 2_048
    anchors = np.asarray(anchors)
    block_indices = np.asarray(block_indices)
    positions = np.asarray(positions)

    block_sizes = []
    for block in np.unique(block_indices):
        members = lanes[block_indices == block]
        assert np.unique(phases[members]).size == 1
        assert np.all(anchors[members] == members[0])
        np.testing.assert_array_equal(positions[members], np.arange(len(members)))
        block_sizes.append(len(members))
    assert min(block_sizes) == 8
    assert max(block_sizes) == 10


def test_separate_lanes_collector_never_activates_potential_lanes() -> None:
    """`potential_lane_fraction` (the no-switching arm): potential lanes run
    z=0 for whole episodes, skill lanes start their skill at reset; the flag is redrawn only
    on reset; fraction 0 is byte-identical to the two-phase collector."""
    env = _FakeEnv()

    def sample_skills(key, count):
        del key
        return jnp.ones((count, 1), dtype=jnp.float32)

    def actions(params, obs, anchors, skills, keys):
        del params, anchors, keys
        return jnp.ones_like(obs) * skills

    base = initialize_collector(env, jax.random.PRNGKey(4), sample_skills, potential_steps=0)
    zero = initialize_collector(env, jax.random.PRNGKey(4), sample_skills, potential_steps=0, potential_lane_fraction=0.0)
    np.testing.assert_array_equal(np.asarray(base.skills), np.asarray(zero.skills))
    assert not bool(np.any(np.asarray(zero.potential_lane)))
    run0 = make_rollout_chunk(env, actions, sample_skills, chunk_steps=4, potential_steps=0)
    runz = make_rollout_chunk(env, actions, sample_skills, chunk_steps=4, potential_steps=0, potential_lane_fraction=0.0)
    _, t0, _ = run0({}, base); _, tz, _ = runz({}, zero)
    np.testing.assert_array_equal(np.asarray(t0.skills), np.asarray(tz.skills))
    # Every lane a potential lane (fraction -> 1): no skill ever activates, anchors track the state.
    full = initialize_collector(env, jax.random.PRNGKey(4), sample_skills, potential_steps=0, potential_lane_fraction=0.999999)
    assert bool(np.all(np.asarray(full.potential_lane)))
    np.testing.assert_array_equal(np.asarray(full.skills), 0.0)
    runf = make_rollout_chunk(env, actions, sample_skills, chunk_steps=6, potential_steps=0, potential_lane_fraction=0.999999)
    after, tf, _ = runf({}, full)   # episodes of length 2 -> three resets inside the chunk
    np.testing.assert_array_equal(np.asarray(tf.skills), 0.0)
    np.testing.assert_array_equal(np.asarray(tf.next_skills), 0.0)
    np.testing.assert_allclose(np.asarray(tf.next_anchor_observations), np.asarray(tf.next_observations))
    assert bool(np.all(np.asarray(after.potential_lane)))
    # Skill lanes (fraction 0) with potential_steps 0 carry a live skill from reset onward.
    _, ts, _ = run0({}, base)
    np.testing.assert_array_equal(np.asarray(ts.skills), 1.0)
