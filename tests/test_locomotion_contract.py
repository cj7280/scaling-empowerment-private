from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.training.sac.config import default_recipe
from empowerment_warp.envs.factory import make_environment
from empowerment_warp.envs.locomotion import AntConfig, AntSoccerConfig, AntSoccerWarpEnv, AntWarpEnv
from empowerment_warp.envs.locomotion.model import build_ant_model


def test_ant_and_soccer_model_contracts() -> None:
    ant = build_ant_model(AntConfig(num_envs=2), soccer=False)
    soccer = build_ant_model(AntSoccerConfig(num_envs=2), soccer=True)
    assert (ant.nq, ant.nv, ant.nu, ant.ngeom) == (15, 14, 8, 14)
    assert (soccer.nq, soccer.nv, soccer.nu, soccer.ngeom) == (22, 20, 8, 44)


def test_ant_and_soccer_warp_reset_step_contracts() -> None:
    cases = (
        (AntConfig(num_envs=2), AntWarpEnv, 29, 0.0),
        (AntSoccerConfig(num_envs=2), AntSoccerWarpEnv, 42, -1.0),
    )
    for config, cls, obs_size, expected_reward in cases:
        env = cls(config)
        state = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
        state = env.step(state, jnp.zeros((2, 8), dtype=jnp.float32))
        jax.block_until_ready(state.obs)
        assert state.obs.shape == (2, obs_size)
        assert np.isfinite(np.asarray(state.obs)).all()
        np.testing.assert_allclose(np.asarray(state.reward), expected_reward)
        replacement = env.reset(jax.random.split(jax.random.PRNGKey(1), 2))
        selected = env.where(jnp.asarray([True, False]), state, replacement)
        jax.block_until_ready(selected.obs)
        np.testing.assert_allclose(selected.obs[0], replacement.obs[0])
        np.testing.assert_allclose(selected.obs[1], state.obs[1])


def test_factory_plumbs_episode_length_preset_and_encoder_view() -> None:
    ant_config = default_recipe(
        "ant",
        "potential",
        num_envs=2,
        episode_length=17,
        allow_reference_deviation=True,
        allow_unpromoted_physics=True,
    )
    ant = make_environment(ant_config)
    assert isinstance(ant, AntSoccerWarpEnv)
    assert ant.config.episode_length == 17
    assert ant.config.spawn_mode == "standard"
    assert ant.observation_size == 42
    assert ant.encoder_observation_indices("full") is None

    soccer_config = default_recipe(
        "ant_soccer",
        "potential",
        num_envs=2,
        episode_length=23,
        antsoccer_preset="closer_ball",
        encoder_obs_mode="ball_xy",
        allow_reference_deviation=True,
        allow_unpromoted_physics=True,
    )
    soccer = make_environment(soccer_config)
    assert soccer.config.episode_length == 23
    assert soccer.config.spawn_mode == "closer_ball"
    assert soccer.encoder_observation_indices("ball_xy") == (15, 16)


def test_soccer_default_task_and_spawn_presets() -> None:
    standard = AntSoccerWarpEnv(AntSoccerConfig(num_envs=2))
    state = standard.reset(jax.random.split(jax.random.PRNGKey(2), 2))
    np.testing.assert_allclose(np.asarray(state.goal_xy), [[0.0, 20.0], [0.0, 20.0]])
    np.testing.assert_allclose(standard.native_model.geom("target").pos[:2], [0.0, 20.0])
    assert np.all(np.abs(np.asarray(state.data.qpos[:, :2]) - np.asarray([20.0, 20.0])) <= 1.0)
    assert np.all(np.abs(np.asarray(state.data.qpos[:, -7:-5])) <= 1.0)

    closer = AntSoccerWarpEnv(AntSoccerConfig(num_envs=2, spawn_mode="closer_ball"))
    close_state = closer.reset(jax.random.split(jax.random.PRNGKey(2), 2))
    np.testing.assert_allclose(np.asarray(close_state.data.qpos[:, -7:-5]), 10.0)
    assert np.all(np.abs(np.asarray(close_state.data.qpos[:, :2]) - 12.0) <= 0.2)

    centered = AntSoccerWarpEnv(AntSoccerConfig(num_envs=2, spawn_mode="center_ball"))
    centered_state = centered.reset(jax.random.split(jax.random.PRNGKey(2), 2))
    np.testing.assert_allclose(np.asarray(centered_state.data.qpos[:, :2]), 10.0)
    np.testing.assert_allclose(np.asarray(centered_state.data.qpos[:, -7:-5]), 10.0)


def test_soccer_single_task_reward_uses_pre_action_success() -> None:
    env = AntSoccerWarpEnv(
        AntSoccerConfig(num_envs=1, spawn_noise_scale=0.0, success_timing="pre")
    )
    state = env.reset(jax.random.split(jax.random.PRNGKey(3), 1))
    qpos = state.data.qpos.at[0, -7:-5].set(state.goal_xy[0])
    qvel = state.data.qvel.at[0, -6].set(10.0)
    state = state._replace(data=state.data.replace(qpos=qpos, qvel=qvel))
    state = env.step(state, jnp.zeros((1, 8), dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(state.reward), 0.0)
    np.testing.assert_allclose(np.asarray(state.metrics.success), 1.0)
    assert float(state.metrics.ball_goal_distance[0]) > env.config.goal_tolerance
