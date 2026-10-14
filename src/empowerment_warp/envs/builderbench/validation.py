"""Native MuJoCo reference stepping for BuilderBench parity probes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from empowerment_warp.envs.builderbench.config import BuilderBenchConfig
from empowerment_warp.envs.builderbench.env import BuilderBenchWarpEnv, State


_ACTION_SCALE = np.asarray([255.0, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
_MAX_QPOS_ERROR_M = 1e-3
_MAX_QVEL_ERROR_MPS = 5e-2
_MAX_CUBE_POSITION_ERROR_M = 1e-3
_MAX_CUBE_PENETRATION_M = 1e-3


def _native_wall_bounce(
    env: BuilderBenchWarpEnv,
    data: mujoco.MjData,
) -> None:
    object_pos = data.qpos[env.bundle.object_qpos_adr[:, None] + np.arange(3)]
    object_linvel = data.qvel[env.bundle.object_qvel_adr[:, None] + np.arange(3)]
    xy_low = env.bundle.workspace_bounds[0, :2] + env.cube_half_extent
    xy_high = env.bundle.workspace_bounds[1, :2] - env.cube_half_extent
    hit_mask = (object_pos[:, :2] < xy_low) | (object_pos[:, :2] > xy_high)
    object_pos[:, :2] = np.clip(object_pos[:, :2], xy_low, xy_high)
    object_linvel[:, :2] = np.where(
        hit_mask,
        -env.config.wall_restitution * object_linvel[:, :2],
        object_linvel[:, :2],
    )
    data.qpos[env.bundle.object_qpos_adr[:, None] + np.arange(3)] = object_pos
    data.qvel[env.bundle.object_qvel_adr[:, None] + np.arange(3)] = object_linvel


def native_rollout(
    env: BuilderBenchWarpEnv,
    initial_state: State,
    actions: np.ndarray,
    lane: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Step native MuJoCo from the exact Warp reset state."""

    data = mujoco.MjData(env.bundle.model)
    data.qpos[:] = np.asarray(initial_state.data.qpos[lane])
    data.qvel[:] = np.asarray(initial_state.data.qvel[lane])
    data.ctrl[:] = np.asarray(initial_state.ctrl[lane])
    data.mocap_pos[env.bundle.mocap_ids] = np.asarray(initial_state.target_goal[lane])
    data.mocap_quat[env.bundle.mocap_ids] = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (3, 1))
    mujoco.mj_forward(env.bundle.model, data)

    qpos = []
    qvel = []
    for action in actions:
        data.ctrl[:] = np.clip(
            data.ctrl + action * _ACTION_SCALE,
            env.bundle.ctrl_bounds[0],
            env.bundle.ctrl_bounds[1],
        )
        mujoco.mj_step(env.bundle.model, data, nstep=env.config.substeps)
        _native_wall_bounce(env, data)
        qpos.append(np.asarray(data.qpos).copy())
        qvel.append(np.asarray(data.qvel).copy())
    return np.asarray(qpos), np.asarray(qvel)


def warp_rollout(
    env: BuilderBenchWarpEnv,
    initial_state: State,
    actions: jax.Array,
) -> tuple[State, tuple[jax.Array, jax.Array]]:
    """Scan a fixed action sequence through the compiled Warp transition."""

    def one_step(state: State, action: jax.Array) -> tuple[State, tuple[jax.Array, jax.Array]]:
        next_state = env._step_batched(state, action)
        return next_state, (next_state.data.qpos, next_state.data.qvel)

    return jax.jit(lambda state, act: jax.lax.scan(one_step, state, act))(initial_state, actions)


def _replace_object_state(
    env: BuilderBenchWarpEnv,
    state: State,
    positions: np.ndarray,
    linear_velocities: np.ndarray,
) -> State:
    """Set a deterministic contact scenario and recompute derived Warp state."""

    position_index = env.bundle.object_qpos_adr[:, None] + np.arange(3)
    velocity_index = env.bundle.object_qvel_adr[:, None] + np.arange(3)
    qpos = state.data.qpos.at[0, position_index].set(jnp.asarray(positions))
    qvel = state.data.qvel.at[0, velocity_index].set(jnp.asarray(linear_velocities))
    data = state.data.replace(qpos=qpos, qvel=qvel)
    data = jax.jit(jax.vmap(lambda lane: mjx.forward(env.model, lane)))(data)
    obs = jax.vmap(env._observation)(data)
    return state._replace(
        data=data,
        obs=obs,
        initial_object_pos=jnp.asarray(positions)[None, ...],
    )


def contact_probe(config: BuilderBenchConfig, seed: int = 0) -> dict[str, Any]:
    """Exercise stable floor contact and a cube-cube impact in both engines."""

    if config.num_envs != 1:
        raise ValueError("contact_probe requires num_envs=1")
    env = BuilderBenchWarpEnv(config)
    initial_state = env.reset(jax.random.split(jax.random.PRNGKey(seed), 1))

    scenarios = {
        "resting_floor": {
            "steps": 200,
            "positions": np.asarray(
                [[0.25, -0.15, 0.02], [0.35, 0.0, 0.02], [0.25, 0.15, 0.02]],
                dtype=np.float32,
            ),
            "linear_velocities": np.zeros((3, 3), dtype=np.float32),
            "max_position_error": 1e-3,
            "max_velocity_error": 5e-2,
        },
        "cube_impact": {
            "steps": 40,
            "positions": np.asarray(
                [[0.24, 0.0, 0.02], [0.29, 0.0, 0.02], [0.40, 0.20, 0.02]],
                dtype=np.float32,
            ),
            "linear_velocities": np.asarray(
                [[0.75, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "max_position_error": 5e-3,
            "max_velocity_error": 1e-1,
        },
    }

    results: dict[str, Any] = {}
    for name, scenario in scenarios.items():
        state = _replace_object_state(
            env,
            initial_state,
            scenario["positions"],
            scenario["linear_velocities"],
        )
        steps = int(scenario["steps"])
        actions = jnp.zeros((steps, 1, env.action_size), dtype=jnp.float32)
        _, (warp_qpos, warp_qvel) = warp_rollout(env, state, actions)
        warp_qpos, warp_qvel = jax.device_get((warp_qpos[:, 0], warp_qvel[:, 0]))
        native_qpos, native_qvel = native_rollout(env, state, np.asarray(actions[:, 0]))

        position_index = env.bundle.object_qpos_adr[:, None] + np.arange(3)
        velocity_index = env.bundle.object_qvel_adr[:, None] + np.arange(3)
        warp_position = warp_qpos[:, position_index]
        native_position = native_qpos[:, position_index]
        warp_linear_velocity = warp_qvel[:, velocity_index]
        native_linear_velocity = native_qvel[:, velocity_index]
        max_position_error = float(np.max(np.abs(warp_position - native_position)))
        max_velocity_error = float(np.max(np.abs(warp_linear_velocity - native_linear_velocity)))
        warp_min_z = float(np.min(warp_position[..., 2]))
        native_min_z = float(np.min(native_position[..., 2]))
        checks = {
            "finite": bool(
                np.isfinite(warp_qpos).all()
                and np.isfinite(warp_qvel).all()
                and np.isfinite(native_qpos).all()
                and np.isfinite(native_qvel).all()
            ),
            "position_parity": max_position_error <= float(scenario["max_position_error"]),
            "velocity_parity": max_velocity_error <= float(scenario["max_velocity_error"]),
            "warp_floor_penetration": max(0.0, env.cube_half_extent - warp_min_z)
            <= _MAX_CUBE_PENETRATION_M,
            "native_floor_penetration": max(0.0, env.cube_half_extent - native_min_z)
            <= _MAX_CUBE_PENETRATION_M,
        }
        if name == "resting_floor":
            checks["warp_rest_stability"] = (
                float(np.max(np.abs(warp_position[-1] - scenario["positions"]))) <= 2e-3
            )
            checks["native_rest_stability"] = (
                float(np.max(np.abs(native_position[-1] - scenario["positions"]))) <= 2e-3
            )
        else:
            checks["warp_impact_transfer"] = float(np.max(np.abs(warp_linear_velocity[:, 1, :2]))) >= 0.05
            checks["native_impact_transfer"] = float(np.max(np.abs(native_linear_velocity[:, 1, :2]))) >= 0.05

        results[name] = {
            "status": "passed" if all(checks.values()) else "failed",
            "steps": steps,
            "checks": checks,
            "max_cube_position_error": max_position_error,
            "max_cube_linear_velocity_error": max_velocity_error,
            "warp_max_cube_penetration": max(0.0, env.cube_half_extent - warp_min_z),
            "native_max_cube_penetration": max(0.0, env.cube_half_extent - native_min_z),
        }

    return {
        "status": "passed" if all(value["status"] == "passed" for value in results.values()) else "failed",
        "config": asdict(config),
        "device": str(jax.devices()[0]),
        "seed": seed,
        "scenarios": results,
    }


def parity_probe(config: BuilderBenchConfig, steps: int, seed: int) -> dict[str, Any]:
    """Compare native and Warp trajectories under floor contact and random control."""

    if config.num_envs != 1:
        raise ValueError("parity_probe requires num_envs=1")
    env = BuilderBenchWarpEnv(config)
    keys = jax.random.split(jax.random.PRNGKey(seed), config.num_envs)
    initial_state = env.reset(keys)
    action_key = jax.random.PRNGKey(seed + 1)
    actions = jax.random.uniform(action_key, (steps, 1, env.action_size), minval=-0.2, maxval=0.2)

    _, (warp_qpos, warp_qvel) = warp_rollout(env, initial_state, actions)
    warp_qpos, warp_qvel = jax.device_get((warp_qpos[:, 0], warp_qvel[:, 0]))
    native_qpos, native_qvel = native_rollout(env, initial_state, np.asarray(actions[:, 0]))

    qpos_error = warp_qpos - native_qpos
    qvel_error = warp_qvel - native_qvel
    object_index = env.bundle.object_qpos_adr[:, None] + np.arange(3)
    warp_object_pos = warp_qpos[:, object_index]
    native_object_pos = native_qpos[:, object_index]
    max_abs_qpos_error = float(np.max(np.abs(qpos_error)))
    max_abs_qvel_error = float(np.max(np.abs(qvel_error)))
    max_cube_position_error = float(np.max(np.abs(warp_object_pos - native_object_pos)))
    warp_min_cube_z = float(np.min(warp_object_pos[..., 2]))
    native_min_cube_z = float(np.min(native_object_pos[..., 2]))
    warp_finite = bool(np.isfinite(warp_qpos).all() and np.isfinite(warp_qvel).all())
    native_finite = bool(np.isfinite(native_qpos).all() and np.isfinite(native_qvel).all())
    checks = {
        "finite": warp_finite and native_finite,
        "qpos_parity": max_abs_qpos_error <= _MAX_QPOS_ERROR_M,
        "qvel_parity": max_abs_qvel_error <= _MAX_QVEL_ERROR_MPS,
        "cube_position_parity": max_cube_position_error <= _MAX_CUBE_POSITION_ERROR_M,
        "warp_floor_penetration": max(0.0, env.cube_half_extent - warp_min_cube_z)
        <= _MAX_CUBE_PENETRATION_M,
        "native_floor_penetration": max(0.0, env.cube_half_extent - native_min_cube_z)
        <= _MAX_CUBE_PENETRATION_M,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "config": asdict(config),
        "device": str(jax.devices()[0]),
        "steps": steps,
        "seed": seed,
        "thresholds": {
            "max_abs_qpos_error": _MAX_QPOS_ERROR_M,
            "max_abs_qvel_error": _MAX_QVEL_ERROR_MPS,
            "max_cube_position_error": _MAX_CUBE_POSITION_ERROR_M,
            "max_cube_penetration": _MAX_CUBE_PENETRATION_M,
        },
        "checks": checks,
        "max_abs_qpos_error": max_abs_qpos_error,
        "rms_qpos_error": float(np.sqrt(np.mean(qpos_error**2))),
        "max_abs_qvel_error": max_abs_qvel_error,
        "rms_qvel_error": float(np.sqrt(np.mean(qvel_error**2))),
        "max_cube_position_error": max_cube_position_error,
        "warp_min_cube_z": warp_min_cube_z,
        "native_min_cube_z": native_min_cube_z,
        "warp_max_cube_penetration": max(0.0, env.cube_half_extent - warp_min_cube_z),
        "native_max_cube_penetration": max(0.0, env.cube_half_extent - native_min_cube_z),
        "warp_finite": warp_finite,
        "native_finite": native_finite,
    }
