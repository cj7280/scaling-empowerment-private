"""Native MuJoCo parity probes for the Warp locomotion environments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from empowerment_warp.envs.locomotion.env import AntWarpEnv, State


def native_rollout(
    env: AntWarpEnv,
    initial_state: State,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(env.native_model)
    data.qpos[:] = np.asarray(initial_state.data.qpos[0])
    data.qvel[:] = np.asarray(initial_state.data.qvel[0])
    data.ctrl[:] = np.asarray(initial_state.data.ctrl[0])
    mujoco.mj_forward(env.native_model, data)

    qpos = []
    qvel = []
    for action in actions:
        data.ctrl[:] = np.clip(
            action,
            env.native_model.actuator_ctrlrange[:, 0],
            env.native_model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(env.native_model, data, nstep=env.config.substeps)
        qpos.append(np.asarray(data.qpos).copy())
        qvel.append(np.asarray(data.qvel).copy())
    return np.asarray(qpos), np.asarray(qvel)


def warp_rollout(
    env: AntWarpEnv,
    initial_state: State,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def one_step(state: State, action: jax.Array) -> tuple[State, tuple[jax.Array, jax.Array]]:
        next_state = env._step_batched(state, action)
        return next_state, (next_state.data.qpos, next_state.data.qvel)

    _, (qpos, qvel) = jax.jit(lambda state, values: jax.lax.scan(one_step, state, values))(
        initial_state,
        actions,
    )
    return qpos[:, 0], qvel[:, 0]


def parity_probe(env: AntWarpEnv, *, steps: int = 10, seed: int = 0) -> dict[str, Any]:
    """Compare one second of identical random control in native MuJoCo and Warp."""

    if env.config.num_envs != 1:
        raise ValueError("locomotion parity_probe requires num_envs=1")
    state = env.reset(jax.random.split(jax.random.PRNGKey(seed), 1))
    actions = jax.random.uniform(
        jax.random.PRNGKey(seed + 1),
        (steps, 1, env.action_size),
        minval=-1.0,
        maxval=1.0,
    )
    warp_qpos, warp_qvel = jax.device_get(warp_rollout(env, state, actions))
    native_qpos, native_qvel = native_rollout(env, state, np.asarray(actions[:, 0]))

    qpos_error = warp_qpos - native_qpos
    qvel_error = warp_qvel - native_qvel
    max_qpos_error = float(np.max(np.abs(qpos_error)))
    max_qvel_error = float(np.max(np.abs(qvel_error)))
    max_root_position_error = float(np.max(np.abs(qpos_error[:, :3])))
    warp_finite = bool(np.isfinite(warp_qpos).all() and np.isfinite(warp_qvel).all())
    native_finite = bool(np.isfinite(native_qpos).all() and np.isfinite(native_qvel).all())
    thresholds = {
        "max_abs_qpos_error": 0.1,
        "max_abs_qvel_error": 1.0,
        "max_root_position_error": 0.02,
        "max_ball_position_error": 0.02,
        "max_ball_floor_penetration": 0.02,
    }
    checks = {
        "finite": warp_finite and native_finite,
        "qpos_parity": max_qpos_error <= thresholds["max_abs_qpos_error"],
        "qvel_parity": max_qvel_error <= thresholds["max_abs_qvel_error"],
        "root_position_parity": max_root_position_error <= thresholds["max_root_position_error"],
    }
    result: dict[str, Any] = {
        "environment": type(env).__name__,
        "config": asdict(env.config),
        "device": str(jax.devices()[0]),
        "steps": steps,
        "seed": seed,
        "thresholds": thresholds,
        "max_abs_qpos_error": max_qpos_error,
        "rms_qpos_error": float(np.sqrt(np.mean(qpos_error**2))),
        "max_abs_qvel_error": max_qvel_error,
        "rms_qvel_error": float(np.sqrt(np.mean(qvel_error**2))),
        "max_root_position_error": max_root_position_error,
        "warp_finite": warp_finite,
        "native_finite": native_finite,
    }
    if env.soccer:
        warp_ball = warp_qpos[:, -7:-4]
        native_ball = native_qpos[:, -7:-4]
        max_ball_position_error = float(np.max(np.abs(warp_ball - native_ball)))
        warp_penetration = max(0.0, 0.25 - float(np.min(warp_ball[:, 2])))
        native_penetration = max(0.0, 0.25 - float(np.min(native_ball[:, 2])))
        checks.update(
            {
                "ball_position_parity": max_ball_position_error
                <= thresholds["max_ball_position_error"],
                "warp_ball_floor_penetration": warp_penetration
                <= thresholds["max_ball_floor_penetration"],
                "native_ball_floor_penetration": native_penetration
                <= thresholds["max_ball_floor_penetration"],
            }
        )
        result.update(
            {
                "max_ball_position_error": max_ball_position_error,
                "warp_max_ball_floor_penetration": warp_penetration,
                "native_max_ball_floor_penetration": native_penetration,
            }
        )
    result["checks"] = checks
    result["status"] = "passed" if all(checks.values()) else "failed"
    return result


def stability_probe(env: AntWarpEnv, *, steps: int = 100, seed: int = 0) -> dict[str, Any]:
    """Stress random contacts and require a bounded, finite state envelope."""

    state = env.reset(jax.random.split(jax.random.PRNGKey(seed), env.config.num_envs))
    actions = jax.random.uniform(
        jax.random.PRNGKey(seed + 1),
        (steps, env.config.num_envs, env.action_size),
        minval=-1.0,
        maxval=1.0,
    )

    def one_step(
        current: State,
        action: jax.Array,
    ) -> tuple[State, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        current = env.step(current, action)
        diagnostics = current.data._impl
        return current, (
            current.data.qpos,
            current.data.qvel,
            diagnostics.nacon,
            diagnostics.nefc,
            diagnostics.overflow,
        )

    _, diagnostics = jax.jit(lambda initial, values: jax.lax.scan(one_step, initial, values))(
        state, actions
    )
    qpos, qvel, nacon, nefc, overflow = jax.device_get(diagnostics)
    max_nacon = int(np.max(nacon))
    max_nefc = int(np.max(nefc))
    overflow_any = bool(np.any(overflow))
    naconmax = int(env._data_template._impl.naconmax)
    njmax = int(env._data_template._impl.njmax)
    checks = {
        "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
        "bounded_root_xy": float(np.max(np.abs(qpos[..., :2]))) <= 100.0,
        "bounded_root_height": -2.0 <= float(np.min(qpos[..., 2])) and float(np.max(qpos[..., 2])) <= 5.0,
        "bounded_velocity": float(np.max(np.abs(qvel))) <= 200.0,
        "no_contact_overflow": not overflow_any,
    }
    result = {
        "environment": type(env).__name__,
        "config": asdict(env.config),
        "device": str(jax.devices()[0]),
        "steps": steps,
        "seed": seed,
        "max_abs_root_xy": float(np.max(np.abs(qpos[..., :2]))),
        "min_root_height": float(np.min(qpos[..., 2])),
        "max_root_height": float(np.max(qpos[..., 2])),
        "max_abs_velocity": float(np.max(np.abs(qvel))),
        "naconmax_total": naconmax,
        "njmax_per_world": njmax,
        "max_nacon_total_observed": max_nacon,
        "max_nefc_per_world_observed": max_nefc,
        "nacon_utilization": max_nacon / naconmax,
        "njmax_utilization": max_nefc / njmax,
        "overflow_any_observed": overflow_any,
        "checks": checks,
    }
    result["status"] = "passed" if all(checks.values()) else "failed"
    return result


def soccer_ball_wall_probe(
    env: AntWarpEnv,
    *,
    steps: int = 10,
    seed: int = 0,
    launch_speed: float = 10.0,
) -> dict[str, Any]:
    """Launch the soccer ball into an arena wall in native MuJoCo and Warp."""

    if not env.soccer or env.config.num_envs != 1:
        raise ValueError("soccer_ball_wall_probe requires a one-lane AntSoccer environment")
    state = env.reset(jax.random.split(jax.random.PRNGKey(seed), 1))
    qpos = state.data.qpos.at[0, :2].set(jnp.asarray([10.0, 5.0]))
    qpos = qpos.at[0, -7:-4].set(jnp.asarray([20.5, 10.0, 0.25]))
    qpos = qpos.at[0, -4:].set(jnp.asarray([1.0, 0.0, 0.0, 0.0]))
    qvel = state.data.qvel.at[0].set(jnp.zeros_like(state.data.qvel[0]))
    if launch_speed <= 0.0:
        raise ValueError("launch_speed must be positive")
    qvel = qvel.at[0, -6].set(launch_speed)
    data = state.data.replace(qpos=qpos, qvel=qvel)
    data = jax.vmap(lambda lane: mjx.forward(env.model, lane))(data)
    state = state._replace(data=data)
    actions = jnp.zeros((steps, 1, env.action_size), dtype=jnp.float32)
    warp_qpos, warp_qvel = jax.device_get(warp_rollout(env, state, actions))
    native_qpos, native_qvel = native_rollout(env, state, np.asarray(actions[:, 0]))
    warp_ball = warp_qpos[:, -7:-4]
    native_ball = native_qpos[:, -7:-4]
    warp_velocity = warp_qvel[:, -6:-3]
    native_velocity = native_qvel[:, -6:-3]
    inner_wall_x = 22.0
    ball_radius = 0.25
    center_limit = inner_wall_x - ball_radius
    thresholds = {
        "max_ball_position_error": 0.1,
        "max_ball_velocity_error": 1.0,
        "max_wall_penetration": 0.02,
        "max_floor_penetration": 0.02,
    }
    position_error = float(np.max(np.abs(warp_ball - native_ball)))
    velocity_error = float(np.max(np.abs(warp_velocity - native_velocity)))
    warp_wall_penetration = max(0.0, float(np.max(warp_ball[:, 0])) - center_limit)
    native_wall_penetration = max(0.0, float(np.max(native_ball[:, 0])) - center_limit)
    warp_floor_penetration = max(0.0, ball_radius - float(np.min(warp_ball[:, 2])))
    native_floor_penetration = max(0.0, ball_radius - float(np.min(native_ball[:, 2])))
    checks = {
        "finite": bool(
            np.isfinite(warp_ball).all()
            and np.isfinite(native_ball).all()
            and np.isfinite(warp_velocity).all()
            and np.isfinite(native_velocity).all()
        ),
        "ball_position_parity": position_error <= thresholds["max_ball_position_error"],
        "ball_velocity_parity": velocity_error <= thresholds["max_ball_velocity_error"],
        "warp_wall_containment": warp_wall_penetration <= thresholds["max_wall_penetration"],
        "native_wall_containment": native_wall_penetration <= thresholds["max_wall_penetration"],
        "warp_floor_containment": warp_floor_penetration <= thresholds["max_floor_penetration"],
        "native_floor_containment": native_floor_penetration <= thresholds["max_floor_penetration"],
        "warp_collision_response": float(np.min(warp_velocity[:, 0])) < 0.5 * launch_speed,
        "native_collision_response": float(np.min(native_velocity[:, 0])) < 0.5 * launch_speed,
    }
    result = {
        "environment": type(env).__name__,
        "config": asdict(env.config),
        "device": str(jax.devices()[0]),
        "steps": steps,
        "seed": seed,
        "launch_speed_m_per_s": launch_speed,
        "thresholds": thresholds,
        "max_ball_position_error": position_error,
        "max_ball_velocity_error": velocity_error,
        "warp_max_wall_penetration": warp_wall_penetration,
        "native_max_wall_penetration": native_wall_penetration,
        "warp_max_floor_penetration": warp_floor_penetration,
        "native_max_floor_penetration": native_floor_penetration,
        "warp_min_postlaunch_x_velocity": float(np.min(warp_velocity[:, 0])),
        "native_min_postlaunch_x_velocity": float(np.min(native_velocity[:, 0])),
        "checks": checks,
    }
    result["status"] = "passed" if all(checks.values()) else "failed"
    return result
