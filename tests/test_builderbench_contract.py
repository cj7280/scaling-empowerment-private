from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.envs.builderbench import BuilderBenchConfig, BuilderBenchWarpEnv
from empowerment_warp.envs.builderbench.model import build_model


def test_release_config_has_ten_physics_steps_per_control() -> None:
    config = BuilderBenchConfig(num_envs=8)
    assert config.sim_dt == 0.001
    assert config.ctrl_dt == 0.01
    assert config.substeps == 10
    assert config.naconmax == 8 * config.naconmax_per_env
    assert config.njmax_per_env == 128


def test_release_config_rejects_unknown_graph_mode() -> None:
    with pytest.raises(ValueError, match="unsupported Warp graph mode"):
        BuilderBenchConfig(graph_mode="typo")


def test_corner3_model_contract() -> None:
    bundle = build_model(BuilderBenchConfig(num_envs=2))
    assert (bundle.model.nq, bundle.model.nv, bundle.model.nu, bundle.model.nsensordata) == (33, 30, 5, 10)
    np.testing.assert_array_equal(bundle.object_qpos_adr, np.asarray([12, 19, 26]))
    np.testing.assert_array_equal(bundle.object_qvel_adr, np.asarray([12, 18, 24]))
    assert bundle.target_mask.tolist() == [True, True, True]


def test_warp_reset_step_and_explicit_reset_selection() -> None:
    config = BuilderBenchConfig(num_envs=2, naconmax_per_env=128, njmax_per_env=256)
    env = BuilderBenchWarpEnv(config)
    keys = jax.random.split(jax.random.PRNGKey(0), config.num_envs)
    state = env.reset(keys)
    jax.block_until_ready(state.obs)

    assert state.obs.shape == (2, 51)
    assert state.data.qpos.shape == (2, 33)
    np.testing.assert_allclose(np.asarray(state.obs[:, 10:19]).reshape(2, 3, 3)[0, :, :2], [[0.4, 0.24], [0.345, 0.24], [0.4, 0.185]], atol=1e-6)

    next_state = env.step(state, jnp.zeros((2, 5), dtype=jnp.float32))
    jax.block_until_ready(next_state.obs)
    assert np.isfinite(np.asarray(next_state.obs)).all()
    assert not np.asarray(next_state.terminated).any()
    assert not np.asarray(next_state.truncated).any()

    reset_state = env.reset(jax.random.split(jax.random.PRNGKey(1), config.num_envs))
    forced_done = next_state._replace(truncated=jnp.asarray([True, False]))
    selected = env.where_done(forced_done, reset_state)
    np.testing.assert_allclose(np.asarray(selected.data.qpos[0]), np.asarray(reset_state.data.qpos[0]))
    np.testing.assert_allclose(np.asarray(selected.data.qpos[1]), np.asarray(next_state.data.qpos[1]))
