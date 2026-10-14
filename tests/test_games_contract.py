from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.envs.games import JaxGameBatchEnv, available_games, make_game


def test_filtered_registry_excludes_key_door() -> None:
    assert available_games() == (
        "craftax:symbolic",
        "craftax:pixels",
        "craftax:classic-symbolic",
        "craftax:classic-pixels",
    )
    with pytest.raises(ValueError, match="unsupported game"):
        make_game("key_door:symbolic")


def test_craftax_classic_batched_reset_and_step() -> None:
    env = JaxGameBatchEnv("craftax:classic-symbolic", num_envs=4)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    state = env.step(state, np.zeros((4,), dtype=np.int32))
    jax.block_until_ready(state.obs)
    assert state.obs.shape[0] == 4
    assert np.isfinite(np.asarray(state.obs)).all()

    actions = jnp.zeros((3, 4), dtype=jnp.int32)
    state, _ = jax.jit(
        lambda initial, values: jax.lax.scan(
            lambda carry, action: (env._step_batched(carry, action), None),
            initial,
            values,
        )
    )(state, actions)
    jax.block_until_ready(state.obs)
    assert np.isfinite(np.asarray(state.obs)).all()
