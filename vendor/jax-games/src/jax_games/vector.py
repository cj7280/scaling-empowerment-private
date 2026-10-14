"""JAX-native vectorization and rollout helpers."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax

from jax_games.core import JaxGame


class JaxGameVectorFns(NamedTuple):
    """Batched reset and step callables; tuple unpacking remains supported."""

    reset: Callable[..., Any]
    step: Callable[..., Any]


def make_jax_game_vector_fns(
    env: JaxGame,
    params: Any = None,
    *,
    jit: bool = True,
) -> JaxGameVectorFns:
    """Vectorize scalar environment semantics with ``vmap`` and optional ``jit``."""
    resolved_params = env.default_params if params is None else params

    reset_many = jax.vmap(lambda key: env.reset(key, resolved_params))
    step_many = jax.vmap(
        lambda key, state, action: env.step(key, state, action, resolved_params)
    )
    if jit:
        reset_many = jax.jit(reset_many)
        step_many = jax.jit(step_many)
    return JaxGameVectorFns(reset_many, step_many)


def make_jax_game_rollout_fn(
    env: JaxGame,
    params: Any = None,
    *,
    jit: bool = True,
) -> Callable[..., Any]:
    """Build a device-side ``lax.scan`` over ``[time, env]`` discrete actions.

    The caller owns reset policy. This function deliberately returns terminal
    states instead of silently auto-resetting them.
    """
    vector_fns = make_jax_game_vector_fns(env, params=params, jit=False)

    def rollout(key, state, actions):
        def step_scan(carry, action):
            next_key, state_in = carry
            next_key, step_root = jax.random.split(next_key)
            step_keys = jax.random.split(step_root, action.shape[0])
            transition = vector_fns.step(step_keys, state_in, action)
            return (next_key, transition[1]), transition

        return jax.lax.scan(step_scan, (key, state), actions)

    return jax.jit(rollout) if jit else rollout
