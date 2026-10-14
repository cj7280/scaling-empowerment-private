"""Pure-JAX game environments and vectorization helpers."""

from jax_games.core import EnvironmentAutoReset, EnvironmentNoAutoReset, JaxGame
from jax_games.registry import (
    JAX_GAME_REGISTRY,
    JaxGameEntry,
    JaxGameRegistry,
    available_jax_game_envs,
    craftax_env_name,
    make_craftax_env_from_name,
    make_craftax_env_from_params,
    make_jax_game_env_from_name,
    make_jax_game_env_from_params,
    normalize_jax_game_id,
)
from jax_games.vector import (
    JaxGameVectorFns,
    make_jax_game_rollout_fn,
    make_jax_game_vector_fns,
)

__all__ = [
    "EnvironmentNoAutoReset",
    "EnvironmentAutoReset",
    "JAX_GAME_REGISTRY",
    "JaxGame",
    "JaxGameEntry",
    "JaxGameRegistry",
    "JaxGameVectorFns",
    "available_jax_game_envs",
    "craftax_env_name",
    "make_craftax_env_from_name",
    "make_craftax_env_from_params",
    "make_jax_game_env_from_name",
    "make_jax_game_env_from_params",
    "make_jax_game_rollout_fn",
    "make_jax_game_vector_fns",
    "normalize_jax_game_id",
]
