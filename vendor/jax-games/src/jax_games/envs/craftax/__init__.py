"""Craftax environments bundled with JAX Games."""

from jax_games.registry import (
    make_craftax_env_from_name,
    make_craftax_env_from_params,
)

__all__ = [
    "make_craftax_env_from_name",
    "make_craftax_env_from_params",
]
