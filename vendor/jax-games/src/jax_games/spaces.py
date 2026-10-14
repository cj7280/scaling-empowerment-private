"""Small immutable spaces with explicit-key JAX sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Box:
    low: Any
    high: Any
    shape: tuple[int, ...]
    dtype: Any = jnp.float32

    def sample(self, key: jax.Array):
        return jax.random.uniform(
            key,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class Discrete:
    n: int
    dtype: Any = jnp.int32
    shape: tuple[int, ...] = ()

    def sample(self, key: jax.Array):
        return jax.random.randint(
            key,
            shape=self.shape,
            minval=0,
            maxval=int(self.n),
            dtype=self.dtype,
        )
