"""Pure functional interface shared by all JAX Games environments."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Protocol, Union, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class JaxGame(Protocol):
    """Structural contract for a scalar, gymnax-style pure-JAX game."""

    @property
    def default_params(self) -> Any:
        ...

    @property
    def num_actions(self) -> int:
        ...

    def reset(self, key: jax.Array, params: Any = None) -> tuple[Any, Any]:
        ...

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: Any,
        params: Any = None,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        ...

    def action_space(self, params: Any = None) -> Any:
        ...

    def observation_space(self, params: Any = None) -> Any:
        ...


class EnvironmentNoAutoReset:
    """Gymnax-style base whose scalar reset and step remain pure JAX."""

    @property
    def default_params(self):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state,
        action: Union[int, float],
        params=None,
    ):
        """Apply one transition without hidden host state or auto-reset."""
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params=None):
        """Sample one initial state from an explicit PRNG key."""
        if params is None:
            params = self.default_params
        return self.reset_env(key, params)

    def step_env(
        self,
        key: jax.Array,
        state,
        action: Union[int, float],
        params,
    ):
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: jax.Array, params):
        """Environment-specific reset."""
        raise NotImplementedError

    def get_obs(self, state) -> jax.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state, params) -> bool:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params):
        """State space of the environment."""
        raise NotImplementedError


class EnvironmentAutoReset(EnvironmentNoAutoReset):
    """Pure-JAX base that resets terminal states with a separate PRNG key."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state,
        action: Union[int, float],
        params=None,
    ):
        if params is None:
            params = self.default_params
        step_key, reset_key = jax.random.split(key)
        obs, next_state, reward, done, info = self.step_env(
            step_key,
            state,
            action,
            params,
        )
        reset_obs, reset_state = self.reset_env(reset_key, params)

        def select_terminal(reset_value, next_value):
            if not hasattr(next_value, "shape"):
                return next_value
            mask = done
            while getattr(mask, "ndim", 0) < next_value.ndim:
                mask = mask[..., None]
            return jnp.where(mask, reset_value, next_value)

        final_obs = select_terminal(reset_obs, obs)
        final_state = jax.tree_util.tree_map(
            select_terminal,
            reset_state,
            next_state,
        )
        return final_obs, final_state, reward, done, info
