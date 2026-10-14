"""Filtered JAX Games 2.0 registry; Key Door is intentionally absent."""

from __future__ import annotations

from dataclasses import dataclass

from jax_games import make_jax_game_env_from_name
from jax_games.core import JaxGame


@dataclass(frozen=True, slots=True)
class GameSpec:
    public_id: str
    observation_kind: str
    description: str


GAME_SPECS = (
    GameSpec("craftax:symbolic", "vector", "Full Craftax symbolic state."),
    GameSpec("craftax:pixels", "pixels", "Full Craftax pixels."),
    GameSpec("craftax:classic-symbolic", "vector", "Craftax Classic symbolic state."),
    GameSpec("craftax:classic-pixels", "pixels", "Craftax Classic pixels."),
)
_GAME_SPECS_BY_ID = {spec.public_id: spec for spec in GAME_SPECS}


def available_games() -> tuple[str, ...]:
    return tuple(spec.public_id for spec in GAME_SPECS)


def game_spec(name: str) -> GameSpec:
    try:
        return _GAME_SPECS_BY_ID[name]
    except KeyError as exc:
        available = ", ".join(available_games())
        raise ValueError(f"unsupported game {name!r}; available games: {available}") from exc


def make_game(name: str) -> JaxGame:
    spec = game_spec(name)
    return make_jax_game_env_from_name(spec.public_id, auto_reset=False)
