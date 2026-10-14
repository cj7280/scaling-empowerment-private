"""Explicit public registry for JAX Games environments."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from jax_games.core import JaxGame
EnvBuilder = Callable[[bool], JaxGame]

def normalize_jax_game_id(name: str) -> str:
    env_name = str(name)
    for prefix in ('jax_games:', 'jax-games:'):
        if env_name.startswith(prefix):
            env_name = env_name[len(prefix):]
    return env_name.replace('_', '-').lower()

@dataclass(frozen=True)
class JaxGameEntry:
    public_id: str
    aliases: tuple[str, ...]
    build: EnvBuilder
    description: str

class JaxGameRegistry:
    """Small deterministic registry with lazy environment imports."""

    def __init__(self, entries: tuple[JaxGameEntry, ...]):
        self._entries = entries
        self._by_alias: dict[str, JaxGameEntry] = {}
        for entry in entries:
            for alias in (entry.public_id, *entry.aliases):
                key = normalize_jax_game_id(alias)
                existing = self._by_alias.get(key)
                if existing is not None and existing is not entry:
                    raise ValueError(f'Duplicate JAX game alias {alias!r}.')
                self._by_alias[key] = entry

    def available(self) -> tuple[str, ...]:
        return tuple((entry.public_id for entry in self._entries))

    def resolve(self, name: str) -> JaxGameEntry:
        try:
            return self._by_alias[normalize_jax_game_id(name)]
        except KeyError as exc:
            available = ', '.join(self.available())
            raise ValueError(f'Unsupported JAX game environment {name!r}. Available ids: {available}.') from exc

    def make(self, name: str, *, auto_reset: bool=False) -> JaxGame:
        return self.resolve(name).build(bool(auto_reset))
CRAFTAX_ENV_NAME_BY_ID = {'craftax:symbolic': 'Craftax-Symbolic-v1', 'craftax:pixels': 'Craftax-Pixels-v1', 'craftax:classic-symbolic': 'Craftax-Classic-Symbolic-v1', 'craftax:classic-pixels': 'Craftax-Classic-Pixels-v1', 'craftax-symbolic': 'Craftax-Symbolic-v1', 'craftax-pixels': 'Craftax-Pixels-v1', 'craftax-classic-symbolic': 'Craftax-Classic-Symbolic-v1', 'craftax-classic-pixels': 'Craftax-Classic-Pixels-v1', 'craftax-symbolic-v1': 'Craftax-Symbolic-v1', 'craftax-pixels-v1': 'Craftax-Pixels-v1', 'craftax-classic-symbolic-v1': 'Craftax-Classic-Symbolic-v1', 'craftax-classic-pixels-v1': 'Craftax-Classic-Pixels-v1'}

def craftax_env_name(name: str) -> str:
    key = normalize_jax_game_id(name)
    try:
        return CRAFTAX_ENV_NAME_BY_ID[key]
    except KeyError as exc:
        available = ', '.join(CRAFTAX_ENV_NAME_BY_ID)
        raise ValueError(f'Unsupported Craftax environment {name!r}. Available ids: {available}.') from exc

def make_craftax_env_from_name(name: str, auto_reset: bool=False) -> JaxGame:
    resolved = craftax_env_name(name)
    if resolved == 'Craftax-Symbolic-v1':
        from jax_games.envs.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv, CraftaxSymbolicEnvNoAutoReset
        return CraftaxSymbolicEnv() if auto_reset else CraftaxSymbolicEnvNoAutoReset()
    if resolved == 'Craftax-Pixels-v1':
        from jax_games.envs.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv, CraftaxPixelsEnvNoAutoReset
        return CraftaxPixelsEnv() if auto_reset else CraftaxPixelsEnvNoAutoReset()
    if resolved == 'Craftax-Classic-Symbolic-v1':
        from jax_games.envs.craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv, CraftaxClassicSymbolicEnvNoAutoReset
        return CraftaxClassicSymbolicEnv() if auto_reset else CraftaxClassicSymbolicEnvNoAutoReset()
    if resolved == 'Craftax-Classic-Pixels-v1':
        from jax_games.envs.craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv, CraftaxClassicPixelsEnvNoAutoReset
        return CraftaxClassicPixelsEnv() if auto_reset else CraftaxClassicPixelsEnvNoAutoReset()
    raise AssertionError(f'Unhandled Craftax environment mapping: {resolved}')

def make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool=False) -> JaxGame:
    if classic:
        env_id = 'craftax:classic-symbolic' if symbolic else 'craftax:classic-pixels'
    else:
        env_id = 'craftax:symbolic' if symbolic else 'craftax:pixels'
    return make_craftax_env_from_name(env_id, auto_reset=auto_reset)

def _build_craftax(env_id: str) -> EnvBuilder:
    return lambda auto_reset: make_craftax_env_from_name(env_id, auto_reset=auto_reset)
JAX_GAME_REGISTRY = JaxGameRegistry((JaxGameEntry('craftax:symbolic', ('craftax-symbolic', 'Craftax-Symbolic-v1'), _build_craftax('craftax:symbolic'), 'Craftax symbolic observations.'), JaxGameEntry('craftax:pixels', ('craftax-pixels', 'Craftax-Pixels-v1'), _build_craftax('craftax:pixels'), 'Craftax pixel observations.'), JaxGameEntry('craftax:classic-symbolic', ('craftax:classic_symbolic', 'craftax-classic-symbolic', 'Craftax-Classic-Symbolic-v1'), _build_craftax('craftax:classic-symbolic'), 'Craftax Classic symbolic observations.'), JaxGameEntry('craftax:classic-pixels', ('craftax:classic_pixels', 'craftax-classic-pixels', 'Craftax-Classic-Pixels-v1'), _build_craftax('craftax:classic-pixels'), 'Craftax Classic pixel observations.')))

def available_jax_game_envs() -> tuple[str, ...]:
    return JAX_GAME_REGISTRY.available()

def make_jax_game_env_from_name(name: str, auto_reset: bool=False) -> JaxGame:
    return JAX_GAME_REGISTRY.make(name, auto_reset=auto_reset)

def make_jax_game_env_from_params(game: str='craftax', *, classic: bool=False, symbolic: bool=True, auto_reset: bool=False) -> JaxGame:
    normalized = str(game).replace('_', '-').lower()
    if normalized == 'craftax':
        return make_craftax_env_from_params(classic=classic, symbolic=symbolic, auto_reset=auto_reset)
    raise ValueError(f'Unknown JAX game {game!r}. Available games: craftax.')
