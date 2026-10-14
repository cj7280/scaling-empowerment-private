"""Command-line entry points for JAX Games."""

from __future__ import annotations

import argparse

from jax_games.registry import make_jax_game_env_from_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_name", nargs="?", default="craftax:symbolic")
    parser.add_argument("--auto-reset", action="store_true")
    args = parser.parse_args()
    env = make_jax_game_env_from_name(
        args.env_name,
        auto_reset=bool(args.auto_reset),
    )
    print(f"{type(env).__name__}: num_actions={int(env.num_actions)}")
