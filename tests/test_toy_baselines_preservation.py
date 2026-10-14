"""PRNG layout checks for the toy LatentGCE and VIM baselines."""

import sys
from types import SimpleNamespace

import jax

from empowerment_warp.baselines import toy_baselines


def test_public_run_pins_saved_prng_layout(monkeypatch):
    previous = jax.config.jax_threefry_partitionable
    monkeypatch.setitem(
        sys.modules,
        'empowerment_warp.baselines.toy_baselines.runner',
        SimpleNamespace(run_experiment=lambda config, max_updates: (
            jax.config.jax_threefry_partitionable, max_updates)),
    )
    try:
        jax.config.update('jax_threefry_partitionable', True)
        assert toy_baselines.run(None, max_updates=2) == (False, 2)
        assert jax.config.jax_threefry_partitionable is True
    finally:
        jax.config.update('jax_threefry_partitionable', previous)
