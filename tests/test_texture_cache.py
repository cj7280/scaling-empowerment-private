"""Texture caches remain portable across JAX array serialization versions."""
import bz2
import pickle
import jax.numpy as jnp
import numpy as np
from jax_games.envs.craftax.environment_base.util import load_compressed_pickle, save_compressed_pickle


def test_texture_cache_stores_numpy_and_restores_exact_values(tmp_path):
    textures = {7: {'pixels': jnp.arange(24, dtype=jnp.uint8).reshape(2,4,3),
                    'alpha': jnp.array([0.,.5,1.], dtype=jnp.float32)}}
    path = tmp_path / 'textures.pbz2'
    save_compressed_pickle(path, textures)
    with bz2.BZ2File(path,'rb') as stream:
        saved = pickle.load(stream)
    restored = load_compressed_pickle(path)
    for name, value in textures[7].items():
        assert isinstance(saved[7][name], np.ndarray)
        np.testing.assert_array_equal(restored[7][name], value)
        assert restored[7][name].dtype == value.dtype
