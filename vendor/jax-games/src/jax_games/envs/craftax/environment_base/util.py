import pickle
import bz2
from typing import Any
import jax
import jax.numpy as jnp


def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title, "w") as f:
        # NumPy payloads do not embed JAX-version-specific array metadata.
        pickle.dump(jax.device_get(data), f)


def load_compressed_pickle(file: str):
    with bz2.BZ2File(file, "rb") as stream:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(stream))
