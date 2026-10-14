"""Gram spectra must preserve the reference water-filled PPO reward."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.latent_gce import channel_singular_values, water_filling_from_singular_values


@pytest.mark.parametrize('shape', [(42,240), (2,240), (9,150), (42,800), (42,8)])
def test_full_rank_and_degenerate_rewards(shape):
    rng = np.random.default_rng(41)
    m, n = shape
    dense = rng.normal(size=(m,n)).astype(np.float32)
    rank_one = np.outer(rng.normal(size=m), rng.normal(size=n)).astype(np.float32)
    near_rank_one = rank_one + 1e-5*dense
    matrices = np.stack([dense, rank_one, near_rank_one, np.zeros_like(dense),
                         dense*1e-3, dense*1e3, rank_one*1e3, near_rank_one*1e3])
    reference = jnp.linalg.svd(jnp.asarray(matrices), compute_uv=False)
    precise_reference = jnp.asarray(np.linalg.svd(matrices.astype(np.float64), compute_uv=False),
                                    dtype=jnp.float32)
    x64_before = jax.config.x64_enabled
    actual = jax.jit(channel_singular_values)(jnp.asarray(matrices))
    mapped = jax.jit(lambda x:jax.lax.map(channel_singular_values,x))(jnp.asarray(matrices))
    assert jax.config.x64_enabled == x64_before
    assert actual.dtype == jnp.float32
    assert actual.shape == (8,min(m,n))
    assert np.isfinite(actual).all() and (np.asarray(actual)>=0).all()
    assert (np.diff(np.asarray(actual), axis=-1)<=0).all()
    # Null-space rounding is unavoidable; capacity and reward must still agree.
    np.testing.assert_allclose(actual[0], reference[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual[3], np.zeros(min(m,n)))
    for power in (0., 1., 10.):
        expected = water_filling_from_singular_values(precise_reference, power=power)[0]
        capacity = water_filling_from_singular_values(actual, power=power)[0]
        mapped_capacity = water_filling_from_singular_values(mapped, power=power)[0]
        np.testing.assert_allclose(mapped_capacity, expected, rtol=1e-5, atol=5e-6)
        np.testing.assert_allclose(capacity, expected, rtol=1e-5, atol=5e-6)
        # exp(capacity) is the existing PPO transform, even though capacity is bits.
        np.testing.assert_allclose(np.exp(np.asarray(capacity, dtype=np.float64)),
                                   np.exp(np.asarray(expected, dtype=np.float64)), rtol=5e-5)
        if power <= 1.:
            # Preserve the actual power=1 recipe. At power=10 in ill-conditioned
            # synthetic cases, the old float32 SVD itself misses the float64 oracle.
            old_capacity = water_filling_from_singular_values(reference, power=power)[0]
            np.testing.assert_allclose(capacity, old_capacity, rtol=1e-5, atol=5e-6)
