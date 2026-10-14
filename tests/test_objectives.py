from __future__ import annotations

import jax.numpy as jnp

from empowerment_warp.utils.objectives import (
    forward_infonce,
)






def test_forward_infonce_prefers_aligned_pairs() -> None:
    skills = jnp.eye(4)
    aligned = forward_infonce(skills, skills, temperature=0.1)
    collapsed = forward_infonce(jnp.ones_like(skills), skills, temperature=0.1)
    assert float(aligned.loss) < float(collapsed.loss)
    assert float(aligned.accuracy) == 1.0
