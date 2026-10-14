"""SkillLogitAdapter on the PQN Q-head: off = byte-identical params and Q-values (oracle = a
pre-build source snapshot when PQN_AGENT_PRE_SNAPSHOT is set); on = z-dependent Q at init,
zero contribution at z = 0, double-centered kernel."""
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.utils import networks as A

PRE = os.environ.get("PQN_AGENT_PRE_SNAPSHOT")
OBS, ANCH, Z, ACT, B = 12, 12, 4, 17, 8


def _net(mod, std):
    kw = dict(skill_dim=Z, skill_logit_init_std=std) if std is not None else {}
    return mod.RecurrentQNetwork(action_dim=ACT, hidden_size=32, num_layers=1, num_rnn_layers=1,
                                 norm_input=True, add_last_action=True, recurrent_context_dim=0, **kw)


def _inputs(key, z):
    obs = jax.random.normal(key, (1, B, OBS + ANCH))
    return jnp.concatenate([obs, jnp.broadcast_to(z, (1, B, Z))], axis=-1)


def _run(net, params, x):
    carry = net.initialize_carry(B)
    (_, q), _ = net.apply(params, carry, x, jnp.ones((1, B), bool), jnp.zeros((1, B), jnp.int32),
                          train=False, mutable=["batch_stats"])
    return q[0]


def _init(net, x):
    carry = net.initialize_carry(B)
    return net.init(jax.random.PRNGKey(0), carry, x, jnp.ones((1, B), bool), jnp.zeros((1, B), jnp.int32))


def test_adapter_off_is_byte_identical():
    z = jax.random.normal(jax.random.PRNGKey(1), (Z,))
    x = _inputs(jax.random.PRNGKey(2), z)
    net0 = _net(A, 0.0); p0 = _init(net0, x)
    assert "skill_logit_adapter" not in p0["params"]
    if PRE:
        spec = importlib.util.spec_from_file_location("empowerment_warp.algorithms.pqn.agent_pre", PRE)
        pre = importlib.util.module_from_spec(spec); sys.modules[spec.name] = pre; spec.loader.exec_module(pre)
        netp = _net(pre, None); pp = _init(netp, x)
        assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(pp)
        for a, b in zip(jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(pp)):
            assert np.array_equal(np.asarray(a), np.asarray(b))
        assert np.array_equal(np.asarray(_run(net0, p0, x)), np.asarray(_run(netp, pp, x)))


def test_adapter_on_breaks_skill_symmetry_at_init():
    net = _net(A, 0.2)
    z1 = jax.random.normal(jax.random.PRNGKey(1), (Z,)); z1 = z1 / jnp.linalg.norm(z1)
    z2 = -z1
    x1 = _inputs(jax.random.PRNGKey(2), z1); x2 = _inputs(jax.random.PRNGKey(2), z2)
    x0 = _inputs(jax.random.PRNGKey(2), jnp.zeros((Z,)))
    p = _init(net, x1)
    k = np.asarray(p["params"]["skill_logit_adapter"]["kernel"]); assert k.shape == (Z, ACT)
    q1, q2, q0 = (np.asarray(_run(net, p, x)) for x in (x1, x2, x0))
    # the adapter's contribution is the antisymmetric part under z -> -z (rest of the net sees z too)
    c = k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()
    assert np.allclose(c.sum(0), 0, atol=1e-6) and np.allclose(c.sum(1), 0, atol=1e-6)
    bias1 = np.asarray(z1) @ c
    assert np.abs(bias1).max() > 0.05, "random init must give a non-trivial skill-dependent preference"
    # z = 0: adapter adds exactly 0, so Q equals the no-adapter net with identical other params
    net0 = _net(A, 0.0); p0 = _init(net0, x0)
    for name in p0["params"]:
        for a, b in zip(jax.tree_util.tree_leaves(p0["params"][name]), jax.tree_util.tree_leaves(p["params"][name])):
            assert np.array_equal(np.asarray(a), np.asarray(b)), name
    assert np.allclose(q0, np.asarray(_run(net0, p0, x0)), atol=1e-6)
    # and the greedy action depends on z at init
    assert not np.array_equal(q1.argmax(-1), q2.argmax(-1)) or np.abs(q1 - q2).max() > 0.05
