"""A3 separate-lanes collector mode: potential lanes run z=0 for whole episodes, skill lanes
start their skill at reset; fraction 0 is byte-identical to the two-phase collector."""
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.pqn import collector as C

PRE = os.environ.get("PQN_COLLECTOR_PRE_SNAPSHOT")  # optional byte-identity oracle


class _ToyEnv:
    """Deterministic scalar env with fixed-length episodes and auto-reset via reset_finished_env."""
    def __init__(self, horizon):
        self.horizon = horizon
        self.action_size = 3

    def reset(self, keys):
        n = keys.shape[0]
        return _State(obs=jnp.zeros((n, 4)), step_count=jnp.zeros((n,), jnp.int32),
                      terminated=jnp.zeros((n,), bool), truncated=jnp.zeros((n,), bool),
                      reward=jnp.zeros((n,)), info={})

    def step(self, state, action):
        sc = state.step_count + 1
        obs = state.obs + 1.0
        trunc = sc >= self.horizon
        return _State(obs=obs, step_count=sc, terminated=jnp.zeros_like(trunc), truncated=trunc,
                      reward=jnp.zeros((sc.shape[0],)), info={})


from flax import struct


@struct.dataclass
class _State:
    obs: jax.Array
    step_count: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    reward: jax.Array
    info: dict


class _Agent:
    def initial_recurrent_state(self, n):
        return jnp.zeros((n, 2))

    def sample_actions_recurrent_from_params(self, params, obs, skills, anchors, rec, last_done, key, eps, deterministic=False):
        n = obs.shape[0]
        idx = jax.random.randint(key, (n,), 0, 3)
        return jax.nn.one_hot(idx, 3), idx, rec


def _reset_finished_env(env, state, keys):
    done = state.terminated | state.truncated
    fresh = env.reset(keys)
    return jax.tree_util.tree_map(lambda f, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s), fresh, state)


def _run(mod, fraction, potential_steps, num_envs=64, horizon=6, steps=30, seed=0):
    env, agent = _ToyEnv(horizon), _Agent()
    mod.reset_finished_env = _reset_finished_env
    sampler = lambda key, n: jax.random.normal(key, (n, 4))
    kw = dict(potential_lane_fraction=fraction) if fraction is not None else {}
    state = mod.initialize_potential_pqn_collector(env, agent, jax.random.PRNGKey(seed), num_envs=num_envs,
                                                   skill_dim=4, potential_steps=potential_steps, skill_sampler=sampler, **kw)
    collect = mod.make_potential_pqn_chunk(env, agent, chunk_steps=steps, num_envs=num_envs, skill_dim=4,
                                           potential_steps=potential_steps, skill_sampler=sampler, **kw)
    state, trans, metrics = collect({}, state, 0.1)
    return state, trans, metrics


def test_fraction_zero_is_byte_identical_to_two_phase():
    if not PRE:
        pytest.skip("set PQN_COLLECTOR_PRE_SNAPSHOT to the pre-build collector.py for the oracle")
    spec = importlib.util.spec_from_file_location("collector_pre", PRE)
    pre = importlib.util.module_from_spec(spec)
    sys.modules["collector_pre"] = pre  # flax dataclass resolves annotations via sys.modules
    spec.loader.exec_module(pre)
    for ps in (0, 3):
        _, a, ma = _run(C, 0.0, ps)
        _, b, mb = _run(pre, None, ps)
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (ps, k)
        for k in ma:
            assert np.array_equal(np.asarray(ma[k]), np.asarray(mb[k])), (ps, k)


def test_separate_lanes_semantics():
    num_envs, horizon, steps = 256, 6, 30
    state, trans, metrics = _run(C, 0.5, 0, num_envs=num_envs, horizon=horizon, steps=steps)
    skills = np.asarray(trans["skills"]).reshape(steps, num_envs, 4)
    ts = np.asarray(trans["timesteps"]).reshape(steps, num_envs)
    ep = np.asarray(trans["episode_ids"]).reshape(steps, num_envs)
    on = np.linalg.norm(skills, axis=-1) > 1e-6
    # Within an episode a lane is either all-off (potential lane) or all-on (skill lane, from step 0).
    for lane in range(num_envs):
        for e in np.unique(ep[:, lane]):
            m = ep[:, lane] == e
            assert on[m, lane].all() or (~on[m, lane]).all(), (lane, e)
            if on[m, lane].all():
                assert ts[m, lane].min() == 0 or m.argmax() == 0  # skill active from the episode's first step
    frac = (~on).mean()
    assert 0.35 < frac < 0.65, frac  # ~half the lane-episodes are potential lanes
    assert 0.35 < float(np.asarray(state.potential_lane).mean()) < 0.65
