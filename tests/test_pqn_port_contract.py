"""Contract tests for recurrent potential-PQN.

CPU-only and fast: they lock the agent math (recurrent state layout,
Q(lambda)+MSE update) and the warp-native collector's recurrence contract
(flat T-major batch, rnn_resets[t0] all-reset, initial_recurrent_state wiring).
Uses the retained Craftax Classic symbolic environment for collector checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from empowerment_warp.envs.games import JaxGameBatchEnv
from empowerment_warp.envs.wrappers import FlatObservationEnv


def _potential_cfg(num_envs, chunk_steps, skill_dim=8):
    from empowerment_warp.agents.see_pqn import get_config as potential_pqn_config
    cfg = dict(potential_pqn_config())
    cfg.update(
        pqn_hidden_size=16, pqn_num_layers=1, pqn_expected_num_envs=num_envs,
        pqn_rollout_steps=chunk_steps, pqn_num_minibatches=2, pqn_num_epochs=2,
        pqn_total_env_steps=num_envs * chunk_steps * 8, skill_dim=skill_dim, num_skills=skill_dim,
        # Continuous skills: the z=0 value averages Monte Carlo unit-sphere draws.
        discrete_skills=False, potential_skill_samples=4,
        # These contracts check the fresh-memory (reset-at-switch) value; get_config preserves memory.
        pqn_preserve_memory_at_switch=False,
    )
    return cfg


def test_potential_agent_two_q_and_potential_values():
    from empowerment_warp.agents.see_pqn import PotentialPQNAgent
    num_envs, chunk_steps, obs, actions, sk = 4, 4, 20, 6, 8
    cfg = _potential_cfg(num_envs, chunk_steps, sk)
    agent = PotentialPQNAgent.create(
        0, jnp.zeros((2, obs), jnp.float32), jnp.zeros((2, actions), jnp.float32), cfg,
    )
    # two-Q recurrent state: potential (c,h) + effective (c,h) + last action = 4*hidden + 1.
    assert agent.recurrent_state_size() == 4 * 16 + 1
    assert set(agent.network.params.keys()) == {"modules_potential_q", "modules_effective_q"}
    o = jax.random.normal(jax.random.PRNGKey(1), (num_envs, obs))
    # The potential averages Monte Carlo skill draws at the current anchor.
    pv = agent._potential_values(o, jax.random.PRNGKey(2))
    assert pv.shape == (num_envs,) and bool(jnp.all(jnp.isfinite(pv)))


def test_potential_collector_skill_schedule():
    from empowerment_warp.agents.see_pqn import PotentialPQNAgent
    from empowerment_warp.training.pqn.collector import (
        initialize_potential_pqn_collector, make_potential_pqn_chunk,
    )
    num_envs, chunk_steps, sk, ps = 4, 6, 8, 2
    env = FlatObservationEnv(JaxGameBatchEnv("craftax:classic-symbolic", num_envs=num_envs, episode_length=16))
    obs, actions = env.observation_size, env.action_size
    cfg = _potential_cfg(num_envs, chunk_steps, sk)
    lib = jnp.eye(sk, dtype=jnp.float32)
    agent = PotentialPQNAgent.create(
        0, jnp.zeros((2, obs), jnp.float32), jnp.zeros((2, actions), jnp.float32), cfg,
    )
    sampler = lambda key, n: lib[jax.random.randint(key, (n,), 0, sk)]
    collect = make_potential_pqn_chunk(
        env, agent, chunk_steps=chunk_steps, num_envs=num_envs, skill_dim=sk,
        potential_steps=ps, skill_sampler=sampler,
    )
    carry = initialize_potential_pqn_collector(
        env, agent, jax.random.PRNGKey(3), num_envs=num_envs, skill_dim=sk,
        potential_steps=ps, skill_sampler=sampler,
    )
    pv = {"params": agent.network.params, "batch_stats": agent.batch_stats}
    _, batch, _ = collect(pv, carry, agent.exploration_epsilon())
    for key in ("observations", "next_observations", "anchor_observations", "skills",
                "next_skills", "next_anchor_observations", "episode_ids"):
        assert key in batch
    skon = jnp.linalg.norm(batch["skills"].reshape(chunk_steps, num_envs, sk), axis=-1) > 1e-6
    # zero-skill prefix then activation (potential_steps=2): first two steps off, later on.
    assert float(skon[0].mean()) == 0.0
    assert float(skon[-1].mean()) == 1.0


def test_frozen_reward_assembly():
    from empowerment_warp.agents.see_pqn import PotentialPQNAgent
    from empowerment_warp.agents.skill_classifier import SkillConfig, SkillLearner
    from empowerment_warp.agents.see_pqn import make_frozen_reward_fn
    n, obs, actions, sk, gamma = 12, 20, 6, 8, 0.95
    cfg = _potential_cfg(n, 4, sk)
    lib = jnp.eye(sk, dtype=jnp.float32)
    agent = PotentialPQNAgent.create(
        0, jnp.zeros((2, obs), jnp.float32), jnp.zeros((2, actions), jnp.float32), cfg,
    )
    skill = SkillLearner(SkillConfig(
        method="potential", observation_dim=obs, skill_dim=sk, discrete_skills=False,
        num_skills=0, potential_reward_reference="anchor",
    ))
    skill_state = skill.initialize(jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    o = jax.random.normal(key, (n, obs)); no = jax.random.normal(key, (n, obs))
    anc = jax.random.normal(key, (n, obs))
    skills = lib[jax.random.randint(key, (n,), 0, sk)] * (jax.random.uniform(key, (n, 1)) > 0.5)
    ts = jnp.zeros((n,), jnp.float32)
    frozen_fn = make_frozen_reward_fn(skill, gamma, zero_eps=1e-6)
    frozen, info = frozen_fn(agent, skill_state, o, no, anc, skills, ts, key)
    assert frozen.shape == (n,) and bool(jnp.all(jnp.isfinite(frozen)))
    # On an all-skill-off batch the frozen reward is exactly (1-gamma) * skill-0 value,
    # averaged over the same Monte Carlo skill draws (the value half of the reward key).
    frozen_off, _ = frozen_fn(agent, skill_state, o, no, anc, jnp.zeros_like(skills), ts, key)
    potential = agent._potential_values(o, jax.random.split(key)[1])
    assert jnp.allclose(frozen_off, (1.0 - gamma) * potential, atol=1e-5)


def test_full_state_roundtrip_restores_every_leaf(tmp_path):
    """The full-state checkpoint restores an arbitrary pytree (optax-style NamedTuples, nested
    dicts, scalars) leaf-for-leaf through a fresh template, and keeps only the newest file."""
    import collections
    import numpy as np
    from empowerment_warp.training.pqn.checkpoint import _save_full_state, _tree_restore, _latest_full_state
    Opt = collections.namedtuple("Opt", "mu nu count")
    def make(seed):
        k = jax.random.PRNGKey(seed)
        return {"params": {"w": jax.random.normal(k, (3, 2)), "b": jnp.ones((2,)) * seed},
                "opt": Opt(mu=jnp.full((3, 2), seed), nu=jnp.zeros((3, 2)), count=jnp.asarray(seed, jnp.int32)),
                "step": jnp.asarray(seed * 10, jnp.int32)}
    live, template = make(7), make(0)
    for step in (100, 200):
        _save_full_state(tmp_path, step, agent=live, skill_state=live["opt"], archive_state=live["params"], carry=live,
                         rng=jax.random.PRNGKey(3), global_step=step, segment=step // 10,
                         expl_acc={"n": 1.0, "ach": np.zeros(2)}, last_media_step=None, last_eval_step=step)
    files = sorted((tmp_path / "checkpoints").glob("full_state_*.pkl"))
    assert [f.name for f in files] == ["full_state_000000000200.pkl"]      # intermediate states rotate
    import pickle as _pkl
    assert "archive_leaves" in _pkl.load(open(files[0], "rb")), "the archive IS stored by default (user: necessary)"
    _save_full_state(tmp_path, 250, agent=live, skill_state=live["opt"], archive_state=live["params"], carry=live,
                     rng=jax.random.PRNGKey(3), global_step=250, segment=25, expl_acc={"n": 1.0, "ach": np.zeros(2)},
                     last_media_step=None, last_eval_step=250, include_archive=False)
    assert "archive_leaves" not in _pkl.load(open(_latest_full_state(tmp_path), "rb")), "include_archive=False drops it"
    # a budget-end state (keep=True) is never rotated by later saves, and the latest-step lookup sees both kinds
    _save_full_state(tmp_path, 300, agent=live, skill_state=live["opt"], archive_state=live["params"], carry=live,
                     rng=jax.random.PRNGKey(3), global_step=300, segment=30,
                     expl_acc={"n": 1.0, "ach": np.zeros(2)}, last_media_step=None, last_eval_step=300, keep=True)
    _save_full_state(tmp_path, 400, agent=live, skill_state=live["opt"], archive_state=live["params"], carry=live,
                     rng=jax.random.PRNGKey(3), global_step=400, segment=40,
                     expl_acc={"n": 1.0, "ach": np.zeros(2)}, last_media_step=None, last_eval_step=400)
    names = sorted(f.name for f in (tmp_path / "checkpoints").glob("full_state_*.pkl"))
    assert names == ["full_state_000000000400.pkl", "full_state_keep_000000000300.pkl"], names
    assert _latest_full_state(tmp_path).name == "full_state_000000000400.pkl"
    (tmp_path / "checkpoints" / "full_state_000000000400.pkl").unlink()
    assert _latest_full_state(tmp_path).name == "full_state_keep_000000000300.pkl"
    import pickle
    payload = pickle.load(open(_latest_full_state(tmp_path), "rb"))
    assert payload["env_steps"] == 300
    payload = {"agent_leaves": payload["agent_leaves"], "env_steps": 200, "segment": 20}
    restored = _tree_restore(template, payload["agent_leaves"])
    for a, b in zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(live)):
        assert jnp.allclose(jnp.asarray(a), jnp.asarray(b))
    assert payload["env_steps"] == 200 and payload["segment"] == 20
    try:
        _tree_restore(template["params"], payload["agent_leaves"])       # structure mismatch is refused
    except ValueError:
        pass
    else:
        raise AssertionError("structure mismatch was not refused")


def test_resume_builds_from_the_frozen_config_and_trims_aborted_rows(tmp_path):
    """Resume robustness: (1) `resume_config_from_run` reconstructs the run's config
    from config.json alone -- launch-env knobs like SKILL_DIM cannot go missing -- and a larger
    budget re-pins the LR schedule horizon to the new budget; a smaller budget is refused;
    (2) `_trim_logs_past` drops metric/eval rows written after the checkpoint by a cancelled
    attempt, keeping the originals beside the trimmed files."""
    import json
    from empowerment_warp.training.pqn.checkpoint import _trim_logs_past, resume_config_from_run
    from empowerment_warp.training.pqn.config import PotentialPQNRunConfig
    frozen = PotentialPQNRunConfig(run_dir=str(tmp_path), environment="craftax:classic-symbolic", skill_dim=8,
                                   total_env_steps=1_000, pqn_schedule_env_steps=1_000, sample_hindsight_anchors=False)
    (tmp_path / "config.json").write_text(json.dumps(frozen.to_dict()))
    cfg = resume_config_from_run(tmp_path, PotentialPQNRunConfig, total_env_steps=2_000)
    assert cfg.skill_dim == 8 and cfg.sample_hindsight_anchors is False, "frozen knobs carried, not recipe defaults"
    assert cfg.total_env_steps == 2_000 and cfg.pqn_schedule_env_steps == 2_000, "schedule re-pinned to the new budget"
    same = resume_config_from_run(tmp_path, PotentialPQNRunConfig)
    assert same.total_env_steps == 1_000 and same.pqn_schedule_env_steps == 1_000
    pinned = resume_config_from_run(tmp_path, PotentialPQNRunConfig, total_env_steps=2_000, schedule_env_steps=1_500)
    assert pinned.pqn_schedule_env_steps == 1_500
    import pytest
    with pytest.raises(ValueError):
        resume_config_from_run(tmp_path, PotentialPQNRunConfig, total_env_steps=500)
    with pytest.raises(ValueError):
        resume_config_from_run(tmp_path, PotentialPQNRunConfig, overrides={"no_such_knob": 1})
    rows = [{"env_steps": s, "x": s} for s in (0, 100, 200, 300, 400, 250, 350)]   # 250/350: an aborted attempt after ckpt 200
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "eval_reference.jsonl").write_text(json.dumps({"env_steps": 0}) + "\n" + json.dumps({"env_steps": 300}) + "\n")
    trimmed = _trim_logs_past(tmp_path, 200)
    kept = [json.loads(l)["env_steps"] for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert kept == [0, 100, 200] and trimmed == {"metrics.jsonl": 4, "eval_reference.jsonl": 1}
    assert (tmp_path / "metrics.jsonl.pre_resume_200").exists() and (tmp_path / "eval_reference.jsonl.pre_resume_200").exists()
    assert [json.loads(l)["env_steps"] for l in (tmp_path / "eval_reference.jsonl").read_text().splitlines()] == [0]
