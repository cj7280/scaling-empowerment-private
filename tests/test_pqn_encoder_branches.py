"""Encoder-only PQN branches retain fixed marginal anchors and valid prefixes."""

import jax
import jax.numpy as jnp
import pickle
from types import SimpleNamespace

from empowerment_warp.agents.see_pqn import PotentialPQNAgent, get_config as potential_pqn_config
from empowerment_warp.utils.objectives import vmf_forward_loss
from empowerment_warp.agents.skill_classifier import SkillConfig, SkillLearner
from empowerment_warp.envs.games import JaxGameBatchEnv
from empowerment_warp.envs.wrappers import FlatObservationEnv
from empowerment_warp.training.pqn.archive import PQNEncoderArchive
from empowerment_warp.training.pqn.collector import make_pqn_encoder_branch_sweep
from empowerment_warp.training.pqn.checkpoint import (
    _save_checkpoint, _save_full_state, _tree_restore,
)
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig


def _agent(env, *, num_envs=2, skill_dim=4):
    config = dict(potential_pqn_config())
    config.update(
        pqn_hidden_size=16, pqn_num_layers=1, pqn_expected_num_envs=num_envs,
        pqn_rollout_steps=4, pqn_num_epochs=1, pqn_num_minibatches=1,
        pqn_total_env_steps=128, skill_dim=skill_dim, num_skills=skill_dim,
        pqn_recurrent_state_size=4 * 16 + 1,
        # Branches start from fresh memory here; get_config preserves it across the switch.
        pqn_preserve_memory_at_switch=False,
    )
    library = jnp.eye(skill_dim, dtype=jnp.float32)
    agent = PotentialPQNAgent.create(
        0, jnp.zeros((1, env.observation_size), jnp.float32),
        jnp.zeros((1, env.action_size), jnp.float32), config,
    )
    return agent, library


def test_branch_uses_real_copied_state_and_fixed_anchor():
    # The real game state carries much more than its visible observation.
    env = FlatObservationEnv(JaxGameBatchEnv(
        "craftax:classic-symbolic", num_envs=2, episode_length=3))
    agent, library = _agent(env)
    sampler = lambda key, n: library[jax.random.randint(key, (n,), 0, 4)]
    source = env.reset(jax.random.split(jax.random.PRNGKey(1), 2))
    # Move to non-reset physical states, including all hidden game fields.
    source = env.step(source, jnp.asarray([0, 1], jnp.int32))
    source_before = jax.tree_util.tree_map(lambda x: jnp.array(x) if hasattr(x, "shape") else x, source)
    sweep = make_pqn_encoder_branch_sweep(
        env, agent, num_envs=2, skill_dim=4, horizon=4,
        skill_sampler=sampler, observation_fn=lambda obs: obs,
    )
    variables = {"params": agent.network.params, "batch_stats": agent.batch_stats}
    batch, valid_count = sweep(variables, source, jnp.asarray(0.2), jax.random.PRNGKey(2),
                               jnp.asarray(1, jnp.int32))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(
        jax.tree_util.tree_leaves(source), jax.tree_util.tree_leaves(source_before)))
    assert batch["observations"].shape == (8, env.observation_size)
    assert bool(jnp.all(batch["anchor_observations"].reshape(4, 2, -1) == source.obs))
    active = jnp.linalg.norm(batch["skills"].reshape(4, 2, -1), axis=-1) > 0
    assert int(valid_count) == int(active.sum())
    # The wrapper branch clock reaches its own limit and excludes terminal/padding rows.
    assert bool(jnp.all(~active[2:]))


def test_branch_archive_samples_fixed_anchor_and_never_crosses_dead_padding():
    archive = PQNEncoderArchive(num_envs=2, capacity=16, episode_length=4,
                                future_gamma=0.5, max_future_steps=1,
                                fixed_anchors=True)
    state = archive.initialize(2, 3, 2)
    observations = jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3)
    anchors = jnp.tile(jnp.asarray([[101., 102., 103.], [201., 202., 203.]]), (4, 1))
    skills = jnp.tile(jnp.eye(2), (4, 1))
    skills = skills.at[4:].set(0.0)  # both branches ended after two outcomes
    state = archive.insert(state, {
        "observations": observations, "anchor_observations": anchors,
        "skills": skills,
        "episode_ids": jnp.tile(jnp.asarray([7, 8]), 4),
        "timesteps": jnp.repeat(jnp.arange(4), 2),
    })
    assert int(state.active_rows) == 4
    batch, _ = archive.sample_encoder_batch(state, jax.random.PRNGKey(3), 16)
    assert bool(jnp.all(jnp.linalg.norm(batch.skills, axis=-1) > 0))
    assert bool(jnp.all(batch.anchor_observations[:, 0] >= 100))
    assert bool(jnp.all(batch.future_valid * (batch.future_observations[:, 0] >= 12) == 0))

    # The learner's potential objective reads `observations` as phi's start.
    # This catches a superficially fixed archive that stores anchors but still
    # optimizes outcome_t -> outcome_(t+k), which is not conditional marginal MI.
    exact, _ = archive._batch_from_indices(
        state, jnp.asarray([0, 1], jnp.int32), jax.random.PRNGKey(4))
    learner = SkillLearner(SkillConfig(
        method="potential", observation_dim=3, skill_dim=2,
        discrete_skills=False, encoder_objective="vmf_forward",
        hidden_dims=(8, 8), phi_norm_cap=0.0, layer_norm=False,
    ))
    learner_state = learner.initialize(jax.random.PRNGKey(5))
    actual = learner.evaluate_preprocessed(learner_state, exact, jax.random.PRNGKey(6))
    fixed_delta = learner.displacement(learner_state.params,
                                       exact.anchor_observations,
                                       exact.future_observations)
    oracle_loss, _ = vmf_forward_loss(
        fixed_delta, exact.skills, temperature=learner.config.temperature,
        valid=exact.future_valid.astype(jnp.bool_))
    sampled_rows = observations[:2]
    wrong_delta = learner.displacement(learner_state.params,
                                       sampled_rows, exact.future_observations)
    wrong_loss, _ = vmf_forward_loss(
        wrong_delta, exact.skills, temperature=learner.config.temperature,
        valid=exact.future_valid.astype(jnp.bool_))
    assert bool(jnp.allclose(actual.loss, oracle_loss, atol=1e-6))
    assert float(jnp.abs(actual.loss - wrong_loss)) > 1e-5
    assert bool(jnp.array_equal(exact.observations, exact.anchor_observations))


def test_default_live_archive_and_branch_recipe_scales():
    live = PotentialPQNRunConfig(environment="craftax:classic-symbolic", run_dir="unused")
    branches = PotentialPQNRunConfig(environment="craftax:classic-symbolic", run_dir="unused",
                                    encoder_data_source="branches")
    assert live.encoder_data_source == "live"
    assert live.effective_recipe()["branch_horizon_steps"] == 0
    recipe = branches.effective_recipe()
    assert recipe["branch_horizon_steps"] == 900
    assert recipe["branch_sweep_interval_live_vector_steps"] == 1000
    assert recipe["branch_physical_steps_per_live_transition"] == 0.9
    assert recipe["collected_transitions_per_chunk"] == 1024 * 128
    assert recipe["encoder_updates_per_chunk_after_ready"] == 16
    assert recipe["encoder_sampled_rows_per_chunk_after_ready"] == 16 * 4096

    archive = PQNEncoderArchive(num_envs=2, capacity=8, episode_length=4,
                                future_gamma=0.5)
    state = archive.initialize(2, 3, 2)
    assert state.data.anchor_observations is None
    state = archive.insert(state, {
        "observations": jnp.arange(12, dtype=jnp.float32).reshape(4, 3),
        "skills": jnp.tile(jnp.eye(2), (2, 1)),
        "episode_ids": jnp.tile(jnp.arange(2), 2),
        "timesteps": jnp.repeat(jnp.arange(2), 2),
    })
    batch, _ = archive.sample_encoder_batch(state, jax.random.PRNGKey(5), 4)
    assert bool(jnp.array_equal(batch.anchor_observations, batch.observations))


def test_branch_archive_full_state_roundtrip(tmp_path):
    archive = PQNEncoderArchive(num_envs=2, capacity=8, episode_length=4,
                                future_gamma=0.5, fixed_anchors=True)
    template = archive.initialize(2, 3, 2)
    state = archive.insert(template, {
        "observations": jnp.arange(12, dtype=jnp.float32).reshape(4, 3),
        "anchor_observations": jnp.full((4, 3), 7.0),
        "skills": jnp.tile(jnp.eye(2), (2, 1)),
        "episode_ids": jnp.tile(jnp.arange(2), 2),
        "timesteps": jnp.repeat(jnp.arange(2), 2),
    })
    rng = jax.random.PRNGKey(9)
    path = _save_full_state(
        tmp_path, 4, agent={"x": jnp.arange(2)}, skill_state={"y": jnp.arange(3)},
        archive_state=state, carry={"z": jnp.arange(4)}, rng=rng,
        global_step=4, segment=1, expl_acc={"n": 0.0},
        last_media_step=None, last_eval_step=None, keep=True,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    restored = _tree_restore(template, payload["archive_leaves"])
    for original, reloaded in zip(jax.tree_util.tree_leaves(state),
                                  jax.tree_util.tree_leaves(restored)):
        assert bool(jnp.array_equal(original, reloaded))
    assert bool(jnp.array_equal(rng, payload["rng"]))


def test_params_checkpoint_retention_keeps_latest_only(tmp_path):
    agent = SimpleNamespace(
        env_steps=jnp.asarray(2), n_updates=jnp.asarray(1),
        network=SimpleNamespace(params={"w": jnp.ones((2,))}),
        batch_stats={},
    )
    _save_checkpoint(tmp_path, 2, agent, keep_checkpoints=1)
    _save_checkpoint(tmp_path, 4, agent, keep_checkpoints=1)
    assert [p.name for p in (tmp_path / "checkpoints").glob("params_*.pkl")] == [
        "params_000000000004.pkl"]
