"""Regression coverage for dense PQN archive semantics and encoder cadence."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillLearner
from empowerment_warp.training.pqn.archive import PQNEncoderArchive
from empowerment_warp.agents.see_pqn import make_encoder_update_burst
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig
from empowerment_warp.training.pqn.runner import run_potential_pqn_training


def _rows(start=0, count=8, width=2):
    ids = jnp.arange(start, start + count)
    time = ids // width
    lane = ids % width
    # Early termination every three steps, before configured horizon eight.
    return dict(
        observations=jnp.stack([ids, lane, time % 3], axis=-1).astype(jnp.float32),
        skills=jnp.eye(2)[lane],
        episode_ids=(time // 3) * width + lane,
        timesteps=time % 3,
    )


def _archive(capacity=9):
    archive = PQNEncoderArchive(num_envs=2, capacity=capacity, episode_length=8,
                                future_gamma=0.95)
    return archive, archive.initialize(2, 3, 2)


def test_ring_counts_transitions_immediately_and_wraps_at_nonvector_capacity():
    archive, state = _archive()
    insert = jax.jit(archive.insert)
    state = insert(state, _rows())
    assert int(state.size) == 8  # No completed-episode admission gate.
    state = insert(state, _rows(8))
    assert int(state.size) == 9 and int(state.total_inserted_rows) == 16
    assert int(state.active_rows) == 9
    slots = archive._storage_indices(state, jnp.arange(9))
    np.testing.assert_array_equal(state.data.obs_tail[slots, 0], np.arange(7, 16))
    # One insert larger than capacity must still retain exactly its newest rows.
    state = insert(state, _rows(16, 12))
    slots = archive._storage_indices(state, jnp.arange(9))
    np.testing.assert_array_equal(state.data.obs_tail[slots, 0], np.arange(19, 28))
    assert int(state.active_rows) == 9
    off = _rows(28)
    off['skills'] = jnp.zeros_like(off['skills'])
    state = insert(state, off)
    assert int(state.active_rows) == 1  # Old active rows removed on overwrite.


def test_sparse_skill_sampler_fills_batch_without_padded_rows():
    archive, state = _archive(64)
    rows = _rows(0, 64)
    rows['skills'] = jnp.zeros_like(rows['skills']).at[31].set(jnp.array([1., 0.]))
    state = jax.jit(archive.insert)(state, rows)
    sample = jax.jit(archive._sample_skill_indices, static_argnums=(2,))
    # Every output must select the only active row, including the no-candidate fallback.
    for seed in range(4):
        indices = sample(state, jax.random.PRNGKey(seed), 16)
        np.testing.assert_array_equal(indices, np.full(16, 31))


def test_no_future_rows_are_masked_and_empty_archive_is_safe():
    archive, state = _archive()
    sample = jax.jit(archive.sample_encoder_batch, static_argnums=(2,))
    batch, _ = sample(state, jax.random.PRNGKey(0), 8)
    assert float(batch.future_valid.sum()) == 0
    state = archive.insert(state, _rows(0, 2))
    batch, _ = sample(state, jax.random.PRNGKey(0), 8)
    assert float(batch.future_valid.sum()) == 0
    np.testing.assert_array_equal(batch.observations, batch.future_observations)


def test_config_uses_optimizer_step_units_and_rejects_unreachable_readiness():
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused')
    recipe = config.effective_recipe()
    assert recipe['encoder_updates_per_chunk_after_ready'] == 16
    assert recipe['encoder_samples_per_collected_transition'] == 0.5
    assert recipe['archive_capacity_transitions'] == 10_000_000
    assert config.encoder_train_start_rows == 10_000
    assert dataclasses.replace(config, representation_archive_update_ratio=0.1).encoder_updates_per_chunk == 2
    assert dataclasses.replace(config, representation_archive_update_ratio=0).encoder_updates_per_chunk == 0
    assert dataclasses.replace(config, representation_train_start_rows=0).encoder_train_start_rows == 4096
    continuous = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
                                       discrete_skills=False, potential_skill_samples=8)
    assert continuous.skill_config(1345).discrete_reward_softmax is True
    assert continuous.skill_config(1345).encoder_max_grad_norm == 0
    assert config.max_future_steps == 0
    with pytest.raises(ValueError, match='threshold'):
        dataclasses.replace(config, representation_archive_size=100)


def test_cli_resolve_overrides_without_creating_run(capsys, tmp_path):
    from empowerment_warp.cli import main
    run_dir = tmp_path / 'not-created'
    recipe = Path(__file__).resolve().parents[1] / 'configs/craftax_classic/see.yaml'
    main(['resolve', '--config', str(recipe), '--run-dir', str(run_dir),
          '--set', 'representation_archive_size=12345',
          '--set', 'representation_archive_update_ratio=0.5',
          '--set', 'encoder_batch_size=512'])
    settings = json.loads(capsys.readouterr().out)['settings']
    config = PotentialPQNRunConfig(**settings)
    assert config.effective_recipe()['encoder_updates_per_chunk_after_ready'] == 8
    assert settings['representation_archive_size'] == 12345
    assert settings['encoder_batch_size'] == 512
    assert not run_dir.exists()


def test_real_encoder_burst_advances_each_optimizer_step():
    archive, state = _archive(16)
    state = archive.insert(state, _rows(0, 16))
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
                                  skill_dim=2, num_skills=2, encoder_hidden_dims=(8,),
                                  discrete_skills=False, potential_skill_samples=2)
    learner = SkillLearner(config.skill_config(3))
    before = learner.initialize(jax.random.PRNGKey(1))
    burst = make_encoder_update_burst(learner, archive, batch_size=8, update_steps=16)
    after, key, metrics, _ = burst(before, state, jax.random.PRNGKey(2))
    assert int(after.step) == 16
    assert np.isfinite(float(metrics.loss))
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(before.params), jax.tree_util.tree_leaves(after.params)))
    again, _, _, _ = burst(after, state, key)
    assert int(again.step) == 32


def test_real_runner_crosses_readiness_and_wraps_archive(monkeypatch, tmp_path):
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    # This archive test uses four-action episodes, not the production 100+900 evaluation.
    monkeypatch.setattr('empowerment_warp.training.pqn.branch_eval.POTENTIAL_STEPS', 1)
    config = PotentialPQNRunConfig(
        environment='craftax:classic-symbolic', run_dir=str(tmp_path / 'run'), allow_cpu=True,
        num_envs=4, rollout_steps=4, episode_length=8, total_env_steps=48,
        pqn_hidden_size=8, pqn_num_layers=1, pqn_num_epochs=2, pqn_num_minibatches=2,
        skill_dim=4, num_skills=4, potential_steps=1, encoder_hidden_dims=(8,),
        discrete_skills=False, potential_skill_samples=2,
        achieved_skill_relabeling=False, relabel_unit='segment',
        encoder_batch_size=8, representation_archive_size=32, representation_train_start_rows=20,
        eval_episode_length=4, log_every_segments=1,
        eval_contexts=1, eval_skills_per_context=1, eval_video_episodes=0,
    )
    run_dir = run_potential_pqn_training(config)
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    rows = [row for row in rows if 'training/representation_archive/update_steps' in row]
    assert [row['training/representation_archive/update_steps'] for row in rows] == [0, 4, 4]
    assert [row['training/representation_archive/size'] for row in rows] == [16, 32, 32]
    assert rows[-1]['training/representation_archive/total_encoder_steps'] == 8
    summary = json.loads((run_dir / 'train_summary.json').read_text())
    assert summary['encoder_optimizer_steps'] == 8
    assert summary['archive_size_transitions'] == 32
