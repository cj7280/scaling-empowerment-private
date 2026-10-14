"""Checkpoint evaluation preserves continuous probe directions and exploration."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.training.pqn import branch_eval
from empowerment_warp.training.pqn.checkpoint import _save_checkpoint
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig


def test_probe_library_matches_original_training_evaluator_exactly():
    config = SimpleNamespace(seed=2, num_skills=30, skill_dim=16)
    old = np.random.RandomState(12347).randn(30, 16)
    old /= np.linalg.norm(old, axis=1, keepdims=True) + 1e-8
    library = branch_eval.evaluation_skill_library(config)
    np.testing.assert_array_equal(library, old.astype(np.float32))
    np.testing.assert_allclose(np.linalg.norm(library, axis=1), 1, atol=2e-7)


def test_real_continuous_checkpoint_restores_library_and_policy_schedule(tmp_path, monkeypatch):
    config = PotentialPQNRunConfig(
        environment='craftax:classic-symbolic', run_dir=str(tmp_path), num_envs=2,
        rollout_steps=4, pqn_num_minibatches=1, pqn_hidden_size=16,
        discrete_skills=False, potential_skill_samples=8, num_skills=4, skill_dim=3,
        pqn_schedule_env_steps=800, pqn_epsilon_fraction=.5,
    )
    env = branch_eval.PQNEnv(config.environment, num_envs=1, episode_length=4, reset_ratio=1)
    original = config.agent_class().create(
        config.seed, jnp.zeros((1, env.observation_size)),
        jnp.zeros((1, env.action_size)), config.agent_config(),
    ).replace(env_steps=jnp.asarray(160), n_updates=jnp.asarray(20))
    (tmp_path/'config.json').write_text(json.dumps(config.to_dict()))
    (tmp_path/'reference_manifest.json').write_text(json.dumps({'agent_config': dict(original.config)}))
    _save_checkpoint(tmp_path, 160, original)
    loaded = branch_eval.load_run(tmp_path)
    assert loaded[1].skill_library is None  # continuous controller stays continuous
    assert loaded[2].shape == (4, 3)
    np.testing.assert_array_equal(loaded[2], loaded[3].library)

    def evaluate(config, agent, library, **kwargs):
        assert int(agent.env_steps) == 160 and int(agent.n_updates) == 20
        np.testing.assert_allclose(agent.exploration_epsilon(), original.exploration_epsilon())
        prefix = branch_eval.PQNEnv(config.environment, num_envs=1, episode_length=4, reset_ratio=1)
        branches = branch_eval.PQNEnv(config.environment, num_envs=2, episode_length=4, reset_ratio=1)
        rollout = branch_eval.make_switching_rollout(prefix, branches, agent,
            prefix_steps=2, episode_length=4, skills_per_context=2)
        arrays = rollout({'params': agent.network.params, 'batch_stats': agent.batch_stats},
                         agent.exploration_epsilon(), jax.random.PRNGKey(1), jax.random.PRNGKey(2),
                         library[:2])
        assert arrays['skills'].shape == (2, 3)
        assert np.isfinite(np.asarray(arrays['episode_returns'])).all()
        return {'epsilon': float(agent.exploration_epsilon())}

    monkeypatch.setattr(branch_eval, 'evaluate_agent', evaluate)
    row = branch_eval.evaluate_checkpoint(tmp_path, tmp_path/'checkpoints/params_000000000160.pkl', loaded=loaded)
    assert row['epsilon'] < config.pqn_epsilon_start
