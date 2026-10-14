from __future__ import annotations

import json
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.checkpointing import CheckpointStore, model_checkpoint
from empowerment_warp.cli import main
from empowerment_warp.training.sac.config import default_recipe
from empowerment_warp.core.rollout import StaggeredBranchState
from empowerment_warp.envs.factory import make_environment


def test_recipes_are_explicit_and_key_door_is_absent() -> None:
    config = default_recipe("builderbench_corner3", "potential", num_envs=128)
    assert config.num_envs == 128
    assert config.target_entropy == -12.5
    assert config.skill_dim == 9
    assert config.num_skills == 30
    assert config.discrete_skills
    assert config.reference_manifest()["contract_fields_match"]
    assert config.to_dict()["hidden_dims"] == [512, 512]
    with pytest.raises(ValueError, match="unknown environment"):
        default_recipe("key_door", "potential")
    with pytest.raises(ValueError, match="unknown algorithm"):
        default_recipe("ant", "csf")
    with pytest.raises(RuntimeError, match="validation"):
        make_environment(config)
    with pytest.raises(ValueError, match="drift detected"):
        default_recipe("builderbench_corner3", "potential", target_entropy=2.0)


def test_optional_training_time_budget_is_positive_and_serialized() -> None:
    config = default_recipe(
        "ant_soccer",
        "potential",
        max_train_seconds=3_000.0,
    )
    assert config.max_train_seconds == 3_000.0
    assert config.to_dict()["max_train_seconds"] == 3_000.0
    assert config.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="max_train_seconds must be positive"):
        default_recipe("ant_soccer", "potential", max_train_seconds=0.0)


def test_random_switch_config_declares_expected_archive_capacity():
    config = default_recipe('ant', 'potential', potential_steps=100,
                            random_potential_switch_max_steps=200,
                            allow_reference_deviation=True)
    assert config.from_dict(config.to_dict()) == config
    manifest = config.reference_manifest()
    assert manifest['representation_active_row_capacity'] is None
    assert manifest['representation_active_row_capacity_expected'] == 3334 * 300
    assert manifest['representation_active_row_capacity_bounds'] == [3334 * 200, 3334 * 400]
    assert 'random_potential_switch_max_steps' in manifest['mismatches']
    with pytest.raises(ValueError, match='random potential switching'):
        default_recipe('ant', 'potential', potential_steps=0,
                       random_potential_switch_max_steps=200, allow_reference_deviation=True)
    with pytest.raises(ValueError, match='random potential switching'):
        default_recipe('ant_soccer', 'potential', potential_steps=100,
                       random_potential_switch_max_steps=200, allow_reference_deviation=True)


def test_anneal_config_round_trip_and_rejects_unnormalized_recipe():
    kwargs = dict(allow_reference_deviation=True, discrete_skills=False, skill_dim=4,
                  encoder_objective='vmf_forward', continuous_reward_temperature=True,
                  continuous_reward_normalization='sphere', potential_reward_reference='anchor',
                  reward_partition_anneal_start_steps=25_000_000,
                  reward_partition_anneal_end_steps=100_000_000)
    config = default_recipe('ant', 'potential', **kwargs)
    assert config.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match='partition annealing'):
        default_recipe('ant', 'potential', **{**kwargs, 'continuous_reward_normalization': 'none'})
    old = config.to_dict()
    old.pop('reward_partition_anneal_start_steps')
    old.pop('reward_partition_anneal_end_steps')
    restored = config.from_dict(old)
    assert restored.reward_partition_anneal_start_steps == restored.reward_partition_anneal_end_steps == 0


def test_optional_progress_eval_cadence_is_positive_and_serialized() -> None:
    config = default_recipe(
        "ant_soccer",
        "potential",
        progress_eval_every_env_steps=50_000_000,
    )
    assert config.progress_eval_every_env_steps == 50_000_000
    assert config.to_dict()["progress_eval_every_env_steps"] == 50_000_000
    assert config.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="progress_eval_every_env_steps must be positive"):
        default_recipe(
            "ant_soccer",
            "potential",
            progress_eval_every_env_steps=0,
        )




def test_orbax_round_trip_and_manifest(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    state = {
        "step": jnp.asarray(3),
        "weights": jnp.arange(4, dtype=jnp.float32),
        "empty": jnp.zeros((2, 0), dtype=jnp.float32),
    }
    path = store.save(3, state, {"environment": "test"})
    assert path.is_dir()
    assert store.latest_step() == 3
    restored = store.restore(state)
    np.testing.assert_array_equal(restored["weights"], state["weights"])
    assert restored["empty"].shape == (2, 0)
    manifest = json.loads((tmp_path / "checkpoints" / "latest.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["payload_kind"] == "full"


def test_model_checkpoint_excludes_collector_and_replay(tmp_path) -> None:
    class State:
        sac = {"weights": jnp.ones((2,), dtype=jnp.float32)}
        skill = {"weights": jnp.ones((3,), dtype=jnp.float32)}
        env_steps = jnp.asarray(7, dtype=jnp.int32)
        collector = {"large": jnp.ones((16,), dtype=jnp.float32)}
        replay = {"large": jnp.ones((16,), dtype=jnp.float32)}

    payload = model_checkpoint(State())
    assert not hasattr(payload, "collector")
    assert not hasattr(payload, "replay")
    store = CheckpointStore(tmp_path / "checkpoints")
    store.save(7, payload, {"environment": "test"}, payload_kind="model")
    restored = store.restore(payload)
    np.testing.assert_array_equal(restored.sac["weights"], payload.sac["weights"])
    assert store.latest_manifest()["payload_kind"] == "model"


def test_evaluation_loads_selected_checkpoint_instead_of_latest(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from empowerment_warp.checkpointing import ModelCheckpoint
    from empowerment_warp.training.sac import evaluation

    config = default_recipe('ant', 'potential')
    (tmp_path/'config.json').write_text(json.dumps(config.to_dict()))
    init = lambda key: {'weight': jnp.zeros(1)}
    monkeypatch.setattr(evaluation, 'require_runtime', lambda config: None)
    monkeypatch.setattr(evaluation, 'make_experiment', lambda config: SimpleNamespace(
        sac=SimpleNamespace(initialize=init), skill=SimpleNamespace(initialize=init)))
    def payload(step):
        return ModelCheckpoint({'weight':jnp.full(1,float(step))},
                               {'weight':jnp.zeros(1)}, jnp.asarray(step, jnp.int32))
    current = CheckpointStore(tmp_path/'checkpoints')
    current.save(10, payload(10), config.to_dict(), payload_kind='model')
    retained = CheckpointStore(tmp_path/'evaluation_checkpoints')
    selected = retained.save(0, payload(0), config.to_dict(), payload_kind='model')
    _, _, state = evaluation.load_run(tmp_path, checkpoint=selected)
    assert int(state.env_steps) == 0
    assert float(state.sac['weight'][0]) == 0
    assert current.latest_step() == 10


def test_full_checkpoint_round_trips_partial_staggered_branch(tmp_path) -> None:
    branch = StaggeredBranchState(
        env_state={"obs": jnp.arange(8, dtype=jnp.float32).reshape(4, 2)},
        policy_anchors=jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
        encoder_anchors=jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
        source_skills=jnp.arange(4, dtype=jnp.float32)[:, None],
        skills=jnp.arange(4, dtype=jnp.float32)[:, None],
        outcomes=jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2),
        valid=jnp.ones((3, 4), dtype=jnp.bool_),
        progress=jnp.asarray([1, 1, 2, 2], dtype=jnp.int32),
        genuine=jnp.asarray([True, True, False, False]),
        alive=jnp.asarray([True, False, True, True]),
        active=jnp.asarray(True),
        skill_shift=jnp.asarray(2, dtype=jnp.int32),
        policy_outcomes=jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2) * 2.0,
        actions=jnp.arange(12, dtype=jnp.float32).reshape(3, 4, 1),
        terminated=jnp.zeros((3, 4), dtype=jnp.bool_),
        truncated=jnp.ones((3, 4), dtype=jnp.bool_),
        alive_rows=jnp.ones((3, 4), dtype=jnp.bool_),
        emit_index=jnp.asarray(5, dtype=jnp.int32),
        anchor_xy=jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
    )
    store = CheckpointStore(tmp_path / "checkpoints")
    store.save(2, branch, {"environment": "test"})
    restored = store.restore(branch)

    assert bool(restored.active)
    assert int(restored.skill_shift) == 2
    np.testing.assert_array_equal(restored.progress, branch.progress)
    np.testing.assert_array_equal(restored.outcomes, branch.outcomes)
    # The policy-facing history is part of the checkpointed branch state, so a
    # resumed run keeps the in-flight rows it will later hand to the learner.
    assert int(restored.emit_index) == 5
    np.testing.assert_array_equal(restored.policy_outcomes, branch.policy_outcomes)
    np.testing.assert_array_equal(restored.actions, branch.actions)
    np.testing.assert_array_equal(restored.alive_rows, branch.alive_rows)


def test_resume_accepts_manifest_after_json_roundtrip(tmp_path, monkeypatch):
    from empowerment_warp.recipes import resolve
    from empowerment_warp.training.sac import runner
    _, config = resolve('configs/ant/see.yaml', run_dir=tmp_path,
                        overrides=['hidden_dims=[16,16]'])
    (tmp_path / 'config.json').write_text(json.dumps(config.to_dict()))
    (tmp_path / 'reference_manifest.json').write_text(json.dumps(config.reference_manifest()))
    monkeypatch.setattr(runner, 'require_runtime', lambda config: None)
    def reached_environment(config):
        raise RuntimeError('resume validation passed')
    monkeypatch.setattr(runner, 'make_experiment', reached_environment)
    with pytest.raises(RuntimeError, match='resume validation passed'):
        runner.train(config, resume=True, stop_after_segments=1)
