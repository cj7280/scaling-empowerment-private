from __future__ import annotations

import numpy as np
import json
import pytest
from types import SimpleNamespace

from empowerment_warp.media.locomotion import _prefix_report
from empowerment_warp.media.locomotion import _video_skills
from empowerment_warp.media.locomotion import create_locomotion_branch_media


def test_locomotion_prefix_report_checks_agent_and_ball_state() -> None:
    arrays = {
        "qpos": np.zeros((5, 4, 22), dtype=np.float32),
        "qvel": np.zeros((5, 4, 20), dtype=np.float32),
        "agent_positions": np.zeros((5, 4, 2), dtype=np.float32),
        "ball_positions": np.zeros((5, 4, 2), dtype=np.float32),
    }
    arrays["ball_positions"][4, 3, 0] = 1.0
    report = _prefix_report(arrays, activation_index=2)
    assert report["valid"]
    arrays["agent_positions"][2, 1, 0] = 1e-3
    report = _prefix_report(arrays, activation_index=2)
    assert not report["valid"]
    assert np.isclose(report["max_abs_delta_by_key"]["agent_positions"], 1e-3)


def test_stochastic_media_rejects_cached_deterministic_trace(tmp_path) -> None:
    np.savez(tmp_path / "branch_trace.npz",
             metadata=np.asarray(json.dumps({"policy_mode": "deterministic"})))
    with pytest.raises(ValueError, match="use a separate output directory"):
        create_locomotion_branch_media(tmp_path / "run", tmp_path, upload_wandb=False)


def test_2d_video_uses_signed_axes_without_metric_probe_library() -> None:
    config = SimpleNamespace(discrete_skills=False, skill_dim=2)
    # No experiment is needed: video commands do not depend on the random probe set.
    skills, indices, selection = _video_skills(config, 4)
    np.testing.assert_array_equal(skills, [[1, 0], [0, 1], [-1, 0], [0, -1]])
    assert indices == (0, 1, 2, 3)
    assert selection == "uniform_circle"


@pytest.mark.parametrize("dim,discrete", [(4, False)])
def test_other_video_skill_spaces_keep_the_existing_library(monkeypatch, dim, discrete) -> None:
    import jax.numpy as jnp
    from empowerment_warp.media import locomotion

    library = jnp.arange(30 * dim, dtype=jnp.float32).reshape(30, dim)
    monkeypatch.setattr(locomotion, "eval_skill_library", lambda *args: library)
    skills, indices, selection = _video_skills(
        SimpleNamespace(discrete_skills=discrete, skill_dim=dim), 4
    )
    assert indices == (0, 7, 15, 22)
    np.testing.assert_array_equal(skills, library[np.asarray(indices)])
    assert selection == "eval_library_subset"


def test_2d_media_rejects_cached_random_directions(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"skill_dim": 2, "discrete_skills": False}))
    np.savez(tmp_path / "branch_trace.npz",
             metadata=np.asarray(json.dumps({"policy_mode": "stochastic"})))
    with pytest.raises(ValueError, match="circle of video skills"):
        create_locomotion_branch_media(run, tmp_path, upload_wandb=False)


@pytest.mark.parametrize('override', [None, 7])
def test_media_rejects_cached_schedule_or_override(tmp_path, override):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({
        'skill_dim': 4, 'discrete_skills': False, 'seed': 0,
        'potential_steps': 100, 'random_potential_switch_max_steps': 200,
    }))
    np.savez(tmp_path / 'branch_trace.npz', metadata=np.asarray(json.dumps({
        'policy_mode': 'stochastic', 'seed': 50_000, 'horizon_steps': 400,
        'prefix_steps': 100,
        'potential_schedule': {'random_potential_switch_max_steps': 0,
                               'nominal_potential_steps': 100, 'prefix_steps_override': None},
    })))
    with pytest.raises(ValueError, match='different potential schedule'):
        create_locomotion_branch_media(run, tmp_path, prefix_steps=override, upload_wandb=False)


@pytest.mark.parametrize('override', [None, 2])
def test_zero_draw_keeps_potential_comparison_and_records_fixed_override(monkeypatch, tmp_path, override):
    from dataclasses import dataclass
    from typing import NamedTuple
    import jax.numpy as jnp
    from empowerment_warp.media import locomotion

    @dataclass
    class Config:
        environment: str = 'ant'
        num_envs: int = 1
        skill_dim: int = 4
        discrete_skills: bool = False
        seed: int = 0
        potential_steps: int = 0
        episode_length: int = 4
        skill_start_mode: str = 'two_phase'
        live_potential_only: bool = False
        random_potential_switch_max_steps: int = 3
        allow_reference_deviation: bool = True
        encoder_obs_mode: str = 'test'
        locomotion_observation_contract: str = 'test'

    class Data(NamedTuple):
        qpos: object
        qvel: object
        ctrl: object
        mocap_pos: object
        mocap_quat: object

    class State(NamedTuple):
        data: Data
        obs: object

    class Env:
        soccer = False
        config = SimpleNamespace(ctrl_dt=.01)
        def reset(self, keys):
            obs = jnp.zeros((len(keys), 2))
            data = Data(obs, obs, obs, jnp.zeros((len(keys), 0, 3)), jnp.zeros((len(keys), 0, 4)))
            return State(data, obs)
        def step(self, state, actions):
            obs = state.obs + actions
            return State(state.data._replace(qpos=obs), obs)

    def actions(state, obs, anchors, skills, keys, deterministic):
        assert not deterministic
        return skills[:, :2] + 1

    def gather(source, template, indices):
        return locomotion.jax.tree_util.tree_map(lambda x: x[indices], source)

    monkeypatch.setattr(locomotion, 'load_run', lambda _: (
        Config(), SimpleNamespace(sac=SimpleNamespace(sample_actions=actions)), SimpleNamespace(sac=None, env_steps=10)))
    monkeypatch.setattr(locomotion, 'make_environment', lambda _: Env())
    monkeypatch.setattr(locomotion, 'eval_skill_library', lambda *args: jnp.eye(4))
    monkeypatch.setattr(locomotion, 'evaluation_switch_steps', lambda *args: jnp.array([0]))
    monkeypatch.setattr(locomotion, 'gather_env_state_lanes', gather)
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/latest.json').write_text(json.dumps({'step': 10}))
    output = locomotion.collect_locomotion_branch_trace(
        tmp_path, tmp_path / 'trace.npz', horizon_steps=4, prefix_steps=override,
    )
    with np.load(output) as archive:
        metadata = json.loads(str(archive['metadata']))
        assert archive['skills'].shape == (5, 4)
        prefix = override if override is not None else 0
        np.testing.assert_array_equal(archive['potential_switch_steps'], np.full(5, prefix))
        assert archive['observations'].shape[1] == 5
    assert metadata['potential_continuation_lane'] == 0
    assert metadata['prefix_steps'] == prefix
    assert metadata['potential_schedule']['prefix_steps_override'] == override
    assert metadata['potential_schedule']['schedule'] == (
        'uniform_integer' if override is None else 'fixed_override')
