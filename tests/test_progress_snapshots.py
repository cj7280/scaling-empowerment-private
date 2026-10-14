from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from empowerment_warp.training.sac.config import default_recipe
from empowerment_warp.training.sac.progress import (
    append_evaluation_metrics,
    create_progress_snapshot,
    due_progress_targets,
    progress_wandb_row,
)


def test_local_eval_history_keeps_step_zero_and_later_metrics(tmp_path):
    import csv
    append_evaluation_metrics(tmp_path, {'env_steps': 0., 'evaluation/distance': 3.})
    append_evaluation_metrics(tmp_path, {'env_steps': 5000000., 'evaluation/distance': 2.,
                                       'evaluation/final_only': 1.})
    rows = [json.loads(line) for line in (tmp_path / 'eval.jsonl').read_text().splitlines()]
    assert [r['env_steps'] for r in rows] == [0., 5000000.]
    with (tmp_path / 'eval.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert table[0]['evaluation/distance'] == '3.0'
    assert table[0]['evaluation/final_only'] == ''
    assert table[1]['evaluation/final_only'] == '1.0'


def test_progress_targets_include_terminal_evaluation() -> None:
    assert due_progress_targets(49_999_999, 500_000_000, 50_000_000) == ()
    assert due_progress_targets(150_000_000, 500_000_000, 50_000_000) == (
        50_000_000,
        100_000_000,
        150_000_000,
    )
    assert due_progress_targets(500_006_912, 500_000_000, 50_000_000) == (
        50_000_000,
        100_000_000,
        150_000_000,
        200_000_000,
        250_000_000,
        300_000_000,
        350_000_000,
        400_000_000,
        450_000_000,
        500_000_000,
    )
    assert due_progress_targets(105, 103, 50) == (50, 100, 103)
    assert due_progress_targets(20, 20, 50) == (20,)


@pytest.mark.parametrize('environment', ['ant', 'ant_soccer', 'builderbench_corner3'])
def test_progress_uses_saved_metric_media_and_retains_checkpoint(monkeypatch, tmp_path, environment):
    run = tmp_path / 'run'
    checkpoint = run / 'checkpoints/step_000050003968'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'weights').write_text('checkpoint')
    (checkpoint.parent / 'latest.json').write_text(json.dumps({'step': 50_003_968}))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index('--output')+1])
        output.mkdir(parents=True)
        assert command[command.index('--checkpoint')+1].endswith('evaluation_checkpoints/step_000050003968')
        if 'branch-media' in command:
            video = output / 'branches.mp4'; video.write_bytes(b'branches')
            (output / 'branch_media_result.json').write_text(json.dumps({'rendered': {
                'metadata': {'checkpoint_step': 50_003_968, 'fps': 38},
                'saved_paths': {'video': str(video)}}}))
            return
        assert command[command.index('--target-step')+1] == '000050000000'
        video = output / 'potential.mp4'; video.write_bytes(b'video')
        metadata = output / 'media.json'
        metadata.write_text(json.dumps({'video': str(video), 'fps': 38}))
        (output / 'metrics.json').write_text(json.dumps({
            'checkpoint_step': 50_003_968, 'policy_mode': 'stochastic',
            'summary': {'episode_ball_disp_xy': {'mean': 1.25}},
            'media': {'metadata': str(metadata), 'video': str(video), 'fps': 38}}))
    monkeypatch.setattr('empowerment_warp.training.sac.progress.subprocess.run', fake_run)
    result = create_progress_snapshot(run, default_recipe(environment, 'potential'),
        target_env_steps=50_000_000, checkpoint_step=50_003_968)
    assert len(calls) == 2
    assert Path(result['branch_video']).read_bytes() == b'branches'
    assert result['state'] == 'complete'
    assert (run / 'evaluation_checkpoints' / checkpoint.name / 'weights').read_text() == 'checkpoint'
    assert progress_wandb_row(result)['evaluation/episode_ball_disp_xy/mean'] == 1.25


def test_progress_commands_are_registered_in_real_cli(tmp_path):
    import subprocess
    from empowerment_warp.training.sac.progress import _commands
    for environment in ('ant', 'ant_soccer', 'builderbench_corner3'):
        for command in _commands(tmp_path / 'run', tmp_path / 'target_000000000000', environment):
            result = subprocess.run((*command, '--help'), capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert '--checkpoint' in result.stdout


def test_progress_output_is_allowed_without_overwriting_run(tmp_path):
    import pytest
    from empowerment_warp.cli import _output_directory
    run = tmp_path / 'run'
    output = run / 'progress' / 'target_000050000000' / 'evaluation'
    assert _output_directory(run, output) == (run, output)
    for unsafe in (run, tmp_path, run / 'checkpoints', run / 'evaluation'):
        with pytest.raises(ValueError, match='outside the source run'):
            _output_directory(run, unsafe)


@pytest.mark.parametrize("failed_command", ["evaluate", "branch-media"])
def test_snapshot_failure_stops_after_checkpoint_and_resume_ignores_prose(
    monkeypatch, tmp_path, failed_command,
):
    import dataclasses
    import subprocess
    import jax.numpy as jnp
    from empowerment_warp.checkpointing import CheckpointStore
    from empowerment_warp.training.sac import runner

    config = dataclasses.replace(
        default_recipe("ant", "potential"), run_dir=str(tmp_path / "run"),
        allow_reference_deviation=True, num_envs=1, rollout_steps=1, batch_size=1,
        total_env_steps=3, progress_eval_every_env_steps=1,
    )

    @dataclasses.dataclass
    class State:
        env_steps: object
        sac: object
        skill: object
        collector: object
        branch_simulator_steps: object = 0

        def replace(self, **kwargs):
            return dataclasses.replace(self, **kwargs)

    state = State(jnp.asarray(0), {"x": jnp.ones(1)}, {"x": jnp.ones(1)},
                  SimpleNamespace(env_state=None, coverage_grids=None))
    trainer = SimpleNamespace(
        branch_phase_groups=1,
        initialize=lambda key: state,
        run=lambda current: (current.replace(env_steps=current.env_steps + 1), None),
    )
    mirror = SimpleNamespace(log=lambda row: None, finish=lambda: None)
    monkeypatch.setattr(runner, "require_runtime", lambda config: None)
    monkeypatch.setattr(runner, "make_experiment", lambda config: SimpleNamespace(trainer=trainer, env=None))
    monkeypatch.setattr(runner.WandbMirror, "start", lambda *a, **k: mirror)
    monkeypatch.setattr(runner, "HoldoutCycleAccumulator", lambda **k: SimpleNamespace(add=lambda metrics: None))
    for name in ("coverage_track_specs", "coverage_floor_masks", "coverage_counts",
                 "summarize_holdout_event", "summarize_metrics", "_summarize_exploration"):
        monkeypatch.setattr(runner, name, lambda *a, **k: {})
    monkeypatch.setattr(runner, "_accumulate_exploration", lambda *a: None)
    monkeypatch.setattr(runner, "log_progress_snapshot", lambda *a: None)

    calls = []

    def snapshot(run_dir, config, *, target_env_steps, checkpoint_step):
        calls.append(target_env_steps)
        # This is the real checkpoint written by the host training loop.
        assert CheckpointStore(run_dir / "checkpoints").latest_step() == checkpoint_step
        if target_env_steps:
            raise subprocess.CalledProcessError(1, failed_command)
        return {}

    monkeypatch.setattr(runner, "create_progress_snapshot", snapshot)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        runner.train(config)
    assert exc.value.cmd == failed_command
    assert calls == [0, 1]
    assert CheckpointStore(config.resolved_run_dir / "checkpoints").latest_step() == 1
    assert not (config.resolved_run_dir / "train_summary.json").exists()

    # Resume must compare the scientific config, not explanatory provenance text.
    provenance = config.resolved_run_dir / "reference_manifest.json"
    provenance.write_text('{"note": "edited provenance prose"}\n')
    monkeypatch.setattr(runner, "create_progress_snapshot", lambda *a, **k: {})
    runner.train(config, resume=True, stop_after_segments=1)
    summary = json.loads((config.resolved_run_dir / "train_summary.json").read_text())
    assert summary["start_env_steps"] == 1
    assert summary["final_env_steps"] == 2
    assert json.loads(provenance.read_text()) == {"note": "edited provenance prose"}
    with pytest.raises(ValueError, match="resume config differs"):
        runner.train(dataclasses.replace(config, discount=.5), resume=True)

    # Completing training must evaluate the saved final checkpoint before success.
    final_targets = []
    def completed_snapshot(run_dir, config, *, target_env_steps, checkpoint_step):
        assert CheckpointStore(run_dir / "checkpoints").latest_step() == checkpoint_step
        final_targets.append(target_env_steps)
        return {}
    monkeypatch.setattr(runner, "create_progress_snapshot", completed_snapshot)
    runner.train(config, resume=True)
    assert 3 in final_targets
    summary = json.loads((config.resolved_run_dir / "train_summary.json").read_text())
    assert summary["final_env_steps"] == 3
