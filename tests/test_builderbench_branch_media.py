from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from empowerment_warp.media import builderbench
from empowerment_warp.media.builderbench import _prefix_report, evenly_spaced_skill_indices
from empowerment_warp.cli import main


def test_rendered_skill_subset_is_evenly_spaced() -> None:
    assert evenly_spaced_skill_indices(30, 4) == (0, 10, 19, 29)


def test_branch_prefix_report_checks_every_render_field() -> None:
    arrays = {
        name: np.zeros((5, 4, *shape), dtype=np.float32)
        for name, shape in {
            "qpos": (7,),
            "qvel": (6,),
            "effector_positions": (3,),
            "cube_positions": (3, 3),
            "mocap_pos": (3, 3),
            "mocap_quat": (3, 4),
        }.items()
    }
    arrays["cube_positions"][3, 3, 0, 0] = 1.0
    report = _prefix_report(arrays, activation_index=2)
    assert report["valid"]
    arrays["qpos"][2, 2, 0] = 1e-3
    report = _prefix_report(arrays, activation_index=2)
    assert not report["valid"]
    assert np.isclose(report["max_abs_delta_by_key"]["qpos"], 1e-3)


@pytest.mark.parametrize('saved', [
    {'potential_steps': 0},
    {'potential_steps': 100, 'skill_start_mode': 'marginal_branch', 'live_potential_only': False},
])
def test_media_defaults_to_saved_zero_prefix_and_stochastic_policy(
    monkeypatch, tmp_path: Path, saved
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(saved))
    captured = {}

    def fake_collect(run_dir, output, **kwargs):
        captured.update(kwargs)
        Path(output).write_bytes(b"trace")
        return Path(output)

    monkeypatch.setattr(builderbench, "collect_builderbench_branch_trace", fake_collect)
    monkeypatch.setattr(
        builderbench,
        "render_builderbench_branch_trace",
        lambda *args, **kwargs: {"metadata": {"policy_mode": "stochastic"}},
    )

    result = builderbench.create_builderbench_branch_media(
        run_dir, tmp_path / "media", upload_wandb=False
    )

    assert captured["prefix_steps"] == 0
    assert captured["policy_mode"] == "stochastic"
    assert captured["horizon_steps"] == 400
    assert result["prefix_steps"] == 0
    assert result["policy_mode"] == "stochastic"


def test_media_keeps_explicit_deterministic_positive_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"potential_steps": 0}))
    captured = {}

    def fake_collect(run_dir, output, **kwargs):
        captured.update(kwargs)
        Path(output).write_bytes(b"trace")
        return Path(output)

    monkeypatch.setattr(builderbench, "collect_builderbench_branch_trace", fake_collect)
    monkeypatch.setattr(
        builderbench,
        "render_builderbench_branch_trace",
        lambda *args, **kwargs: {"metadata": {"policy_mode": "deterministic"}},
    )

    result = builderbench.create_builderbench_branch_media(
        run_dir,
        tmp_path / "media",
        prefix_steps=100,
        policy_mode="deterministic",
        upload_wandb=False,
    )

    assert captured["prefix_steps"] == 100
    assert captured["policy_mode"] == "deterministic"
    assert result["prefix_steps"] == 100
    assert result["policy_mode"] == "deterministic"


def test_builderbench_media_cli_defaults_to_stochastic_and_saved_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run/config.json').write_text(json.dumps({'environment': 'builderbench_corner3'}))

    def fake_create(run_dir, output_dir, **kwargs):
        captured.update(kwargs)
        return {"policy_mode": kwargs["policy_mode"]}

    monkeypatch.setattr(builderbench, "create_builderbench_branch_media", fake_create)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "empowerment-warp",
            "branch-media",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "media"),
        ],
    )

    main()

    assert captured.get("prefix_steps") is None
    assert captured["policy_mode"] == "stochastic"
    assert captured['upload_wandb'] is False
