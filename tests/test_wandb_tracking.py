from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from empowerment_warp.tracking import WandbMirror, sync_wandb_run


class _FakeRun:
    entity = "entity"
    project = "project"
    id = "run-id"
    name = "run-name"
    url = "https://wandb.ai/entity/project/runs/run-id"

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []
        self.metric_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.finished = False

    def define_metric(self, *args: object, **kwargs: object) -> None:
        self.metric_calls.append((args, kwargs))

    def log(self, row: dict[str, float]) -> None:
        self.rows.append(row)

    def finish(self) -> None:
        self.finished = True


def _fake_wandb(monkeypatch) -> _FakeRun:
    run = _FakeRun()
    module = SimpleNamespace(
        init=lambda **kwargs: run,
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr("empowerment_warp.tracking.import_module", lambda name: module)
    return run


def test_mirror_writes_identity_and_uses_env_steps(monkeypatch, tmp_path: Path) -> None:
    run = _fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")

    mirror = WandbMirror.start(
        tmp_path,
        {"environment": "builderbench_corner3", "algorithm": "potential"},
    )
    row = {"segment": 10, "env_steps": 327680.0, "skill/accuracy": 0.1}
    mirror.log(row)
    mirror.finish()

    identity = json.loads((tmp_path / "wandb_run.json").read_text())
    status = json.loads((tmp_path / "wandb_status.json").read_text())
    assert identity["url"] == run.url
    assert (tmp_path / "wandb_links.txt").read_text().strip() == run.url
    assert identity["wandb_run_id"] == run.id
    assert identity["wandb_url"] == run.url
    assert (("env_steps",), {}) in run.metric_calls
    assert (("*",), {"step_metric": "env_steps"}) in run.metric_calls
    assert run.rows == [row]
    assert status == {
        "last_segment": 10,
        "mode": "online",
        "run_id": run.id,
        "state": "finished",
        "url": run.url,
    }


def test_timed_run_uses_training_seconds_as_wandb_axis(monkeypatch, tmp_path: Path) -> None:
    run = _fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")

    WandbMirror.start(
        tmp_path,
        {
            "environment": "ant_soccer",
            "algorithm": "potential",
            "max_train_seconds": 3_000.0,
        },
    )

    assert (("train/wall_seconds",), {}) in run.metric_calls
    assert (("*",), {"step_metric": "train/wall_seconds"}) in run.metric_calls
    assert (("*",), {"step_metric": "env_steps"}) not in run.metric_calls


def test_active_mirror_logs_local_video(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun()
    video_calls: list[tuple[str, int, str]] = []
    module = SimpleNamespace(
        init=lambda **kwargs: run,
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        Video=lambda path, fps, format: video_calls.append((path, fps, format)) or "video",
    )
    monkeypatch.setattr("empowerment_warp.tracking.import_module", lambda name: module)
    monkeypatch.setenv("WANDB_MODE", "online")
    video = tmp_path / "branch.mp4"
    video.write_bytes(b"video")

    mirror = WandbMirror.start(
        tmp_path,
        {"environment": "ant_soccer", "algorithm": "potential"},
    )
    uploaded = mirror.log_video(
        {"env_steps": 50_003_968.0},
        key="media/trajectory_ant_overlay_video",
        path=video,
        fps=38,
    )

    assert uploaded
    assert video_calls == [(str(video), 38, "mp4")]
    assert run.rows == [
        {
            "env_steps": 50_003_968.0,
            "media/trajectory_ant_overlay_video": "video",
        }
    ]


def test_sync_resumes_after_last_local_segment(monkeypatch, tmp_path: Path) -> None:
    run = _fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")
    (tmp_path / "config.json").write_text(
        json.dumps({"environment": "builderbench_corner3", "algorithm": "potential"})
    )
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"segment": segment, "env_steps": float(segment * 100)})
            for segment in (1, 2, 3)
        )
        + "\n"
    )
    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"id": "run-id", "url": run.url})
    )
    (tmp_path / "wandb_status.json").write_text(
        json.dumps({"state": "finished", "last_segment": 2})
    )

    sync_wandb_run(tmp_path)

    assert [row["segment"] for row in run.rows] == [3]


def test_disabled_mode_never_imports_wandb(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(
        "empowerment_warp.tracking.import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("wandb should not import")),
    )

    mirror = WandbMirror.start(tmp_path, {})

    assert mirror.run is None
    assert json.loads((tmp_path / "wandb_status.json").read_text())["state"] == "disabled"
    assert (tmp_path / "wandb_run.json").exists()
    assert json.loads((tmp_path / "wandb_run.json").read_text())["entity"] is None
    assert (tmp_path / "wandb_links.txt").read_text().strip() == ""


def test_init_failure_preserves_retryable_identity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "configured-entity")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_RUN_GROUP", "standing-builderbench")
    monkeypatch.setattr(
        "empowerment_warp.tracking.import_module",
        lambda name: SimpleNamespace(
            init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("service unavailable")),
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    mirror = WandbMirror.start(
        tmp_path,
        {"environment": "builderbench_corner3", "algorithm": "potential"},
    )

    identity = json.loads((tmp_path / "wandb_run.json").read_text())
    status = json.loads((tmp_path / "wandb_status.json").read_text())
    assert mirror.run is None
    assert status["state"] == "failed"
    assert status["retryable"] is True
    assert status["run_id"] == identity["id"]
    assert status["url"] == identity["url"]
    assert (tmp_path / "wandb_links.txt").read_text().strip() == identity["url"]
