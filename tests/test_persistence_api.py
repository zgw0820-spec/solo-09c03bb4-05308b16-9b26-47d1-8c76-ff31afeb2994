"""HTTP-level crash/recovery tests with a real file-backed engine.

A second :class:`fastapi.testclient.TestClient` is built around a fresh
``Engine`` on the same data directory to model the container restart, while
the volume (directory) stays in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.engine import Engine
from app.storage import FileStore
from tests.test_api import CAM0, CAM1


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "volume"
    monkeypatch.setenv("QUALITY_DATA_DIR", str(data_dir))
    from app import api as api_module

    engine = Engine(store=FileStore(data_dir))
    monkeypatch.setattr(api_module, "engine", engine)
    return TestClient(app), data_dir


def _restarted_client(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app import api as api_module

    monkeypatch.setattr(api_module, "engine", Engine(store=FileStore(data_dir)))
    return TestClient(app)


def test_health_reports_durable_storage(client) -> None:
    c, data_dir = client
    info = c.get("/api/health").json()
    assert info["storage"] == {"mode": "file", "durable": True, "data_dir": str(data_dir)}


def test_full_gap_lifecycle_survives_http_restart(client, monkeypatch: pytest.MonkeyPatch) -> None:
    c, data_dir = client
    for n in (1, 2, 3):
        c.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
    c.post("/api/cameras/0/missing", json={"frame_no": 4})
    for n in (5, 6):
        r = c.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
        assert r.json()["estimated"] is True and r.json()["durable"] is True
    # Camera B has data too: both documents must come back independently.
    c.post("/api/cameras/1/frames", json={"frame_no": 1, "features": CAM1[1]})

    restarted = _restarted_client(data_dir, monkeypatch)
    snap = restarted.get("/api/cameras/0/state").json()
    assert snap["gap"]["frame_no"] == 4
    assert [s["frame_no"] for s in snap["scores"]] == [1, 2, 3, 4, 5, 6]
    assert snap["scores"][3]["estimated"] is True
    assert restarted.get("/api/cameras/1/state").json()["scores"][0]["frame_no"] == 1

    fill = restarted.post(
        "/api/cameras/0/fill", json={"frame_no": 4, "features": CAM0[4]}
    )
    assert fill.status_code == 200
    body = fill.json()
    assert body["replay"]["resumed"] is False
    assert body["replay"]["to"] == 6 and body["durable"] is True

    # Another restart: revised scores and the corrected recurrence persist.
    again = _restarted_client(data_dir, monkeypatch)
    snap = again.get("/api/cameras/0/state").json()
    assert snap["gap"] is None and snap["pending_fill"] is None
    assert snap["scores"][3]["revised_score"] is not None
    assert snap["scores"][3]["score"] == fill.json()["gap_score"]["published_score"]
    nxt = again.post(
        "/api/cameras/0/frames", json={"frame_no": 7, "features": CAM0[7]}
    ).json()
    assert nxt["estimated"] is False


def test_pending_fill_resumes_over_http(client, monkeypatch: pytest.MonkeyPatch) -> None:
    c, data_dir = client
    for n in (1, 2, 3):
        c.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
    c.post("/api/cameras/0/missing", json={"frame_no": 4})
    c.post("/api/cameras/0/frames", json={"frame_no": 5, "features": CAM0[5]})
    # A failed fill is durable: the pending marker survives the restart.
    bad = c.post(
        "/api/cameras/0/fill",
        json={"frame_no": 4, "features": CAM0[4], "simulate_failure": True},
    )
    assert bad.status_code == 422 and bad.json()["retryable"] is True

    restarted = _restarted_client(data_dir, monkeypatch)
    snap = restarted.get("/api/cameras/0/state").json()
    assert snap["gap"]["frame_no"] == 4
    assert snap["pending_fill"]["attempt_failed"] is True
    # Resubmitting the gap frame resumes the fill instead of erroring.
    ok = restarted.post(
        "/api/cameras/0/fill", json={"frame_no": 4, "features": CAM0[4]}
    )
    assert ok.status_code == 200 and ok.json()["replay"]["resumed"] is True
    assert restarted.get("/api/cameras/0/state").json()["gap"] is None


def test_reset_survives_restart(client, monkeypatch: pytest.MonkeyPatch) -> None:
    c, data_dir = client
    c.post("/api/cameras/0/frames", json={"frame_no": 1, "features": CAM0[1]})
    assert (data_dir / "camera-0.state.json").exists()
    r = c.post("/api/reset", json={"confirm": True})
    assert r.status_code == 200 and r.json()["durable"] is True
    assert not (data_dir / "camera-0.state.json").exists()
    restarted = _restarted_client(data_dir, monkeypatch)
    snap = restarted.get("/api/cameras/0/state").json()
    assert snap["scores"] == [] and snap["last_frame"] == 0


def test_persistence_failure_returns_retryable_503(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, data_dir = client
    c.post("/api/cameras/0/frames", json={"frame_no": 1, "features": CAM0[1]})
    store = c.app  # noqa: F841 -- clarity only
    from app import api as api_module

    def broken(camera: int, doc: dict) -> None:  # noqa: ANN001
        from app.storage import StorageError

        raise StorageError("volume read-only")

    monkeypatch.setattr(api_module.engine._store, "save_camera", broken)
    resp = c.post("/api/cameras/0/frames", json={"frame_no": 2, "features": CAM0[2]})
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "persistence_failed"
    assert body["retryable"] is True and body["state_preserved"] is True
    # The failed frame really did not take effect anywhere.
    snap = c.get("/api/cameras/0/state").json()
    assert [s["frame_no"] for s in snap["scores"]] == [1]
    monkeypatch.undo()
    ok = c.post("/api/cameras/0/frames", json={"frame_no": 2, "features": CAM0[2]})
    assert ok.status_code == 200


def test_state_files_are_valid_json_documents(client) -> None:
    c, data_dir = client
    c.post("/api/cameras/0/frames", json={"frame_no": 1, "features": CAM0[1]})
    doc = json.loads((data_dir / "camera-0.state.json").read_text("utf-8"))
    assert doc["camera"] == 0 and doc["last_frame"] == 1
    assert doc["scores"]["1"]["feature_source"] == "submitted"
