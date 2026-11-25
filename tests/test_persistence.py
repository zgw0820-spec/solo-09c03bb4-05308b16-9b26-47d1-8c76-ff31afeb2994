"""Durable-persistence and crash-recovery tests.

These tests exercise the FileStore backend directly and, more importantly, the
"restart" scenario: a fresh :class:`Engine` instance is pointed at the same
data directory, modelling a container that exited and came back with the same
volume mounted.  Every guarantee from the brief is checked against the
restored engine, not just the one that wrote the state.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from app import model
from app.engine import ConflictError, Engine, PersistenceError, ReplayError
from app.storage import FileStore, FORMAT_VERSION, StorageError

from tests.test_engine import SAMPLES, clean_stream, feed, frames  # noqa: F401


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


def _engine(data_dir: Path) -> Engine:
    return Engine(store=FileStore(data_dir))


def _camera_doc(data_dir: Path, camera: int) -> dict:
    return json.loads((data_dir / f"camera-{camera}.state.json").read_text("utf-8"))


def test_state_files_are_written_for_every_effective_change(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2])
    doc = _camera_doc(data_dir, 0)
    assert doc["format_version"] == FORMAT_VERSION
    assert sorted(int(f) for f in doc["scores"]) == [1, 2]
    assert sorted(int(f) for f in doc["frames"]) == [1, 2]
    assert (data_dir / "manifest.json").exists()
    manifest = json.loads((data_dir / "manifest.json").read_text("utf-8"))
    assert manifest["model"]["weight_digest"] == model.weight_digest()
    # Only this camera's file exists: camera 1 has never accepted a frame.
    assert not (data_dir / "camera-1.state.json").exists()


def test_restart_restores_scores_gap_provisional_state_and_continues(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6])
    feed(e, 1, [1, 2, 3, 4])
    before0 = e.snapshot(0)
    before1 = e.snapshot(1)

    # Simulate the container exiting: drop every in-memory object, reopen the
    # same volume with a brand-new engine.
    del e
    restored = _engine(data_dir)

    snap0 = restored.snapshot(0)
    assert [s["frame_no"] for s in snap0["scores"]] == [1, 2, 3, 4, 5, 6]
    assert snap0["gap"]["frame_no"] == 4
    assert snap0["state_digest"] == before0["state_digest"]
    assert snap0["last_output_frame"] == before0["last_output_frame"] == 6
    assert snap0["scores"][3]["estimated"] is True
    # Original and (not yet existing) revised score columns survive.
    assert all(s["revised_score"] is None for s in snap0["scores"])
    assert [s["score"] for s in snap0["scores"]] == [
        s["score"] for s in before0["scores"]
    ]
    snap1 = restored.snapshot(1)
    assert snap1["state_digest"] == before1["state_digest"]
    assert len(snap1["scores"]) == 4

    # Continue from frame 7 on the restored engine: no old frame is recomputed,
    # and the provisional recurrence lines up exactly with the old process.
    r = restored.ingest(0, 7, frames(0)[7])
    assert r["estimated"] is True
    # Filling the gap after restart yields the same corrected stream as a
    # gap-free reference run.
    result = restored.fill_gap(0, 4, frames(0)[4])
    assert result["status"] == "replayed"
    _, ref = clean_stream(0, limit=7)
    snap = restored.snapshot(0)
    for s in snap["scores"]:
        if s["frame_no"] >= 4:
            assert s["final_score"] == pytest.approx(ref[s["frame_no"] - 1], abs=1e-12)
    # Next frame uses the restored corrected recurrent state.
    nxt = restored.ingest(0, 8, frames(0)[8])
    assert nxt["estimated"] is False
    assert nxt["score"] == pytest.approx(clean_stream(0, limit=8)[1][-1], abs=1e-12)


def test_crash_before_fill_commit_keeps_pending_fill_and_resumes(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7])
    # Start the fill (its pending record is persisted immediately) but never
    # let it commit: kill the process model right after start.
    cam = e._camera(0)

    def boom(camera: int, frame_no: int) -> None:
        raise RuntimeError("simulated process death mid-replay")

    e.replay_step_hook = boom
    with pytest.raises(ReplayError):
        e.fill_gap(0, 4, frames(0)[4])
    # Frames that arrived "meanwhile" (here: none yet) plus the pending fill
    # are on disk.  Restart with a fresh engine.
    del e
    restored = _engine(data_dir)
    snap = restored.snapshot(0)
    assert snap["gap"]["frame_no"] == 4
    assert snap["pending_fill"]["frame_no"] == 4
    assert snap["pending_fill"]["attempt_failed"] is True
    assert len(snap["scores"]) == 7

    # The gap can be re-submitted; replay resumes from the pre-gap state and
    # catches up, producing exactly the gap-free reference scores.
    result = restored.fill_gap(0, 4, frames(0)[4])
    assert result["replay"]["resumed"] is True
    assert result["replay"]["to"] == 7
    _, ref = clean_stream(0, limit=7)
    for s in restored.snapshot(0)["scores"]:
        if s["frame_no"] >= 4:
            assert s["final_score"] == pytest.approx(ref[s["frame_no"] - 1], abs=1e-12)
    assert restored.snapshot(0)["pending_fill"] is None
    # The restored corrected state is itself durable for the next restart.
    doc = _camera_doc(data_dir, 0)
    assert doc["gap"] is None and doc["pending_fill"] is None


def test_frames_arriving_while_fill_is_pending_survive_restart(data_dir: Path) -> None:
    """Crash window includes new frames: they stay and the resumed fill chases them."""
    e = _engine(data_dir)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6])

    # Abort the worker mid-replay without running the failure handler, the way
    # SIGKILL / an OOM exit would: the pending_fill write from fill start and
    # every frame write are already on disk, but no commit ever happened.
    def hard_stop(camera: int, frame_no: int) -> None:
        if frame_no >= 5:
            raise KeyboardInterrupt("simulated SIGKILL mid-replay")

    e.replay_step_hook = hard_stop
    with pytest.raises(KeyboardInterrupt):
        e.fill_gap(0, 4, frames(0)[4], pace_ms=10)
    # Two more frames had been accepted from the provisional state before the
    # "kill"; they were each persisted on ingest.
    assert e.snapshot(0)["last_output_frame"] == 6
    del e

    restored = _engine(data_dir)
    snap = restored.snapshot(0)
    assert snap["gap"]["frame_no"] == 4
    assert snap["pending_fill"]["frame_no"] == 4
    assert [s["frame_no"] for s in snap["scores"]] == [1, 2, 3, 4, 5, 6]
    assert all(s["estimated"] for s in snap["scores"][3:])

    # New frames keep flowing after the restart, still provisional ...
    r7 = restored.ingest(0, 7, frames(0)[7])
    assert r7["estimated"] is True
    # ... then the gap is re-submitted and the resumed replay catches up to
    # the newest published frame, matching a gap-free reference.
    result = restored.fill_gap(0, 4, frames(0)[4])
    assert result["replay"]["resumed"] is True
    assert result["replay"]["to"] == 7
    _, ref = clean_stream(0, limit=7)
    for s in restored.snapshot(0)["scores"]:
        if s["frame_no"] >= 4:
            assert s["final_score"] == pytest.approx(ref[s["frame_no"] - 1], abs=1e-12)
    nxt = restored.ingest(0, 8, frames(0)[8])
    assert nxt["estimated"] is False
    assert nxt["score"] == pytest.approx(clean_stream(0, limit=8)[1][-1], abs=1e-12)


def test_committed_fill_restores_revised_scores_and_recurrent_state(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6])
    commit = e.fill_gap(0, 4, frames(0)[4])
    assert commit["durable"] is True
    revised_before = {
        s["frame_no"]: (s["score"], s["revised_score"]) for s in e.snapshot(0)["scores"]
    }
    del e
    restored = _engine(data_dir)
    snap = restored.snapshot(0)
    assert snap["gap"] is None
    for s in snap["scores"]:
        old_score, old_revised = revised_before[s["frame_no"]]
        assert s["score"] == pytest.approx(old_score, abs=0)
        if old_revised is not None:
            assert s["revised_score"] == pytest.approx(old_revised, abs=1e-12)
            assert s["revision"] == 1
    # Recurrent state for the next frame is the corrected one.
    _, ref_scores = clean_stream(0, limit=7)
    nxt = restored.ingest(0, 7, frames(0)[7])
    assert nxt["estimated"] is False
    assert nxt["score"] == pytest.approx(ref_scores[6], abs=1e-12)


def test_write_failure_takes_no_effect_and_request_is_retryable(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(data_dir)
    e = Engine(store=store)
    feed(e, 0, [1, 2])
    before = e.snapshot(0)

    def broken_save(camera: int, doc: dict) -> None:  # noqa: ANN001
        raise StorageError("simulated disk fault", path=str(data_dir))

    monkeypatch.setattr(store, "save_camera", broken_save)
    with pytest.raises(PersistenceError) as info:
        e.ingest(0, 3, frames(0)[3])
    assert info.value.retryable is True
    assert info.value.payload()["state_preserved"] is True

    # In-memory state is exactly the last durably-effective one ...
    after = e.snapshot(0)
    assert after["state_digest"] == before["state_digest"]
    assert [s["frame_no"] for s in after["scores"]] == [1, 2]
    # ... and the disk document agrees, so a concurrent reader/restart sees
    # nothing partial either.
    doc = _camera_doc(data_dir, 0)
    assert sorted(int(f) for f in doc["scores"]) == [1, 2]

    monkeypatch.undo()
    r = e.ingest(0, 3, frames(0)[3])  # exact retry now succeeds
    assert r["frame_no"] == 3 and r["durable"] is True
    assert [s["frame_no"] for s in e.snapshot(0)["scores"]] == [1, 2, 3]


def test_fill_commit_write_failure_keeps_gap_and_pending_fill(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(data_dir)
    e = Engine(store=store)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5])
    before = e.snapshot(0)

    real_save = store.save_camera
    calls = {"n": 0}

    def flaky_save(camera: int, doc: dict) -> None:  # noqa: ANN001
        calls["n"] += 1
        if doc.get("pending_fill") is None and doc.get("gap") is None:
            # Only the commit document (gap cleared) fails; the pending-write
            # at fill start must still succeed.
            raise StorageError("simulated disk fault on commit")
        real_save(camera, doc)

    monkeypatch.setattr(store, "save_camera", flaky_save)
    with pytest.raises(PersistenceError):
        e.fill_gap(0, 4, frames(0)[4])
    snap = e.snapshot(0)
    assert snap["gap"]["frame_no"] == 4
    assert snap["pending_fill"]["frame_no"] == 4
    assert snap["state_digest"] == before["state_digest"]
    assert all(s["revision"] == 0 for s in snap["scores"])
    monkeypatch.undo()
    result = e.fill_gap(0, 4, frames(0)[4])
    assert result["status"] == "replayed" and result["replay"]["resumed"] is True


def test_reset_is_durable_and_clears_pending_fill(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2])
    e.declare_missing(0, 3)
    assert (data_dir / "camera-0.state.json").exists()
    out = e.reset()
    assert out["durable"] is True
    assert not (data_dir / "camera-0.state.json").exists()
    del e
    restored = _engine(data_dir)
    snap = restored.snapshot(0)
    assert snap["scores"] == [] and snap["gap"] is None and snap["last_frame"] == 0
    # Fresh ingest after restart starts the recurrence from zero.
    r = restored.ingest(0, 1, frames(0)[1])
    assert r["estimated"] is False


def test_reset_write_failure_leaves_everything_intact(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(data_dir)
    e = Engine(store=store)
    feed(e, 0, [1, 2, 3])
    before = e.snapshot(0)
    monkeypatch.setattr(
        store,
        "clear_camera",
        lambda camera: (_ for _ in ()).throw(StorageError("disk busy")),
    )
    with pytest.raises(PersistenceError):
        e.reset()
    after = e.snapshot(0)
    assert after["state_digest"] == before["state_digest"]
    assert [s["frame_no"] for s in after["scores"]] == [1, 2, 3]
    assert (data_dir / "camera-0.state.json").exists()


def test_atomic_write_leaves_no_tmp_files_and_survives_partial_state(
    data_dir: Path,
) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2, 3, 4])
    leftovers = list(data_dir.glob(".*.tmp"))
    assert leftovers == []
    # A corrupted temp/unknown file must not poison loading of real documents.
    (data_dir / ".camera-0.state.json.deadbeef.tmp").write_text("{not json", "utf-8")
    restored = _engine(data_dir)
    assert len(restored.snapshot(0)["scores"]) == 4


def test_manifest_rejects_weight_digest_mismatch(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1])
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["model"]["weight_digest"] = "deadbeefdeadbeef"
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(StorageError, match="weight"):
        _engine(data_dir)


def test_journal_records_effective_events_only(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1])
    with pytest.raises(ConflictError):
        e.ingest(0, 1, frames(0)[1])  # duplicate rejected: must not be journaled
    e.declare_missing(0, 2)
    e.ingest(0, 3, frames(0)[3])
    e.fill_gap(0, 2, frames(0)[2])
    lines = [
        json.loads(line)
        for line in (data_dir / "journal.jsonl").read_text("utf-8").splitlines()
    ]
    events = [line["event"] for line in lines]
    assert events.count("ingest") == 2  # frames 1 and 3, not the rejected dup
    assert "gap_opened" in events and "gap_filled" in events


def test_events_are_persisted_in_the_same_write_and_restored(data_dir: Path) -> None:
    e = _engine(data_dir)
    e.ingest(0, 1, frames(0)[1])
    doc = _camera_doc(data_dir, 0)
    assert [ev["event"] for ev in doc["events"]] == ["ingest"]
    e.declare_missing(0, 2)
    del e
    restored = _engine(data_dir)
    names = [ev["event"] for ev in restored.snapshot(0)["events"]]
    assert names == ["ingest", "gap_opened"]


def test_reset_during_pending_replay_cancels_and_is_restart_clean(data_dir: Path) -> None:
    import time

    e = _engine(data_dir)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6])
    outcome: list[BaseException] = []

    def run_fill() -> None:
        try:
            e.fill_gap(0, 4, frames(0)[4], pace_ms=60)
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    worker = threading.Thread(target=run_fill)
    worker.start()
    time.sleep(0.05)
    reset = e.reset()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert reset["cancelled_replays"] == [{"camera": 0, "frame_no": 4}]
    assert outcome and isinstance(outcome[0], ReplayError)
    assert not (data_dir / "camera-0.state.json").exists()
    restored = _engine(data_dir)
    snap = restored.snapshot(0)
    assert snap["scores"] == [] and snap["gap"] is None and snap["pending_fill"] is None


def test_two_cameras_write_independent_documents_and_both_restore(data_dir: Path) -> None:
    e = _engine(data_dir)
    feed(e, 0, [1, 2])
    feed(e, 1, [1, 2, 3])
    assert (data_dir / "camera-0.state.json").exists()
    assert (data_dir / "camera-1.state.json").exists()
    del e
    restored = _engine(data_dir)
    assert len(restored.snapshot(0)["scores"]) == 2
    assert len(restored.snapshot(1)["scores"]) == 3
    # Wiping one camera's file only affects that camera.
    (data_dir / "camera-1.state.json").unlink()
    again = Engine(store=FileStore(data_dir))
    assert len(again.snapshot(0)["scores"]) == 2
    assert again.snapshot(1)["scores"] == []


def test_rejected_request_writes_nothing(data_dir: Path) -> None:
    e = _engine(data_dir)
    e.ingest(0, 1, frames(0)[1])
    mtime = (data_dir / "camera-0.state.json").stat().st_mtime_ns
    with pytest.raises(ConflictError):
        e.ingest(0, 1, frames(0)[1])  # duplicate
    with pytest.raises(ConflictError):
        e.ingest(0, 4, frames(0)[4])  # two-frame jump
    assert (data_dir / "camera-0.state.json").stat().st_mtime_ns == mtime


def test_memory_store_backend_keeps_engine_working_without_disk() -> None:
    e = Engine()  # default MemoryStore
    assert e.summary()["storage"]["durable"] is False
    feed(e, 0, [1, 2])
    e.declare_missing(0, 3)
    result = e.fill_gap(0, 3, frames(0)[3])
    assert result["durable"] is False
    assert e.snapshot(0)["gap"] is None
