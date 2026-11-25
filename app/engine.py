"""Per-camera ingestion engine with one-frame gap recovery and durable state.

Rules implemented here
----------------------
* Every accepted frame publishes a score immediately.  Published scores are
  immutable: a later replay never edits them, it only attaches a revised value
  next to the original.
* Each camera may have at most one outstanding missing frame ("gap").  While it
  is missing, the next frame is advanced with a zero-vector substitution and its
  score is flagged ``estimated``; following frames continue from that
  provisional state and are flagged estimated as well.
* Both cameras keep ingesting while one camera is replaying.  Frames that
  arrive on the replaying camera during replay are also scored from the
  provisional state and flagged estimated.
* When the original frame is supplied, replay starts from the state *before*
  the gap and walks the stored frames in frame order.  The corrected state is
  only committed once replay has caught up with the newest frame that already
  has a published score (a target that may grow while replay runs).  Only then
  does subsequent inference continue from the corrected state.
* If replay fails, nothing published is mutated: the provisional state, the
  open gap and every published score survive, and the fill can be retried.
* One lock per camera serialises ingest, missing-frame declaration and the
  short bookkeeping sections of replay.  Model arithmetic during replay runs
  outside the lock so the camera can keep accepting frames.

Durability
----------
Every state-changing request (frame ingest, missing-frame declaration, fill
commit, reset) is first serialised into a full per-camera document and handed
to the :mod:`app.storage` backend.  The in-memory state is only mutated *after*
the durable write succeeds, so a write failure means "nothing took effect":
the client gets a retryable 503 and every previously effective result and
state stays in place.  An in-flight fill is persisted as ``pending_fill`` when
it starts; if the process exits before the commit, the gap, the provisional
state and every frame received meanwhile are restored on restart and the same
fill call resumes the replay.  A committed fill restores both the revised
scores and the recurrent state used by the next frame.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from . import model
from .storage import FORMAT_VERSION, MemoryStore, StorageError, Store, default_manifest

MAX_FRAMES_PER_CAMERA = 10_000
DEFAULT_CAMERAS = (0, 1)
SOURCE_SUBMITTED = "submitted"
SOURCE_ZERO = "zero_substitution"
REVISION_EPS = 1e-12
MAX_PACE_MS = 1000.0
EVENT_LIMIT = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EngineError(Exception):
    """Base error carrying an HTTP status and a machine-readable payload."""

    status_code = 400
    code = "engine_error"

    def __init__(self, message: str, *, retryable: bool = False, **extra: Any) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        body.update(self.extra)
        return body


class ValidationError(EngineError):
    status_code = 422
    code = "invalid_frame"


class ConflictError(EngineError):
    status_code = 409
    code = "conflict"


class ReplayError(EngineError):
    status_code = 422
    code = "replay_failed"


class PersistenceError(EngineError):
    """Durable write failed; the in-memory state was not mutated."""

    status_code = 503
    code = "persistence_failed"


@dataclass
class ScoreRecord:
    frame_no: int
    score: float
    estimated: bool
    feature_source: str
    published_at: str
    state_after: list[float]
    revision: int = 0
    revised_score: float | None = None
    revised_estimated: bool | None = None
    revised_at: str | None = None
    delta: float | None = None
    applied_to_stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_no": self.frame_no,
            "score": self.score,
            "estimated": self.estimated,
            "feature_source": self.feature_source,
            "published_at": self.published_at,
            "state_after": self.state_after,
            "revision": self.revision,
            "revised_score": self.revised_score,
            "revised_estimated": self.revised_estimated,
            "revised_at": self.revised_at,
            "delta": self.delta,
            "applied_to_stream": self.applied_to_stream,
            "final_score": self.revised_score if self.revised_score is not None else self.score,
            "final_estimated": (
                self.revised_estimated if self.revised_estimated is not None else self.estimated
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreRecord":
        return cls(
            frame_no=int(data["frame_no"]),
            score=float(data["score"]),
            estimated=bool(data["estimated"]),
            feature_source=str(data["feature_source"]),
            published_at=str(data["published_at"]),
            state_after=[float(v) for v in data["state_after"]],
            revision=int(data.get("revision", 0)),
            revised_score=data.get("revised_score"),
            revised_estimated=data.get("revised_estimated"),
            revised_at=data.get("revised_at"),
            delta=data.get("delta"),
            applied_to_stream=bool(data.get("applied_to_stream", False)),
        )


@dataclass
class FrameRecord:
    frame_no: int
    features: list[float]
    feature_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_no": self.frame_no,
            "features": list(self.features),
            "feature_source": self.feature_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_no=int(data["frame_no"]),
            features=[float(v) for v in data["features"]],
            feature_source=str(data["feature_source"]),
        )


@dataclass
class Gap:
    frame_no: int
    state_before: list[float]
    opened_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_no": self.frame_no,
            "state_before": list(self.state_before),
            "opened_at": self.opened_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Gap":
        return cls(
            frame_no=int(data["frame_no"]),
            state_before=[float(v) for v in data["state_before"]],
            opened_at=str(data["opened_at"]),
        )


@dataclass
class ReplayJob:
    frame_no: int
    features: list[float]
    state_before: np.ndarray
    state: np.ndarray
    cursor: int
    scores: dict[int, float] = field(default_factory=dict)
    states: dict[int, np.ndarray] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    target_at_start: int = 0
    simulate_failure: bool = False
    cancelled: bool = False
    attempt_failed: bool = False
    resumed: bool = False

    def to_pending(self) -> dict[str, Any]:
        """Persist the fill before/while it is uncommitted.

        Only the inputs and the pre-gap state are needed: the replay is a pure
        deterministic function of the fixed weights and the stored frames, so
        progress is cheap to rebuild on resume instead of being journaled.
        """
        return {
            "frame_no": self.frame_no,
            "features": list(self.features),
            "state_before": [float(v) for v in self.state_before],
            "started_at": self.started_at,
            "target_at_start": self.target_at_start,
            "attempt_failed": self.attempt_failed,
        }

    @classmethod
    def from_pending(cls, data: dict[str, Any]) -> "ReplayJob":
        before = np.asarray(data["state_before"], dtype=np.float64)
        return cls(
            frame_no=int(data["frame_no"]),
            features=[float(v) for v in data["features"]],
            state_before=before.copy(),
            state=before.copy(),
            cursor=int(data["frame_no"]),
            started_at=str(data.get("started_at", _now())),
            target_at_start=int(data.get("target_at_start", data["frame_no"])),
            attempt_failed=bool(data.get("attempt_failed", False)),
            resumed=True,
        )


@dataclass
class CameraState:
    camera: int
    state: np.ndarray = field(default_factory=model.initial_state)
    last_frame: int = 0
    last_output_frame: int = 0
    gap: Gap | None = None
    replay: ReplayJob | None = None
    pending_fill: dict[str, Any] | None = None
    scores: dict[int, ScoreRecord] = field(default_factory=dict)
    frames: dict[int, FrameRecord] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class Engine:
    def __init__(
        self,
        cameras: tuple[int, ...] = DEFAULT_CAMERAS,
        store: Store | None = None,
    ) -> None:
        self._cameras_cfg = tuple(cameras)
        self._store: Store = store if store is not None else MemoryStore()
        self._states = {c: CameraState(camera=c) for c in self._cameras_cfg}
        self._registry_lock = threading.Lock()
        # Observability/test hook, called with (camera, frame_no) before every
        # replayed frame.  Raising from it simulates a replay failure.
        self.replay_step_hook: Callable[[int, int], None] | None = None
        self._bootstrap()

    # ------------------------------------------------------------- bootstrap
    def _bootstrap(self) -> None:
        """Create or verify the manifest and restore any durable state."""
        if self._store.durable:
            manifest = self._store.load_manifest()
            if manifest is None:
                self._store.save_manifest(default_manifest(self._cameras_cfg))
            else:
                self._verify_manifest(manifest)
        self._restore()

    def _verify_manifest(self, manifest: dict[str, Any]) -> None:
        if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
            raise StorageError(
                f"unsupported state format {manifest.get('format_version')}; "
                f"this build expects {FORMAT_VERSION}"
            )
        stored = manifest.get("model", {}).get("weight_digest")
        current = model.weight_digest()
        if stored is not None and stored != current:
            raise StorageError(
                "model weights changed since the data was written: "
                f"stored digest {stored} != current {current}; refusing to mix states"
            )
        stored_cameras = tuple(sorted(int(c) for c in manifest.get("cameras", [])))
        if stored_cameras and stored_cameras != tuple(sorted(self._cameras_cfg)):
            raise StorageError(
                f"camera set changed: stored {list(stored_cameras)} "
                f"!= configured {list(sorted(self._cameras_cfg))}"
            )

    def _restore(self) -> None:
        docs = self._store.load_cameras()
        for camera, cam in self._states.items():
            doc = docs.get(camera)
            if doc is None:
                continue
            cam.state = np.asarray(doc["state"], dtype=np.float64)
            cam.last_frame = int(doc["last_frame"])
            cam.last_output_frame = int(doc["last_output_frame"])
            cam.gap = Gap.from_dict(doc["gap"]) if doc.get("gap") else None
            cam.scores = {
                int(f): ScoreRecord.from_dict(s) for f, s in doc.get("scores", {}).items()
            }
            cam.frames = {
                int(f): FrameRecord.from_dict(fr) for f, fr in doc.get("frames", {}).items()
            }
            cam.events = list(doc.get("events", []))[-EVENT_LIMIT:]
            pending = doc.get("pending_fill")
            if pending is not None:
                # The process died (or the fill failed) before a commit.  The
                # gap and provisional state above are intact; keep the fill
                # inputs so the same call resumes instead of starting over.
                cam.pending_fill = pending
                if cam.gap is None or cam.gap.frame_no != int(pending["frame_no"]):
                    raise StorageError(
                        f"camera {camera}: pending fill for frame {pending.get('frame_no')} "
                        "but no matching open gap; state file is inconsistent"
                    )

    # ------------------------------------------------------------------ helpers
    def _camera(self, camera: int) -> CameraState:
        with self._registry_lock:
            cam = self._states.get(camera)
        if cam is None:
            raise ValidationError(
                f"unknown camera {camera}; configured cameras: {sorted(self._states)}"
            )
        return cam

    @staticmethod
    def _validate_features(features: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if arr.shape != (model.FEATURES,):
            raise ValidationError(
                f"feature vector must contain exactly {model.FEATURES} numbers, got {arr.size}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValidationError("feature vector contains NaN or infinite values")
        if np.max(np.abs(arr)) > 1e6:
            raise ValidationError("feature values must be within +/-1e6")
        return arr

    @staticmethod
    def _check_frame_no(frame_no: int) -> None:
        if frame_no < 1:
            raise ValidationError("frame_no must be >= 1")

    @staticmethod
    def _check_pace(pace_ms: float | None) -> float:
        if pace_ms is None:
            return max(0.0, float(os.environ.get("QUALITY_REPLAY_PACE_MS", "0") or 0))
        if not (0.0 <= pace_ms <= MAX_PACE_MS):
            raise ValidationError(f"pace_ms must be between 0 and {MAX_PACE_MS:g}")
        return float(pace_ms)

    def _log(self, cam: CameraState, event: str, **payload: Any) -> None:
        entry = {"at": _now(), "camera": cam.camera, "event": event, **payload}
        cam.events.append(entry)
        if len(cam.events) > EVENT_LIMIT:
            del cam.events[: len(cam.events) - EVENT_LIMIT]

    def _journal(self, cam: CameraState, event: str, **payload: Any) -> None:
        """Best-effort audit append; the state document is the source of truth."""
        if not self._store.durable:
            return
        try:
            self._store.append_journal(
                {"at": _now(), "camera": cam.camera, "event": event, **payload}
            )
        except StorageError:
            # The state write already succeeded; a missing audit line must not
            # roll the effective result back or fail the client request.
            pass

    @staticmethod
    def _serialize(cam: CameraState) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "camera": cam.camera,
            "saved_at": _now(),
            "state": [round(float(v), 9) for v in cam.state],
            "last_frame": cam.last_frame,
            "last_output_frame": cam.last_output_frame,
            "gap": None if cam.gap is None else cam.gap.to_dict(),
            "pending_fill": cam.pending_fill,
            "scores": {str(f): rec.to_dict() for f, rec in sorted(cam.scores.items())},
            "frames": {str(f): rec.to_dict() for f, rec in sorted(cam.frames.items())},
            "events": list(cam.events)[-EVENT_LIMIT:],
        }

    def _emit(
        self,
        cam: CameraState,
        frame_no: int,
        features: list[float],
        source: str,
        state_after: np.ndarray,
    ) -> ScoreRecord:
        record = ScoreRecord(
            frame_no=frame_no,
            score=model.score(state_after),
            estimated=(source == SOURCE_ZERO) or cam.gap is not None,
            feature_source=source,
            published_at=_now(),
            state_after=[round(float(v), 9) for v in state_after],
        )
        cam.scores[frame_no] = record
        cam.frames[frame_no] = FrameRecord(
            frame_no=frame_no, features=list(features), feature_source=source
        )
        cam.state = state_after
        cam.last_frame = frame_no
        cam.last_output_frame = frame_no
        return record

    def _staged_camera(self, cam: CameraState) -> "CameraState":
        """Deep-ish copy of the mutable stream state for write-then-apply.

        A stage holds fresh ``scores``/``frames`` dicts, a copied recurrent
        state and its own event list, so candidate mutations (including the
        audit entries describing them) never touch the live camera and are
        written to disk in the same atomic document.
        """
        stage = CameraState(
            camera=cam.camera,
            state=cam.state.copy(),
            last_frame=cam.last_frame,
            last_output_frame=cam.last_output_frame,
            gap=None if cam.gap is None else Gap(**cam.gap.__dict__),
            replay=cam.replay,
            pending_fill=cam.pending_fill,
            scores={f: ScoreRecord.from_dict(r.to_dict()) for f, r in cam.scores.items()},
            frames={f: FrameRecord(**r.__dict__) for f, r in cam.frames.items()},
            events=list(cam.events),
        )
        return stage

    def _adopt_stage(self, cam: CameraState, stage: CameraState) -> None:
        cam.state = stage.state
        cam.last_frame = stage.last_frame
        cam.last_output_frame = stage.last_output_frame
        cam.gap = stage.gap
        cam.pending_fill = stage.pending_fill
        cam.scores = stage.scores
        cam.frames = stage.frames
        cam.events = stage.events

    def _persist_stage(self, cam: CameraState, stage: CameraState) -> None:
        """Write the staged document; on failure the live camera is untouched."""
        try:
            self._store.save_camera(cam.camera, self._serialize(stage))
        except StorageError as exc:
            raise PersistenceError(
                f"{exc.message}; no state change took effect, retry the request",
                retryable=True,
                state_preserved=True,
                camera=cam.camera,
            ) from exc

    # ------------------------------------------------------------------ ingest
    def ingest(self, camera: int, frame_no: int, features: list[float]) -> dict[str, Any]:
        cam = self._camera(camera)
        self._check_frame_no(frame_no)
        arr = self._validate_features(features)
        with cam.lock:
            if frame_no <= cam.last_frame:
                existing = cam.scores.get(frame_no)
                raise ConflictError(
                    f"frame {frame_no} was already scored; published scores are immutable",
                    frame_no=frame_no,
                    published_score=None if existing is None else existing.score,
                )
            if len(cam.frames) >= MAX_FRAMES_PER_CAMERA:
                raise ConflictError("camera frame buffer is full")
            skipped = frame_no - cam.last_frame - 1
            if skipped > 1:
                raise ConflictError(
                    f"jumped from frame {cam.last_frame} to {frame_no}; at most one frame "
                    "may be missing per camera at a time",
                    last_frame=cam.last_frame,
                )
            if skipped == 1:
                if cam.gap is not None:
                    raise ConflictError(
                        f"camera already has an open gap at frame {cam.gap.frame_no}; "
                        "fill it before another frame can be missing",
                        gap_frame=cam.gap.frame_no,
                    )

            # Build the candidate next state on a stage.  The live camera is
            # only adopted once the durable write is confirmed.
            stage = self._staged_camera(cam)
            gap_record: ScoreRecord | None = None
            if skipped == 1:
                before = stage.state
                stage.gap = Gap(
                    frame_no=stage.last_frame + 1,
                    state_before=[float(v) for v in before],
                    opened_at=_now(),
                )
                zero = [0.0] * model.FEATURES
                gap_after = model.step(before, np.asarray(zero))
                gap_record = self._emit(
                    stage, stage.gap.frame_no, zero, SOURCE_ZERO, gap_after
                )

            after = model.step(stage.state, arr)
            record = self._emit(stage, frame_no, arr.tolist(), SOURCE_SUBMITTED, after)
            if gap_record is not None:
                self._log(
                    stage,
                    "gap_opened",
                    frame_no=gap_record.frame_no,
                    score=gap_record.score,
                    state_digest=model.state_digest(gap_after),
                )
            gap_payload = self._gap_payload(stage)
            self._log(
                stage,
                "ingest",
                frame_no=frame_no,
                score=record.score,
                estimated=record.estimated,
                gap=gap_payload,
                state_digest=model.state_digest(after),
                replay_in_progress=stage.replay is not None,
            )
            self._persist_stage(cam, stage)
            self._adopt_stage(cam, stage)
            self._journal(cam, "ingest", frame_no=frame_no, estimated=record.estimated)
            if gap_record is not None:
                self._journal(cam, "gap_opened", frame_no=gap_record.frame_no)
            return {
                "camera": camera,
                "frame_no": frame_no,
                "score": record.score,
                "estimated": record.estimated,
                "feature_source": record.feature_source,
                "state_digest": model.state_digest(after),
                "gap": gap_payload,
                "replay_in_progress": cam.replay is not None,
                "last_output_frame": cam.last_output_frame,
                "published_scores_unchanged": True,
                "durable": self._store.durable,
            }

    def declare_missing(self, camera: int, frame_no: int) -> dict[str, Any]:
        cam = self._camera(camera)
        self._check_frame_no(frame_no)
        with cam.lock:
            expected = cam.last_frame + 1
            if frame_no != expected:
                raise ConflictError(
                    f"missing frame must be the next frame ({expected}), got {frame_no}",
                    expected_frame=expected,
                )
            if cam.gap is not None:
                raise ConflictError(
                    f"camera already has an open gap at frame {cam.gap.frame_no}",
                    gap_frame=cam.gap.frame_no,
                )
            stage = self._staged_camera(cam)
            before = stage.state
            stage.gap = Gap(
                frame_no=frame_no,
                state_before=[float(v) for v in before],
                opened_at=_now(),
            )
            zero = [0.0] * model.FEATURES
            after = model.step(before, np.asarray(zero))
            record = self._emit(stage, frame_no, zero, SOURCE_ZERO, after)
            self._log(
                stage,
                "gap_opened",
                frame_no=frame_no,
                score=record.score,
                state_digest=model.state_digest(after),
            )
            self._persist_stage(cam, stage)
            self._adopt_stage(cam, stage)
            self._journal(cam, "gap_opened", frame_no=frame_no)
            return {
                "camera": camera,
                "frame_no": frame_no,
                "score": record.score,
                "estimated": True,
                "feature_source": record.feature_source,
                "state_digest": model.state_digest(cam.state),
                "gap": self._gap_payload(cam),
                "last_output_frame": cam.last_output_frame,
                "published_scores_unchanged": True,
                "durable": self._store.durable,
            }

    # -------------------------------------------------------------------- fill
    def fill_gap(
        self,
        camera: int,
        frame_no: int,
        features: list[float],
        pace_ms: float | None = None,
        simulate_failure: bool = False,
    ) -> dict[str, Any]:
        cam = self._camera(camera)
        self._check_frame_no(frame_no)
        arr = self._validate_features(features)
        pace = self._check_pace(pace_ms)
        with cam.lock:
            if cam.gap is None:
                raise ConflictError(f"camera {camera} has no open gap to fill", frame_no=frame_no)
            if cam.gap.frame_no != frame_no:
                raise ConflictError(
                    f"open gap is at frame {cam.gap.frame_no}, not {frame_no}",
                    gap_frame=cam.gap.frame_no,
                )
            if cam.replay is not None:
                raise ConflictError(
                    f"a replay for frame {cam.replay.frame_no} is already running",
                    retryable=True,
                    gap_frame=cam.replay.frame_no,
                )
            resumed = False
            if cam.pending_fill is not None:
                # Restart (or a previous failed attempt) left a durable fill
                # for exactly this gap: resume it with the supplied features.
                if int(cam.pending_fill["frame_no"]) != frame_no:
                    raise ConflictError(
                        f"a fill for frame {cam.pending_fill['frame_no']} is already pending; "
                        "resume that one first",
                        pending_frame=cam.pending_fill["frame_no"],
                    )
                job = ReplayJob.from_pending(cam.pending_fill)
                job.simulate_failure = simulate_failure
                resumed = True
            else:
                job = ReplayJob(
                    frame_no=frame_no,
                    features=arr.tolist(),
                    state_before=np.asarray(cam.gap.state_before, dtype=np.float64),
                    state=np.asarray(cam.gap.state_before, dtype=np.float64).copy(),
                    cursor=frame_no,
                    target_at_start=cam.last_output_frame,
                    simulate_failure=simulate_failure,
                )
            # Persist the pending fill before any model arithmetic, so a crash
            # at any point before the commit still leaves a resumable job.  The
            # live camera only adopts it after the write is confirmed.
            stage = self._staged_camera(cam)
            stage.pending_fill = job.to_pending()
            if resumed:
                self._log(
                    stage,
                    "replay_resumed",
                    frame_no=frame_no,
                    attempt_failed=job.attempt_failed,
                )
            self._log(
                stage,
                "replay_started",
                frame_no=frame_no,
                target_at_start=job.target_at_start,
                pace_ms=pace,
                resumed=resumed,
            )
            self._persist_stage(cam, stage)
            self._adopt_stage(cam, stage)
            cam.replay = job
            gap_payload = self._gap_payload(cam)
            self._journal(cam, "replay_started", frame_no=frame_no, resumed=resumed)
        return self._run_replay(cam, job, pace, gap_payload)

    def _run_replay(
        self, cam: CameraState, job: ReplayJob, pace: float, gap_payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        try:
            while True:
                with cam.lock:
                    if cam.replay is not job:
                        raise ReplayError("replay was cancelled")
                    if job.cursor > cam.last_output_frame:
                        return self._commit(cam, job)
                    frame_no = job.cursor
                    if frame_no == job.frame_no:
                        x = np.asarray(job.features, dtype=np.float64)
                    else:
                        frame = cam.frames.get(frame_no)
                        if frame is None:
                            raise ReplayError(f"frame {frame_no} is missing from the replay window")
                        if frame.feature_source != SOURCE_SUBMITTED:
                            raise ReplayError(
                                f"frame {frame_no} has no real features; cannot replay across "
                                "a second gap"
                            )
                        x = np.asarray(frame.features, dtype=np.float64)

                # Demo-only fault injection: fail once at least one frame has
                # been recomputed (or immediately if the window is just the gap
                # frame).  Nothing has been committed at this point, so the
                # provisional state and published scores stay untouched.
                if job.simulate_failure and (
                    job.cursor > job.frame_no or job.target_at_start == job.frame_no
                ):
                    raise ReplayError("simulated replay failure (demo fault injection)")

                if self.replay_step_hook is not None:
                    self.replay_step_hook(cam.camera, frame_no)
                if pace > 0:
                    time.sleep(pace / 1000.0)

                state = model.step(job.state, x)
                if not np.all(np.isfinite(state)):
                    raise ReplayError(f"replay diverged at frame {frame_no}")
                job.state = state
                job.states[frame_no] = state.copy()
                job.scores[frame_no] = model.score(state)
                job.cursor = frame_no + 1
        except (StorageError, PersistenceError):
            # Commit (or pending) write failed: the stage was never adopted, so
            # the live camera still holds the pre-fill state.  The durable
            # pending fill from fill start remains and the request can retry.
            with cam.lock:
                if cam.replay is job:
                    cam.replay = None
            raise
        except ReplayError as exc:
            return self._fail_replay(cam, job, exc.message, gap_payload)
        except Exception as exc:  # defensive: never leave a half-applied replay
            return self._fail_replay(cam, job, f"replay failed: {exc}", gap_payload)

    def _fail_replay(
        self, cam: CameraState, job: ReplayJob, message: str, gap_payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        with cam.lock:
            if cam.replay is job:
                cam.replay = None
            if not job.cancelled:
                # Keep the gap and provisional state; mark the durable pending
                # fill as failed so a restart still offers a clean retry.
                job.attempt_failed = True
                stage = self._staged_camera(cam)
                stage.replay = None
                stage.pending_fill = job.to_pending()
                self._log(
                    stage,
                    "replay_failed",
                    frame_no=job.frame_no,
                    message=message,
                    cursor=job.cursor,
                )
                try:
                    self._persist_stage(cam, stage)
                except PersistenceError:
                    pass
                else:
                    self._adopt_stage(cam, stage)
                self._journal(cam, "replay_failed", frame_no=job.frame_no)
            raise ReplayError(
                message,
                retryable=True,
                state_preserved=True,
                gap_frame=job.frame_no,
                gap=gap_payload,
                last_output_frame=cam.last_output_frame,
                replayed_before_failure=job.cursor - job.frame_no,
            )

    def _commit(self, cam: CameraState, job: ReplayJob) -> dict[str, Any]:
        """Apply a completed replay; caller holds ``cam.lock``."""
        target = job.cursor - 1
        if target < job.frame_no:
            raise ReplayError(f"replay did not reach frame {job.frame_no}")
        provisional_final = cam.state
        corrected_final = job.state
        revised_at = _now()
        revisions: list[dict[str, Any]] = []

        # Build the whole corrected picture first, persist it, and only then
        # mutate the live objects: a failed write leaves zero visible changes.
        new_scores = {f: ScoreRecord.from_dict(rec.to_dict()) for f, rec in cam.scores.items()}
        for f in range(job.frame_no, target + 1):
            record = new_scores[f]
            new_score = job.scores[f]
            record.revision += 1
            record.revised_score = new_score
            record.revised_estimated = False
            record.revised_at = revised_at
            record.delta = new_score - record.score
            record.applied_to_stream = f == target
            old = cam.scores[f]
            revisions.append(
                {
                    "frame_no": f,
                    "published_score": old.score,
                    "revised_score": new_score,
                    "delta": record.delta,
                    "changed": abs(record.delta) > REVISION_EPS,
                    "estimated_at_publish": old.estimated,
                    "applied_to_stream": record.applied_to_stream,
                }
            )

        new_frames = dict(cam.frames)
        new_frames[job.frame_no] = FrameRecord(
            frame_no=job.frame_no, features=list(job.features), feature_source=SOURCE_SUBMITTED
        )
        state_shift = float(np.max(np.abs(corrected_final - provisional_final)))

        # Build the whole corrected picture on a stage, persist it, and only
        # then adopt it onto the live camera: a failed write leaves zero
        # visible changes and the same fill can simply be retried.
        stage = self._staged_camera(cam)
        stage.scores = new_scores
        stage.frames = new_frames
        stage.state = corrected_final
        stage.gap = None
        stage.replay = None
        stage.pending_fill = None
        self._log(
            stage,
            "gap_filled",
            frame_no=job.frame_no,
            replay_from=job.frame_no,
            replay_to=target,
            replayed=len(revisions),
            target_at_start=job.target_at_start,
            target_at_commit=target,
            grew_during_replay=target > job.target_at_start,
            state_shift=state_shift,
            state_digest=model.state_digest(corrected_final),
        )
        self._persist_stage(cam, stage)
        self._adopt_stage(cam, stage)
        cam.replay = None
        self._journal(
            cam,
            "gap_filled",
            frame_no=job.frame_no,
            replay_to=target,
            frames=len(revisions),
        )
        return {
            "camera": cam.camera,
            "frame_no": job.frame_no,
            "status": "replayed",
            "replay": {
                "from": job.frame_no,
                "to": target,
                "frames": len(revisions),
                "target_at_start": job.target_at_start,
                "grew_during_replay": target > job.target_at_start,
                "caught_up_with_last_published_frame": True,
                "target_frame": target,
                "resumed": job.resumed,
            },
            "gap_score": revisions[0],
            "revisions": revisions,
            "state_applied": True,
            "state_shift": state_shift,
            "state_digest_before": model.state_digest(provisional_final),
            "state_digest_after": model.state_digest(corrected_final),
            "published_scores_unchanged": True,
            "last_output_frame": cam.last_output_frame,
            "gap": None,
            "replay_in_progress": False,
            "durable": self._store.durable,
        }

    # ---------------------------------------------------------------- snapshot
    @staticmethod
    def _gap_payload(cam: CameraState) -> dict[str, Any] | None:
        if cam.gap is None:
            return None
        payload = {"frame_no": cam.gap.frame_no, "opened_at": cam.gap.opened_at}
        if cam.pending_fill is not None:
            payload["pending_fill"] = {
                "frame_no": int(cam.pending_fill["frame_no"]),
                "started_at": cam.pending_fill.get("started_at"),
                "attempt_failed": bool(cam.pending_fill.get("attempt_failed", False)),
                "resume_hint": "restart interrupted this fill; POST the same frame to /fill to resume",
            }
        return payload

    def snapshot(self, camera: int) -> dict[str, Any]:
        cam = self._camera(camera)
        with cam.lock:
            replay = None
            if cam.replay is not None:
                replay = {
                    "frame_no": cam.replay.frame_no,
                    "cursor": cam.replay.cursor,
                    "started_at": cam.replay.started_at,
                    "target_at_start": cam.replay.target_at_start,
                    "last_output_frame": cam.last_output_frame,
                }
            return {
                "camera": camera,
                "last_frame": cam.last_frame,
                "last_output_frame": cam.last_output_frame,
                "state": [round(float(v), 9) for v in cam.state],
                "state_digest": model.state_digest(cam.state),
                "gap": self._gap_payload(cam),
                "replay": replay,
                "pending_fill": cam.pending_fill,
                "scores": [cam.scores[f].to_dict() for f in sorted(cam.scores)],
                "events": list(cam.events),
                "durable": self._store.durable,
            }

    def scores(self, camera: int) -> dict[str, Any]:
        cam = self._camera(camera)
        with cam.lock:
            return {
                "camera": camera,
                "last_frame": cam.last_frame,
                "last_output_frame": cam.last_output_frame,
                "gap": self._gap_payload(cam),
                "replay_in_progress": cam.replay is not None,
                "scores": [cam.scores[f].to_dict() for f in sorted(cam.scores)],
                "durable": self._store.durable,
            }

    def reset(self) -> dict[str, Any]:
        cancelled = []
        reset_at = _now()
        with self._registry_lock:
            cams = list(self._states.values())
            for cam in cams:
                with cam.lock:
                    if cam.replay is not None:
                        cancelled.append(
                            {"camera": cam.camera, "frame_no": cam.replay.frame_no}
                        )
            # Durable first: every camera document must be removed before any
            # in-memory state is touched.  Removals are all-or-nothing too —
            # if one fails, the files already removed are restored from the
            # in-memory documents so disk and memory never diverge.
            backups = {cam.camera: self._serialize(cam) for cam in cams}
            try:
                for cam in cams:
                    self._store.clear_camera(cam.camera)
                if self._store.durable:
                    self._store.save_manifest(default_manifest(self._cameras_cfg))
            except StorageError as exc:
                for cam in cams:
                    try:
                        self._store.save_camera(cam.camera, backups[cam.camera])
                    except StorageError:
                        break
                raise PersistenceError(
                    f"{exc.message}; reset did not take effect, retry it",
                    retryable=True,
                    state_preserved=True,
                ) from exc
            for cam in cams:
                with cam.lock:
                    if cam.replay is not None:
                        # Cancel: the replay worker notices the job is gone and
                        # aborts without committing anything.
                        cam.replay.cancelled = True
                        cam.replay = None
                    cam.state = model.initial_state()
                    cam.last_frame = 0
                    cam.last_output_frame = 0
                    cam.gap = None
                    cam.replay = None
                    cam.pending_fill = None
                    cam.scores.clear()
                    cam.frames.clear()
                    cam.events.clear()
                    self._journal(cam, "reset", at=reset_at)
        return {
            "status": "reset",
            "cameras": sorted(self._states),
            "cancelled_replays": cancelled,
            "durable": self._store.durable,
        }

    def summary(self) -> dict[str, Any]:
        with self._registry_lock:
            states = list(self._states.values())
        out = []
        for cam in states:
            with cam.lock:
                out.append(
                    {
                        "camera": cam.camera,
                        "last_frame": cam.last_frame,
                        "last_output_frame": cam.last_output_frame,
                        "gap": self._gap_payload(cam),
                        "replay_in_progress": cam.replay is not None,
                        "pending_fill": cam.pending_fill is not None,
                        "published": len(cam.scores),
                        "estimated_published": sum(
                            1 for s in cam.scores.values() if s.estimated
                        ),
                        "revised": sum(1 for s in cam.scores.values() if s.revision > 0),
                        "state_digest": model.state_digest(cam.state),
                    }
                )
        return {"cameras": out, "model": model.info(), "storage": self._store.describe()}
