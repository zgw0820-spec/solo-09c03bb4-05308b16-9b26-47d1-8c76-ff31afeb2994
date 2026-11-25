"""Durable persistence backends for the inference engine.

Why this module exists
----------------------
The engine itself is an in-memory state machine.  When the container exits
(crash, OOM, ``docker compose restart`` ...) every frame already received and
every score already published must still be there when the web page is opened
again, so this module owns the on-disk representation and, importantly, the
*atomic* write semantics behind the rule "a frame / a missing marker / a fill
commit / a reset only takes effect once the whole write has succeeded".

Layout under the data directory (a Docker named volume in production)
---------------------------------------------------------------------
``manifest.json``        format version, fixed-weight model digest, cameras
``camera-{id}.state.json`` one document per camera, replaced atomically
``journal.jsonl``        append-only audit trail of state-changing events

A state document is written to a unique temporary file in the *same*
directory, flushed and ``fsync``-ed, then moved into place with
``os.replace`` (atomic rename).  A crash therefore leaves either the previous
document or the new one, never a half-written file.  The directory itself is
``fsync``-ed afterwards so the rename survives a power loss.

The engine writes the *full* document of a single camera per state change,
not a diff of frames: on restart the document is the whole truth and no old
frame ever has to be recomputed.  Individual frame ingestion only appends one
record to that document (the "save only this change" rule is about not
recomputing history, not about storing deltas).
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from . import model

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
JOURNAL_NAME = "journal.jsonl"
CAMERA_FILE = "camera-{camera}.state.json"


class StorageError(RuntimeError):
    """Raised when a durable write cannot be completed.

    The engine treats this as "nothing took effect": the in-memory state is
    rolled back to what was durably stored and the caller may retry the exact
    same request.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path


class Store:
    """Interface every persistence backend implements."""

    durable = False
    kind = "memory"

    def load_manifest(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def load_cameras(self) -> dict[int, dict[str, Any]]:
        raise NotImplementedError

    def save_camera(self, camera: int, doc: dict[str, Any]) -> None:
        raise NotImplementedError

    def clear_camera(self, camera: int) -> None:
        raise NotImplementedError

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        raise NotImplementedError

    def append_journal(self, entry: dict[str, Any]) -> None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


class MemoryStore(Store):
    """No-disk backend used by the unit tests and by local runs without a
    configured data directory.  Keeps the exact same interface so the engine
    code path is identical in every deployment."""

    durable = False
    kind = "memory"

    def __init__(self) -> None:
        self._docs: dict[int, dict[str, Any]] = {}
        self._manifest: dict[str, Any] | None = None
        self.journal: list[dict[str, Any]] = []

    def load_manifest(self) -> dict[str, Any] | None:
        return None if self._manifest is None else json.loads(json.dumps(self._manifest))

    def load_cameras(self) -> dict[int, dict[str, Any]]:
        return {c: json.loads(json.dumps(d)) for c, d in self._docs.items()}

    def save_camera(self, camera: int, doc: dict[str, Any]) -> None:
        self._docs[camera] = json.loads(json.dumps(doc))

    def clear_camera(self, camera: int) -> None:
        self._docs.pop(camera, None)

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self._manifest = json.loads(json.dumps(manifest))

    def append_journal(self, entry: dict[str, Any]) -> None:
        self.journal.append(json.loads(json.dumps(entry)))

    def describe(self) -> dict[str, Any]:
        return {"mode": "memory", "durable": False}


class FileStore(Store):
    """JSON-file backend with crash-safe atomic replacements."""

    durable = True
    kind = "file"

    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                f"cannot open data directory {self.data_dir}: {exc}", path=str(self.data_dir)
            ) from exc
        self._journal_path = self.data_dir / JOURNAL_NAME
        self._manifest_path = self.data_dir / MANIFEST_NAME

    # ------------------------------------------------------------- primitives
    def _path_for(self, camera: int) -> Path:
        return self.data_dir / CAMERA_FILE.format(camera=camera)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_write(self, target: Path, payload: str) -> None:
        """Write ``payload`` next to ``target`` and rename it into place."""
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            self._fsync_dir(target.parent)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise StorageError(
                f"durable write failed for {target.name}: {exc}", path=str(target)
            ) from exc

    # ------------------------------------------------------------------ loads
    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"cannot read state file {path.name}: {exc}", path=str(path)) from exc

    def load_manifest(self) -> dict[str, Any] | None:
        if not self._manifest_path.exists():
            return None
        return self._read_json(self._manifest_path)

    def load_cameras(self) -> dict[int, dict[str, Any]]:
        docs: dict[int, dict[str, Any]] = {}
        for path in sorted(self.data_dir.glob("camera-*.state.json")):
            doc = self._read_json(path)
            try:
                docs[int(doc["camera"])] = doc
            except (KeyError, TypeError, ValueError) as exc:
                raise StorageError(
                    f"state file {path.name} has no valid camera id", path=str(path)
                ) from exc
        return docs

    # ------------------------------------------------------------------ writes
    def save_camera(self, camera: int, doc: dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(self._path_for(camera), json.dumps(doc, ensure_ascii=False))

    def clear_camera(self, camera: int) -> None:
        with self._lock:
            path = self._path_for(camera)
            try:
                path.unlink(missing_ok=True)
                self._fsync_dir(path.parent)
            except OSError as exc:
                raise StorageError(
                    f"could not remove state file {path.name}: {exc}", path=str(path)
                ) from exc

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(self._manifest_path, json.dumps(manifest, ensure_ascii=False))

    def append_journal(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            try:
                with open(self._journal_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError as exc:
                raise StorageError(
                    f"could not append journal: {exc}", path=str(self._journal_path)
                ) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "file",
            "durable": True,
            "data_dir": str(self.data_dir),
        }


def default_manifest(cameras: tuple[int, ...]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "model": model.info(),
        "cameras": list(cameras),
    }
