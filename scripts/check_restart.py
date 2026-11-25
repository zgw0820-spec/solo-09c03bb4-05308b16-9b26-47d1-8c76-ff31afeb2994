#!/usr/bin/env python3
"""Deliverable check: container restart and fill-interruption recovery.

Mirrors what ``docker compose`` does in production:

* the server is started with ``QUALITY_DATA_DIR`` pointing at a directory that
  plays the role of the named volume;
* frames, a missing-frame gap and an *uncommitted* fill are produced;
* the server process is killed (``SIGKILL`` where possible) and a fresh server
  is started against the same directory;
* the reopened state is asserted frame by frame: original scores, provisional
  state, the open gap and the pending fill all come back, both cameras keep
  their own next frame, and resubmitting the gap resumes the replay with
  revised scores and corrected recurrent state identical to a gap-free run.

Run with:  python scripts/check_restart.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import model  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)


def wait_healthy(base: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=1) as resp:
                last = json.load(resp)
                if last["status"] == "ok":
                    return last
        except Exception:  # noqa: BLE001 - retry until healthy
            time.sleep(0.2)
    raise RuntimeError(f"server did not become healthy: {last}")


def reference(camera: int, samples: dict, limit: int) -> list[float]:
    frames = next(c for c in samples["cameras"] if c["camera"] == camera)["frames"]
    state = model.initial_state()
    out = []
    for frame in frames:
        if frame["frame_no"] > limit:
            break
        state = model.step(state, np.asarray(frame["features"]))
        out.append(model.score(state))
    return out


def check(condition: bool, message: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + message)
    if not condition:
        raise SystemExit(1)


def start_server(port: int, data_dir: str) -> subprocess.Popen:
    env = dict(os.environ, QUALITY_DATA_DIR=data_dir)
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.api:app",
            "--host", "127.0.0.1", "--port", str(port), "--no-server-header",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_server(proc: subprocess.Popen, *, hard: bool) -> None:
    if hard:
        proc.send_signal(signal.SIGKILL)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="quality-restart-")
    data_dir = Path(workdir) / "volume"
    data_dir.mkdir()
    port = free_port()
    base = f"http://127.0.0.1:{port}"

    print(f"[setup] 数据卷目录（模拟 Docker named volume）：{data_dir}")
    proc = start_server(port, str(data_dir))
    try:
        health = wait_healthy(base)
        check(health["storage"]["durable"] is True, "服务以文件持久化模式启动")
        c = Client(base)
        samples = c.call("GET", "/api/samples")[1]
        cam0 = {f["frame_no"]: f["features"] for f in samples["cameras"][0]["frames"]}
        cam1 = {f["frame_no"]: f["features"] for f in samples["cameras"][1]["frames"]}
        c.call("POST", "/api/reset", {"confirm": True})

        print("[1] 两路分别送帧并各开一个缺口（A 缺 #4，B 缺 #5）")
        for n in (1, 2, 3):
            c.call("POST", "/api/cameras/0/frames", {"frame_no": n, "features": cam0[n]})
        c.call("POST", "/api/cameras/0/missing", {"frame_no": 4})
        for n in (5, 6, 7):
            _, r = c.call("POST", "/api/cameras/0/frames", {"frame_no": n, "features": cam0[n]})
            check(r["estimated"] is True, f"A#{n} 暂估出分 {r['score']:.4f}")
        for n in (1, 2, 3, 4):
            c.call("POST", "/api/cameras/1/frames", {"frame_no": n, "features": cam1[n]})
        c.call("POST", "/api/cameras/1/missing", {"frame_no": 5})
        c.call("POST", "/api/cameras/1/frames", {"frame_no": 6, "features": cam1[6]})

        print("[2] A 路开始补算并在提交前 SIGKILL 容器（故障注入让补算先失败落盘）")
        # A durable failed fill leaves the pending marker on disk; then the
        # process is hard-killed to prove restart (not in-process retry) works.
        status, err = c.call(
            "POST", "/api/cameras/0/fill",
            {"frame_no": 4, "features": cam0[4], "simulate_failure": True},
        )
        check(status == 422 and err["retryable"] is True, "补算失败可重试，状态已保留")
        # Frames keep arriving while the fill is pending; they must survive too.
        _, r8 = c.call("POST", "/api/cameras/0/frames", {"frame_no": 8, "features": cam0[8]})
        check(r8["estimated"] is True, f"待续期间 A#8 照常暂估出分 {r8['score']:.4f}")
        # Snapshot the last durable state immediately before the hard kill.
        before_a = c.call("GET", "/api/cameras/0/state")[1]
        before_b = c.call("GET", "/api/cameras/1/state")[1]
        files_before = sorted(p.name for p in data_dir.iterdir())
        stop_server(proc, hard=True)
        print(f"       卷内文件：{files_before}")
    finally:
        if proc.poll() is None:
            stop_server(proc, hard=True)

    print("[3] 用同一卷重启新容器，打开网页等价的 GET /state 核对恢复结果")
    proc = start_server(port, str(data_dir))
    try:
        wait_healthy(base)
        c = Client(base)
        a = c.call("GET", "/api/cameras/0/state")[1]
        b = c.call("GET", "/api/cameras/1/state")[1]
        before_scores = {s["frame_no"]: s["score"] for s in before_a["scores"]}
        check(
            all(s["score"] == before_scores[s["frame_no"]] for s in a["scores"]
                if s["frame_no"] in before_scores),
            "A 路重启前已发原分逐帧恢复、数值不变",
        )
        check(len(a["scores"]) == 8 and a["last_output_frame"] == 8, "A 路已收 8 帧全部恢复")
        check(a["gap"]["frame_no"] == 4, "A 路缺口 #4 仍开放")
        check(
            a["pending_fill"]["frame_no"] == 4 and a["pending_fill"]["attempt_failed"] is True,
            "A 路待续补算已恢复（含失败标记）",
        )
        check(a["state_digest"] == before_a["state_digest"], "A 路暂估递推状态指纹一致")
        check(
            b["state_digest"] == before_b["state_digest"] and b["gap"]["frame_no"] == 5,
            "B 路状态与缺口独立恢复",
        )

        print("[4] 两路从各自下一帧继续，补算期间另一路照常出分")
        _, a9 = c.call("POST", "/api/cameras/0/frames", {"frame_no": 9, "features": cam0[9]})
        check(a9["estimated"] is True, f"复机后 A#9 继续暂估出分 {a9['score']:.4f}")
        _, b7 = c.call("POST", "/api/cameras/1/frames", {"frame_no": 7, "features": cam1[7]})
        check(b7["estimated"] is True, f"A 路待续时 B#7 照常出分 {b7['score']:.4f}")

        print("[5] 重新补交 A#4：续算追上最新已出分帧，修正分与递推状态一起恢复")
        status, fill = c.call("POST", "/api/cameras/0/fill", {"frame_no": 4, "features": cam0[4]})
        check(status == 200 and fill["replay"]["resumed"] is True, "补算按续算执行")
        check(fill["replay"]["to"] == 9, f"回放追上复机后最新帧 #9（实际 #{fill['replay']['to']}）")
        snap = c.call("GET", "/api/cameras/0/state")[1]
        ref = reference(0, samples, 9)
        ok = all(
            abs(s["final_score"] - ref[s["frame_no"] - 1]) < 1e-12
            for s in snap["scores"] if s["frame_no"] >= 4
        )
        check(ok, "#4–#9 修正分与无缺口参照逐位一致，原分仍并列保留")
        check(
            all(s["score"] is not None for s in snap["scores"]),
            "已发原分一列仍在（网页两列并列）",
        )
        _, a10 = c.call("POST", "/api/cameras/0/frames", {"frame_no": 10, "features": cam0[10]})
        check(
            a10["estimated"] is False
            and abs(a10["score"] - reference(0, samples, 10)[-1]) < 1e-12,
            f"A#10 使用恢复后的修正递推状态，实测出分 {a10['score']:.4f}",
        )

        print("[6] 再重启一次：已提交的修正分和下一帧递推状态仍然可用")
        stop_server(proc, hard=False)
        proc = start_server(port, str(data_dir))
        wait_healthy(base)
        c = Client(base)
        snap = c.call("GET", "/api/cameras/0/state")[1]
        check(snap["gap"] is None and snap["pending_fill"] is None, "缺口与待续标记均已清空")
        # Frames 4-9 were inside the replay window; frame 10 arrived after the
        # commit and was scored directly from the corrected state.
        check(
            all(s["revised_score"] is not None for s in snap["scores"]
                if 4 <= s["frame_no"] <= fill["replay"]["to"]),
            "补算窗口内的修正分跨重启保留",
        )
        _, a11 = c.call("POST", "/api/cameras/1/frames", {"frame_no": 8, "features": cam1[8]})
        check(a11["estimated"] is True, "B 路自己的缺口仍可独立继续暂估出分")
        _, b_fill = c.call("POST", "/api/cameras/1/fill", {"frame_no": 5, "features": cam1[5]})
        check(b_fill["status"] == "replayed", "B 路缺口补交后同样完成补算")

        print("\n重启与续跑交付核对通过。")
        return 0
    finally:
        if proc.poll() is None:
            stop_server(proc, hard=True)


if __name__ == "__main__":
    raise SystemExit(main())
