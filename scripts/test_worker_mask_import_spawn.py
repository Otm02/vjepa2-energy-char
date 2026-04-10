#!/usr/bin/env python3
"""Simulate DataLoader's spawn worker: a fresh interpreter must import shims.

Run from repo root:
  python3 scripts/test_worker_mask_import_spawn.py

Exit 0 if imports succeed in a spawned child; nonzero otherwise.
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def _spawn_import_job(result_q: multiprocessing.Queue) -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    try:
        from src.masks.multiblock3d import MaskCollator  # noqa: F401
        from src.datasets.utils.video.randaugment import _rotate_level_to_arg  # noqa: F401

        result_q.put(("ok", None))
    except Exception as e:
        result_q.put(("fail", repr(e)))


def main() -> int:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ctx = multiprocessing.get_context("spawn")
    result_q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_spawn_import_job, args=(result_q,))
    p.start()
    p.join(timeout=90)
    if p.is_alive():
        p.terminate()
        print("timeout waiting for spawn child", file=sys.stderr)
        return 1
    if p.exitcode not in (0, None):
        print(f"child exit code {p.exitcode}", file=sys.stderr)
    try:
        status, err = result_q.get(timeout=1)
    except Exception as e:
        print(f"no queue result: {e}", file=sys.stderr)
        return 1
    if status != "ok":
        print(f"import failed: {err}", file=sys.stderr)
        return 1
    print("spawn import test: ok (src.masks + randaugment private helpers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
