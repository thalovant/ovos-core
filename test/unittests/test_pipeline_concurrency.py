"""The optional concurrent intent pipeline.

A synchronous OVOS runtime processes utterances one at a time on the bus
handler thread, so a fleet of satellites sharing a runtime queues -- the
dominant latency under load. ``pipeline_workers`` > 1 runs the pipeline on a
bounded pool, serialized per session. These tests exercise the dispatch
scaffolding (default-off is inline; on is concurrent; same-session serializes;
different sessions parallelize) without the full plugin stack, by stubbing
``_run_pipeline``.
"""
import time
import threading
from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_core.intent_services import IntentService


def _service(workers):
    svc = object.__new__(IntentService)
    svc.config = {"pipeline_workers": workers}
    # replicate the two lines __init__ runs for the concurrency machinery
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    w = max(1, int(svc.config.get("pipeline_workers", 1)))
    svc._pipeline_executor = (
        ThreadPoolExecutor(max_workers=w, thread_name_prefix="intent-pipeline")
        if w > 1 else None)
    svc._session_locks = {}
    svc._session_locks_guard = Lock()
    return svc


def _msg(session_id):
    return Message("recognizer_loop:utterance",
                   {"utterances": ["hello"]},
                   {"session": {"session_id": session_id}})


def test_default_is_inline_no_executor():
    """workers=1 must be the historical inline path: no pool, direct call."""
    svc = _service(1)
    assert svc._pipeline_executor is None
    called = []
    svc._run_pipeline = lambda m: called.append(m) or "result"
    out = svc.handle_utterance(_msg("s1"))
    assert out == "result"
    assert len(called) == 1


def test_workers_gt_1_runs_on_the_pool():
    svc = _service(4)
    assert svc._pipeline_executor is not None
    ran = threading.Event()
    svc._run_pipeline = lambda m: ran.set()
    assert svc.handle_utterance(_msg("s1")) is None  # returns immediately
    assert ran.wait(timeout=5), "pipeline never ran on the pool"


def test_same_session_serializes():
    """Two utterances for one session must not overlap."""
    svc = _service(4)
    overlap = {"max": 0, "cur": 0}
    lock = threading.Lock()

    def slow(_m):
        with lock:
            overlap["cur"] += 1
            overlap["max"] = max(overlap["max"], overlap["cur"])
        time.sleep(0.1)
        with lock:
            overlap["cur"] -= 1
    svc._run_pipeline = slow

    svc.handle_utterance(_msg("same"))
    svc.handle_utterance(_msg("same"))
    svc._pipeline_executor.shutdown(wait=True)
    assert overlap["max"] == 1, "same-session utterances overlapped"


def test_different_sessions_parallelize():
    """Different sessions must run concurrently, not queue."""
    svc = _service(4)
    overlap = {"max": 0, "cur": 0}
    lock = threading.Lock()

    def slow(_m):
        with lock:
            overlap["cur"] += 1
            overlap["max"] = max(overlap["max"], overlap["cur"])
        time.sleep(0.2)
        with lock:
            overlap["cur"] -= 1
    svc._run_pipeline = slow

    for i in range(4):
        svc.handle_utterance(_msg(f"sess-{i}"))
    svc._pipeline_executor.shutdown(wait=True)
    assert overlap["max"] >= 2, "different sessions did not run concurrently"


def test_worker_exception_does_not_kill_the_pool():
    svc = _service(2)
    results = []
    calls = {"n": 0}

    def sometimes_fail(_m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        results.append("ok")
    svc._run_pipeline = sometimes_fail

    svc.handle_utterance(_msg("s1"))       # raises inside the worker
    time.sleep(0.1)
    svc.handle_utterance(_msg("s2"))       # pool must still accept work
    svc._pipeline_executor.shutdown(wait=True)
    assert results == ["ok"], "a worker exception poisoned the pool"
