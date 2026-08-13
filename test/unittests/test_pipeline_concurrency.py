"""The optional concurrent intent pipeline.

A synchronous OVOS runtime processes utterances one at a time on the bus
handler thread, so a fleet of satellites sharing a runtime queues -- the
dominant latency under load. ``pipeline_workers`` > 1 runs the pipeline on a
bounded pool with a per-session FIFO queue: same-session utterances execute in
submission order on a single drainer, different sessions run in parallel, and
a global pending bound sheds load with an explicit no-match terminal instead
of buffering without limit. These tests exercise the dispatch scaffolding
(default-off is inline; on is concurrent; same-session is FIFO; different
sessions parallelize; overload sheds; shutdown drains) without the full plugin
stack, by stubbing ``_run_pipeline``.
"""
import threading
from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_core.intent_services import IntentService


def _service(workers, max_pending=None):
    svc = object.__new__(IntentService)
    svc.config = {"pipeline_workers": workers}
    if max_pending is not None:
        svc.config["pipeline_max_pending"] = max_pending
    svc._init_pipeline_concurrency(svc.config)
    svc.send_complete_intent_failure = MagicMock()
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


def test_same_session_is_fifo():
    """Same-session utterances must run one at a time IN SUBMISSION ORDER."""
    svc = _service(4)
    order = []
    overlap = {"max": 0, "cur": 0}
    lock = threading.Lock()
    gate = threading.Event()

    def record(m):
        with lock:
            overlap["cur"] += 1
            overlap["max"] = max(overlap["max"], overlap["cur"])
        gate.wait(timeout=5)  # hold the first utterance until all are queued
        order.append(m.data["seq"])
        with lock:
            overlap["cur"] -= 1
    svc._run_pipeline = record

    for i in range(6):
        m = _msg("same")
        m.data["seq"] = i
        svc.handle_utterance(m)
    gate.set()
    svc._pipeline_executor.shutdown(wait=True)
    assert overlap["max"] == 1, "same-session utterances overlapped"
    assert order == list(range(6)), f"same-session order broken: {order}"


def test_different_sessions_parallelize():
    """Different sessions must run concurrently, not queue."""
    svc = _service(4)
    overlap = {"max": 0, "cur": 0}
    lock = threading.Lock()
    both_running = threading.Event()

    def slow(_m):
        with lock:
            overlap["cur"] += 1
            overlap["max"] = max(overlap["max"], overlap["cur"])
            if overlap["cur"] >= 2:
                both_running.set()
        both_running.wait(timeout=5)
        with lock:
            overlap["cur"] -= 1
    svc._run_pipeline = slow

    for i in range(4):
        svc.handle_utterance(_msg(f"sess-{i}"))
    svc._pipeline_executor.shutdown(wait=True)
    assert overlap["max"] >= 2, "different sessions did not run concurrently"


def test_worker_exception_does_not_kill_the_pool():
    """One failing utterance must not strand the session queue or the pool."""
    svc = _service(2)
    results = []
    done = threading.Event()
    calls = {"n": 0}

    def sometimes_fail(_m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        results.append("ok")
        done.set()
    svc._run_pipeline = sometimes_fail

    # same session: the second utterance sits BEHIND the failing one in the
    # FIFO queue, proving a failure does not strand the rest of the queue.
    svc.handle_utterance(_msg("s1"))
    svc.handle_utterance(_msg("s1"))
    assert done.wait(timeout=5), "queue stalled after a worker exception"
    svc._pipeline_executor.shutdown(wait=True)
    assert results == ["ok"]


def test_overload_sheds_with_no_match_terminal():
    """Past the pending bound, utterances get an explicit no-match, never an
    unbounded queue."""
    svc = _service(2, max_pending=3)
    release = threading.Event()
    started = threading.Event()

    def blocked(_m):
        started.set()
        release.wait(timeout=10)
    svc._run_pipeline = blocked

    svc.handle_utterance(_msg("s1"))          # running (still counts pending)
    assert started.wait(timeout=5)
    svc.handle_utterance(_msg("s1"))          # queued
    svc.handle_utterance(_msg("s1"))          # queued (bound reached: 3)
    svc.handle_utterance(_msg("s1"))          # shed
    svc.handle_utterance(_msg("s2"))          # shed (bound is global)
    assert svc.send_complete_intent_failure.call_count == 2
    assert svc._pending_count == 3
    release.set()
    svc._pipeline_executor.shutdown(wait=True)
    assert svc._pending_count == 0
    assert svc._session_queues == {}, "session map leaked entries"


def test_shutdown_drains_inflight_work():
    """shutdown() must wait for queued work, then run without the executor."""
    svc = _service(2)
    done = []
    release = threading.Event()

    def slow(m):
        release.wait(timeout=10)
        done.append(m.data.get("seq"))
    svc._run_pipeline = slow

    for i in range(3):
        m = _msg("s1")
        m.data["seq"] = i
        svc.handle_utterance(m)
    release.set()
    # mimic the executor-drain portion of IntentService.shutdown()
    svc._pipeline_executor.shutdown(wait=True)
    svc._pipeline_executor = None
    assert done == [0, 1, 2], "shutdown did not drain queued work in order"
    # post-shutdown the inline path still works (executor gone -> inline)
    svc._run_pipeline = lambda m: done.append("inline")
    svc.handle_utterance(_msg("s1"))
    assert done[-1] == "inline"
