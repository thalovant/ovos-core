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


def _shutdown_executor(svc):
    """Mimic the executor-drain portion of IntentService.shutdown()."""
    with svc._session_guard:
        svc._pipeline_accepting = False
    svc._pipeline_executor.shutdown(wait=True)


def test_shutdown_drains_inflight_work():
    """shutdown() must wait for queued work; late utterances get a terminal."""
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
    _shutdown_executor(svc)
    assert done == [0, 1, 2], "shutdown did not drain queued work in order"
    # a late utterance after shutdown is shed with the no-match terminal --
    # never run against plugins that are being torn down, never queued
    svc.handle_utterance(_msg("s1"))
    svc.send_complete_intent_failure.assert_called_once()
    assert done == [0, 1, 2]
    assert svc._pending_count == 0
    assert svc._session_queues == {}


def test_admission_shutdown_interleaving_leaks_nothing():
    """Admissions racing shutdown either fully land or are fully shed.

    Drives the race from many threads while shutdown flips the accepting
    flag mid-storm: every utterance must either run or produce a no-match
    terminal -- none silently dropped -- and the pending counter and session
    map must end clean (no leaked capacity).
    """
    svc = _service(4)
    ran = []
    ran_lock = threading.Lock()

    def run(m):
        with ran_lock:
            ran.append(m.data["seq"])
    svc._run_pipeline = run

    N = 200
    barrier = threading.Barrier(9)

    def submitter(base):
        barrier.wait(timeout=5)
        for i in range(N // 8):
            m = _msg(f"sess-{(base + i) % 5}")
            m.data["seq"] = base + i
            svc.handle_utterance(m)

    def stopper():
        barrier.wait(timeout=5)
        _shutdown_executor(svc)

    threads = [threading.Thread(target=submitter, args=(k * (N // 8),))
               for k in range(8)] + [threading.Thread(target=stopper)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    shed = svc.send_complete_intent_failure.call_count
    assert len(ran) + shed == N, (
        f"{N - len(ran) - shed} utterances vanished without running "
        f"or receiving a terminal (ran={len(ran)}, shed={shed})")
    assert svc._pending_count == 0, "pending capacity leaked"
    assert svc._session_queues == {}, "session map leaked entries"


# ---------------------------------------------------------------------------
# get_pipeline matcher cache (perf: skip the ~160us per-utterance rebuild)
# ---------------------------------------------------------------------------

def _cache_service():
    svc = object.__new__(IntentService)
    svc.config = {}
    svc._init_pipeline_concurrency(svc.config)
    plugin = MagicMock()
    plugin.match = MagicMock()
    svc.pipeline_plugins = {"ovos-converse-pipeline-plugin": plugin}
    return svc


def _session(pipeline, blacklisted=None, session_id="s1"):
    sess = MagicMock()
    sess.session_id = session_id
    sess.pipeline = pipeline
    sess.blacklisted_pipelines = blacklisted or []
    return sess


def test_get_pipeline_is_cached_per_pipeline_and_blacklist():
    svc = _cache_service()
    sess = _session(["converse"])
    first = svc.get_pipeline(session=sess)
    assert len(first) == 1
    # same (pipeline, blacklist) -> the cached object, even for another session
    assert svc.get_pipeline(session=_session(["converse"], session_id="s2")) is first
    # different blacklist -> different entry
    blocked = svc.get_pipeline(session=_session(
        ["converse"], blacklisted=["converse"]))
    assert blocked is not first
    assert blocked == []


def test_pipeline_cache_cleared_on_reload():
    svc = _cache_service()
    sess = _session(["converse"])
    first = svc.get_pipeline(session=sess)
    # reload swaps the plugin; the stale matcher must not be served
    svc._pipeline_matcher_cache.clear()  # what handle_reload_pipelines does
    new_plugin = MagicMock()
    new_plugin.match = MagicMock()
    svc.pipeline_plugins = {"ovos-converse-pipeline-plugin": new_plugin}
    second = svc.get_pipeline(session=sess)
    assert second is not first
    assert second[0][1] is new_plugin.match


def test_pipeline_cache_is_bounded():
    svc = _cache_service()
    for i in range(80):  # more unique pipelines than the 64-entry bound
        svc.get_pipeline(session=_session(["converse"] * (i % 3 + 1),
                                          blacklisted=[f"nope-{i}"]))
    assert len(svc._pipeline_matcher_cache) <= 64
