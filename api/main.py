"""
API / orchestration service.

This is the public-facing layer of the comment-classification stack. Clients
POST a comment to /classify and receive a request_id; they then poll
/result/{request_id} until the result is ready.

Internally the service does not run the model itself. Instead it:

1.  Persists each incoming comment to a Redis list (`classify:queue`).
2.  A background worker thread pops items off the queue and accumulates them
    in an in-process microbatch buffer.
3.  When the buffer reaches BATCH_SIZE comments, or BATCH_TIMEOUT seconds
    elapse since the first item entered the buffer, the buffer is flushed to
    the inference service's /predict_batch endpoint.
4.  Results are written back to Redis under `classify:result:{request_id}`
    with a TTL, where the polling endpoint can pick them up.
5.  On failure the batch is retried with exponential backoff up to
    MAX_RETRIES times; after that an error result is stored.

Reliability features
---------------------
- In-flight items live in a separate `classify:processing` list so that a
  crash mid-flush leaves them recoverable: on startup `_recover_orphans`
  moves them back to the queue.
- A sorted set `classify:retry` holds items waiting for backoff; the worker
  periodically promotes due items back to the queue.
- All Redis result keys expire (RESULT_TTL) so storage is self-cleaning.

Endpoints
---------
POST /classify              - enqueue a comment for classification.
GET  /result/{request_id}   - fetch the result, or {"status": "pending"}.
GET  /status                - readiness probe that also pings the inference
                              service's /health endpoint.
GET  /metrics               - Prometheus scrape endpoint.
"""

import os
import json
import time
import uuid
import logging
import threading
from contextlib import asynccontextmanager

import redis
import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# Configure root logger: timestamp + level + logger name + message. The same
# format is used by the inference service so logs from both containers align.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("api")

# --- Configuration (overridable via environment variables) ------------------
# Inference endpoint. Default uses the Docker service name `inference` and the
# batch endpoint directly, since the worker always sends batches.
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8001/predict_batch")
# Base URL without the path, used to call /health for the /status probe.
INFERENCE_BASE = INFERENCE_URL.rsplit("/", 1)[0]
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# Allows disabling the worker (useful for tests or a write-only replica).
BATCH_WORKER_ENABLED = os.getenv("BATCH_WORKER_ENABLED", "1") == "1"

# --- Redis key layout -------------------------------------------------------
# A Redis list acting as the FIFO queue of pending comments.
QUEUE_KEY = "classify:queue"
# A Redis list holding items currently being processed (in-flight). Used to
# recover items lost to a crash via `_recover_orphans`.
PROCESSING_KEY = "classify:processing"
# A sorted set of items awaiting retry; score is the unix timestamp at which
# they become eligible to be re-queued.
RETRY_KEY = "classify:retry"
# Template for the per-request result key. Filled with the request_id.
RESULT_KEY = "classify:result:{}"
# Time-to-live in seconds for result keys. Keeps Redis from growing forever.
RESULT_TTL = 300

# --- Microbatching / retry tunables -----------------------------------------
BATCH_SIZE = 4          # max comments sent to inference in one call.
BATCH_TIMEOUT = 30      # max seconds the first item waits before a forced flush.
MAX_RETRIES = 3         # max retry attempts before an item is dropped with an error.
MAX_COMMENT_LENGTH = 4000  # mirrors the inference service's cap.

# Synchronous Redis client. decode_responses=True returns strings instead of
# bytes, which is convenient since we (de)serialize JSON ourselves.
r = redis.from_url(REDIS_URL, decode_responses=True)

# In-memory microbatch buffer. Each entry is a tuple of
#   (data, raw, enqueue_time)
# where:
#   data         - the parsed JSON dict ({request_id, comment, retry_count?}).
#   raw          - the exact JSON string currently held in the Redis
#                  PROCESSING_KEY list; kept so we can ack it with LREM by
#                  value once the batch succeeds or is retried.
#   enqueue_time - monotonic-ish time.time() at the moment the item was
#                  pulled off the queue; used for the batch-wait-time metric
#                  and the BATCH_TIMEOUT flush decision.
batch_buffer = []
# Flag checked by the worker loop; flipped to False on shutdown so the daemon
# thread can exit cleanly.
worker_running = True

# --- Prometheus metrics -----------------------------------------------------
comments_classified_total = Counter(
    "comments_classified_total",
    "Total number of comments classified",
)
batches_processed_total = Counter(
    "batches_processed_total",
    "Total number of batches processed",
)
batch_size = Histogram(
    "batch_size",
    "Size of batches processed",
    buckets=[1, 2, 4, 8, 16],
)
batch_inference_duration_seconds = Histogram(
    "batch_inference_duration_seconds",
    "Duration of model inference per batch in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
batch_wait_time_seconds = Histogram(
    "batch_wait_time_seconds",
    "How long a comment has to wait in a microbatch in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)
microbatch_queue_size = Gauge(
    "microbatch_queue_size",
    "Current number of comments waiting in the queue",
)
comments_classified_by_label_total = Counter(
    "comments_classified_by_label_total",
    "Distribution of predicted labels",
    ["label"],
)


def _recover_orphans():
    """Re-queue items left in the processing list by a previous crash.

    Called once at worker startup. `rpoplpush` atomically pops from
    PROCESSING_KEY and pushes onto QUEUE_KEY; we loop until the processing
    list is empty. Without this, any item being flushed when the process
    died would be silently lost.
    """
    moved = 0
    while r.rpoplpush(PROCESSING_KEY, QUEUE_KEY):
        moved += 1
    if moved:
        logger.warning("recovered %d orphaned in-flight item(s)", moved)


def _requeue_due_retries():
    """Move retry items whose backoff has elapsed back onto the queue.

    The RETRY_KEY sorted set is scored by available-at timestamp. We fetch
    all members with score <= now, and for each one atomically remove it
    from the set and push it back to the queue. The `zrem` is checked so we
    don't double-enqueue if another worker raced us.
    """
    now = time.time()
    for raw in r.zrangebyscore(RETRY_KEY, 0, now):
        if r.zrem(RETRY_KEY, raw):
            r.lpush(QUEUE_KEY, raw)


def _send_batch(entries):
    """POST a batch of comments to the inference service and persist results.

    Returns True on full success, False on any failure (network error,
    malformed response, count mismatch). On False, the caller routes the
    batch through `_handle_retry`. This function does NOT remove items from
    PROCESSING_KEY on failure; that is the retry handler's job.
    """
    comments = [data["comment"] for data, _, _ in entries]
    try:
        t0 = time.time()
        # 120s timeout matches the largest Prometheus bucket; the inference
        # service should never legitimately take longer on a small batch.
        resp = httpx.post(INFERENCE_URL, json={"comments": comments}, timeout=120)
        resp.raise_for_status()
        duration = time.time() - t0
        results = resp.json()["results"]
    except (httpx.HTTPError, KeyError, ValueError):
        # HTTPError covers connection/timeout/5xx; KeyError/ValueError cover
        # a malformed JSON body or missing "results" field.
        logger.exception("inference request failed")
        return False

    # Defend against the inference service returning a different number of
    # results than comments. We retry the whole batch rather than risk
    # mis-aligning results with comments.
    if len(results) != len(entries):
        logger.error(
            "inference returned %d results for %d comments; retrying batch",
            len(results),
            len(entries),
        )
        return False

    # Validate the shape of each result before storing, so a single bad
    # result doesn't poison the result store with a confusing payload.
    for i, result in enumerate(results):
        if not isinstance(result, dict) or "label" not in result or "score" not in result:
            logger.error("invalid result format at index %d: %s", i, result)
            return False

    # Success path: store each result under its request_id and ack the item
    # in the processing list by removing the exact raw JSON string we added.
    for (data, raw, _), result in zip(entries, results):
        r.set(
            RESULT_KEY.format(data["request_id"]),
            json.dumps({
                "comment": data["comment"],
                "label": result["label"],
                "score": result["score"],
            }),
            ex=RESULT_TTL,
        )
        r.lrem(PROCESSING_KEY, 1, raw)

    # Update Prometheus metrics: batch size, inference latency, totals, and
    # the per-label distribution counter.
    batch_size.observe(len(entries))
    batch_inference_duration_seconds.observe(duration)
    comments_classified_total.inc(len(entries))
    batches_processed_total.inc()
    for result in results:
        comments_classified_by_label_total.labels(label=result["label"]).inc()
    return True


def _handle_retry(entries):
    """Schedule failed entries for retry with exponential backoff.

    Each entry's retry_count is bumped. If it exceeds MAX_RETRIES the entry
    is dropped and an error result is stored so the client's polling loop
    gets a terminal (non-pending) response. Otherwise the entry is added to
    the RETRY_KEY sorted set with a score equal to its available-at time,
    and the original raw string is removed from the processing list.
    """
    if not entries:
        return
    for data, raw, _ in entries:
        retry_count = data.get("retry_count", 0) + 1
        if retry_count > MAX_RETRIES:
            # Give up: write a terminal error result so /result stops
            # returning "pending" for this request_id.
            logger.warning(
                "dropping request %s after %d retries", data["request_id"], MAX_RETRIES
            )
            r.set(
                RESULT_KEY.format(data["request_id"]),
                json.dumps({
                    "comment": data["comment"],
                    "label": "",
                    "score": 0.0,
                    "status": "error",
                    "detail": "max retries exceeded",
                }),
                ex=RESULT_TTL,
            )
            r.lrem(PROCESSING_KEY, 1, raw)
            continue

        # Exponential backoff: 2^retry seconds, capped at 30s. The cap keeps
        # retries from being pushed minutes into the future for large
        # retry counts.
        backoff = min(2 ** retry_count, 30)
        available_at = time.time() + backoff
        logger.info(
            "scheduling request %s for retry %d (backoff %ds)",
            data["request_id"],
            retry_count,
            backoff,
        )
        # Persist the updated retry_count by re-serializing data into the
        # retry set. We store the data (not raw) here because the original
        # raw string was shaped for the processing list and we want the
        # next dequeue to see the bumped retry_count.
        data["retry_count"] = retry_count
        r.zadd(RETRY_KEY, {json.dumps(data): available_at})
        r.lrem(PROCESSING_KEY, 1, raw)


def _flush(reason):
    """Flush the current microbatch buffer to the inference service.

    `reason` is a short human-readable string included in the log line so we
    can tell from logs whether flushes were triggered by "batch full" or by
    the "30s window" timeout. The buffer is swapped out atomically so any
    concurrent append (there shouldn't be any, since only the worker
    appends) goes into a fresh buffer.

    On a failed send the batch is handed to `_handle_retry`. If something
    throws unexpectedly outside those two paths we fall back to pushing the
    entries back onto the queue so they are not lost.
    """
    global batch_buffer
    entries, batch_buffer = batch_buffer, []
    if not entries:
        return
    # Record how long each item waited in the microbatch buffer before being
    # flushed. This is the user-visible "queue + batch" latency component.
    wait_times = [time.time() - enqueue_time for _, _, enqueue_time in entries]
    for wt in wait_times:
        batch_wait_time_seconds.observe(wt)
    logger.info("flushing %d item(s) (%s)", len(entries), reason)
    try:
        if not _send_batch(entries):
            _handle_retry(entries)
    except Exception:
        # Last-resort safety net: never lose items to an exception. Put
        # them back at the head of the queue for immediate reprocessing.
        logger.exception("flush failed, requeueing %d item(s)", len(entries))
        for data, raw, _ in entries:
            r.lrem(PROCESSING_KEY, 1, raw)
            r.lpush(QUEUE_KEY, raw)


def batch_worker():
    """Background worker that drains the Redis queue and forms microbatches.

    The loop runs forever (until `worker_running` is flipped False on
    shutdown) and on each iteration:

    1. Promotes any due retry items back to the queue.
    2. If the buffer has items and the oldest one has been waiting longer
       than BATCH_TIMEOUT, flush the buffer on timeout grounds.
    3. Updates the queue-size gauge so we can alert on backlog growth.
    4. Blocks up to 1 second waiting for a new item from the queue. The
       short block keeps the loop responsive to shutdown and to retry
       promotion while still being efficient when idle.
    5. Appends the popped item to the buffer and flushes if the buffer is
       now full.
    """
    _recover_orphans()
    while worker_running:
        try:
            # Step 1: promote retries whose backoff has elapsed.
            _requeue_due_retries()

            # Step 2: time-based flush so a partial buffer is not held
            # forever waiting for BATCH_SIZE to fill.
            if batch_buffer and (time.time() - batch_buffer[0][2]) >= BATCH_TIMEOUT:
                _flush("30s window")
                continue

            # Step 3: report current backlog size to Prometheus. We sum
            # across all the queues so the gauge reflects total pending
            # work, not just the main queue.
            microbatch_queue_size.set(
                r.llen(QUEUE_KEY)
                + len(batch_buffer)
                + r.zcard(RETRY_KEY)
                + r.llen(PROCESSING_KEY)
            )

            # Step 4: atomically pop from QUEUE_KEY and push onto
            # PROCESSING_KEY. A 1s timeout means brpoplpush returns None on
            # idle, letting the loop re-check shutdown / retries promptly.
            raw = r.brpoplpush(QUEUE_KEY, PROCESSING_KEY, timeout=1)
            if raw is None:
                continue

            # Step 5: buffer the item; flush immediately if we hit the
            # configured batch size.
            batch_buffer.append((json.loads(raw), raw, time.time()))
            if len(batch_buffer) >= BATCH_SIZE:
                _flush("batch full")
        except Exception:
            # Never let the worker thread die; log and back off briefly so
            # a tight error loop does not spin a CPU.
            logger.exception("unexpected worker error")
            time.sleep(1)


@asynccontextmanager
async def lifespan(app):
    """Start the background worker thread on FastAPI startup, stop it on shutdown."""
    if BATCH_WORKER_ENABLED:
        # Daemon thread so it is killed automatically if the process exits
        # abruptly; we still try to stop it cleanly via worker_running.
        t = threading.Thread(target=batch_worker, daemon=True)
        t.start()
    yield
    global worker_running
    worker_running = False


app = FastAPI(lifespan=lifespan)

# Mount the Prometheus ASGI app under /metrics so the scrape target is
# `http://api:8000/metrics`.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# --- Request / response schemas --------------------------------------------
class ClassifyRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=MAX_COMMENT_LENGTH)


class ClassifyResponse(BaseModel):
    request_id: str


class ResultResponse(BaseModel):
    comment: str
    label: str
    score: float


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Enqueue a comment for asynchronous classification.

    Returns immediately with a request_id the client can use to poll
    /result/{request_id}. The actual classification happens later in the
    worker thread when the comment is flushed as part of a microbatch.
    """
    request_id = str(uuid.uuid4())
    payload = {"request_id": request_id, "comment": req.comment}
    # LPUSH + RPOP give FIFO order: new items go to the head, the worker
    # pops from the tail via brpoplpush.
    r.lpush(QUEUE_KEY, json.dumps(payload))
    return ClassifyResponse(request_id=request_id)


@app.get("/result/{request_id}")
def get_result(request_id: str):
    """Poll for a classification result.

    Returns the stored result JSON if present (which may include a
    terminal "error" status when retries were exhausted), otherwise a
    minimal {"status": "pending"} body so clients can keep polling.
    """
    data = r.get(RESULT_KEY.format(request_id))
    if data is None:
        return {"status": "pending"}
    return json.loads(data)


@app.get("/status")
async def status():
    """Aggregate readiness probe.

    Pings the inference service's /health endpoint asynchronously and
    reports "ready" only when the downstream model is up. Any HTTP error
    or non-200 response is interpreted as the system still warming up.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{INFERENCE_BASE}/health")
            if resp.status_code == 200:
                return {"status": "ready", "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"}
            body = resp.json()
            return {"status": body.get("status", "loading")}
    except httpx.HTTPError:
        return {"status": "loading"}
