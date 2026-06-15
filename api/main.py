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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("api")

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8001/predict_batch")
INFERENCE_BASE = INFERENCE_URL.rsplit("/", 1)[0]
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
BATCH_WORKER_ENABLED = os.getenv("BATCH_WORKER_ENABLED", "1") == "1"

QUEUE_KEY = "classify:queue"
PROCESSING_KEY = "classify:processing"
RETRY_KEY = "classify:retry"
RESULT_KEY = "classify:result:{}"
RESULT_TTL = 300

BATCH_SIZE = 4
BATCH_TIMEOUT = 30
MAX_RETRIES = 3
MAX_COMMENT_LENGTH = 4000

r = redis.from_url(REDIS_URL, decode_responses=True)

# Each entry is (data, raw, enqueue_time). `raw` is the exact JSON string that
# lives in the Redis processing list, so we can ack it with LREM by value.
batch_buffer = []
worker_running = True

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
    """Re-queue items left in the processing list by a previous crash."""
    moved = 0
    while r.rpoplpush(PROCESSING_KEY, QUEUE_KEY):
        moved += 1
    if moved:
        logger.warning("recovered %d orphaned in-flight item(s)", moved)


def _requeue_due_retries():
    """Move retry items whose backoff has elapsed back onto the queue."""
    now = time.time()
    for raw in r.zrangebyscore(RETRY_KEY, 0, now):
        if r.zrem(RETRY_KEY, raw):
            r.lpush(QUEUE_KEY, raw)


def _send_batch(entries):
    comments = [data["comment"] for data, _, _ in entries]
    try:
        t0 = time.time()
        resp = httpx.post(INFERENCE_URL, json={"comments": comments}, timeout=120)
        resp.raise_for_status()
        duration = time.time() - t0
        results = resp.json()["results"]
    except (httpx.HTTPError, KeyError, ValueError):
        logger.exception("inference request failed")
        return False

    if len(results) != len(entries):
        logger.error(
            "inference returned %d results for %d comments; retrying batch",
            len(results),
            len(entries),
        )
        return False

    for i, result in enumerate(results):
        if not isinstance(result, dict) or "label" not in result or "score" not in result:
            logger.error("invalid result format at index %d: %s", i, result)
            return False

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

    batch_size.observe(len(entries))
    batch_inference_duration_seconds.observe(duration)
    comments_classified_total.inc(len(entries))
    batches_processed_total.inc()
    for result in results:
        comments_classified_by_label_total.labels(label=result["label"]).inc()
    return True


def _handle_retry(entries):
    if not entries:
        return
    for data, raw, _ in entries:
        retry_count = data.get("retry_count", 0) + 1
        if retry_count > MAX_RETRIES:
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

        backoff = min(2 ** retry_count, 30)
        available_at = time.time() + backoff
        logger.info(
            "scheduling request %s for retry %d (backoff %ds)",
            data["request_id"],
            retry_count,
            backoff,
        )
        data["retry_count"] = retry_count
        r.zadd(RETRY_KEY, {json.dumps(data): available_at})
        r.lrem(PROCESSING_KEY, 1, raw)


def _flush(reason):
    global batch_buffer
    entries, batch_buffer = batch_buffer, []
    if not entries:
        return
    wait_times = [time.time() - enqueue_time for _, _, enqueue_time in entries]
    for wt in wait_times:
        batch_wait_time_seconds.observe(wt)
    logger.info("flushing %d item(s) (%s)", len(entries), reason)
    try:
        if not _send_batch(entries):
            _handle_retry(entries)
    except Exception:
        logger.exception("flush failed, requeueing %d item(s)", len(entries))
        for data, raw, _ in entries:
            r.lrem(PROCESSING_KEY, 1, raw)
            r.lpush(QUEUE_KEY, raw)


def batch_worker():
    _recover_orphans()
    while worker_running:
        try:
            _requeue_due_retries()

            if batch_buffer and (time.time() - batch_buffer[0][2]) >= BATCH_TIMEOUT:
                _flush("30s window")
                continue

            microbatch_queue_size.set(
                r.llen(QUEUE_KEY)
                + len(batch_buffer)
                + r.zcard(RETRY_KEY)
                + r.llen(PROCESSING_KEY)
            )

            raw = r.brpoplpush(QUEUE_KEY, PROCESSING_KEY, timeout=1)
            if raw is None:
                continue

            batch_buffer.append((json.loads(raw), raw, time.time()))
            if len(batch_buffer) >= BATCH_SIZE:
                _flush("batch full")
        except Exception:
            logger.exception("unexpected worker error")
            time.sleep(1)


@asynccontextmanager
async def lifespan(app):
    if BATCH_WORKER_ENABLED:
        t = threading.Thread(target=batch_worker, daemon=True)
        t.start()
    yield
    global worker_running
    worker_running = False


app = FastAPI(lifespan=lifespan)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


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
    request_id = str(uuid.uuid4())
    payload = {"request_id": request_id, "comment": req.comment}
    r.lpush(QUEUE_KEY, json.dumps(payload))
    return ClassifyResponse(request_id=request_id)


@app.get("/result/{request_id}")
def get_result(request_id: str):
    data = r.get(RESULT_KEY.format(request_id))
    if data is None:
        return {"status": "pending"}
    return json.loads(data)


@app.get("/status")
async def status():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{INFERENCE_BASE}/health")
            if resp.status_code == 200:
                return {"status": "ready", "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"}
            body = resp.json()
            return {"status": body.get("status", "loading")}
    except httpx.HTTPError:
        return {"status": "loading"}
