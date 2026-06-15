import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import pipeline
import torch
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("inference")

MAX_COMMENT_LENGTH = 4000

candidate_labels = [
    "sachliche Kritik",
    "Zustimmung",
    "Sarkasmus oder Ironie",
    "Off-Topic-Kommentar",
    "Empörung oder Rant",
    "Desinformation oder Verschwörung",
]

classifier = None
model_ready = False
model_error = None

predictions_total = Counter(
    "inference_predictions_total",
    "Total number of comments scored by the model",
)
inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Model forward-pass duration per request in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
inference_batch_size = Histogram(
    "inference_batch_size",
    "Number of comments per model call",
    buckets=[1, 2, 4, 8, 16],
)
model_ready_gauge = Gauge(
    "inference_model_ready",
    "1 when the model is loaded and serving, 0 otherwise",
)


@asynccontextmanager
async def lifespan(app):
    global classifier, model_ready, model_error
    logger.info("loading model...")
    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        loop = asyncio.get_running_loop()
        classifier = await loop.run_in_executor(
            None,
            lambda: pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                device=device,
            ),
        )
        model_ready = True
        model_ready_gauge.set(1)
        logger.info("model ready")
    except Exception as e:
        model_error = str(e)
        logger.error("model loading failed: %s", e)
    yield
    model_ready = False
    model_ready_gauge.set(0)
    classifier = None


app = FastAPI(lifespan=lifespan)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class PredictRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=MAX_COMMENT_LENGTH)


class PredictResponse(BaseModel):
    label: str
    score: float


class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(min_length=1)


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


def _top_label(output):
    # The zero-shot pipeline returns labels/scores sorted by descending score.
    return PredictResponse(label=output["labels"][0], score=round(float(output["scores"][0]), 2))


@app.get("/health")
def health():
    if model_error:
        return JSONResponse(status_code=503, content={"status": "error", "detail": model_error})
    if not model_ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model_ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    t0 = time.time()
    output = classifier(req.comment, candidate_labels)
    inference_duration_seconds.observe(time.time() - t0)
    inference_batch_size.observe(1)
    predictions_total.inc()
    return _top_label(output)


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    if not model_ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    t0 = time.time()
    outputs = classifier(req.comments, candidate_labels)
    inference_duration_seconds.observe(time.time() - t0)
    inference_batch_size.observe(len(req.comments))
    predictions_total.inc(len(req.comments))
    return PredictBatchResponse(results=[_top_label(o) for o in outputs])
