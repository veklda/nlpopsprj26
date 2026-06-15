from pydantic import BaseModel, Field
from fastapi import FastAPI
from transformers import pipeline
import torch

app = FastAPI()

MAX_COMMENT_LENGTH = 4000

candidate_labels = [
    "sachliche Kritik",
    "Zustimmung",
    "Sarkasmus oder Ironie",
    "Off-Topic-Kommentar",
    "Empörung oder Rant",
    "Desinformation oder Verschwörung",
]

device = 0 if torch.cuda.is_available() else "cpu"
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device=device,
)


class ClassifyRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=MAX_COMMENT_LENGTH)


class ClassifyResponse(BaseModel):
    comment: str
    label: str
    score: float


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    # The zero-shot pipeline returns labels/scores sorted by descending score.
    results = classifier(req.comment, candidate_labels)
    return ClassifyResponse(
        comment=req.comment,
        label=results["labels"][0],
        score=round(float(results["scores"][0]), 2),
    )
