"""HTTP inference service for production-style deployment.

The original assessment entrypoints (`predict.py`, `apply_fabric.py`) remain the
source of truth for model behavior. This module wraps them with a small FastAPI
surface so the same code can run behind Docker/Kubernetes without changing the
segmentation logic.
"""
import io
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from apply_fabric import apply_fabric
from model.postprocess import OUTPUT_INDEX
from predict import load_model, predict_mask

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "weights/best.pt")
SERVICE_NAME = os.getenv("SERVICE_NAME", "panel-seg")

app = FastAPI(
    title="Garment Panel Segmentation Service",
    version="1.0.0",
    description="CPU garment panel segmentation and deterministic fabric fill API.",
)


@lru_cache(maxsize=1)
def get_model():
    weights_path = Path(MODEL_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    device = torch.device("cpu")
    return load_model(str(weights_path), device)


def _png_response(array: np.ndarray) -> Response:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


def _read_image(upload: UploadFile, mode: str) -> np.ndarray:
    try:
        image = Image.open(upload.file).convert(mode)
        return np.array(image)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {upload.filename}") from exc


@app.get("/health")
def health():
    try:
        get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "service": SERVICE_NAME, "device": "cpu"}


@app.get("/metadata")
def metadata():
    return {
        "service": SERVICE_NAME,
        "weights": MODEL_WEIGHTS,
        "output_index": OUTPUT_INDEX,
        "device": "cpu",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        mask = predict_mask(get_model(), tmp_path, torch.device("cpu"))
        return _png_response(mask)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/apply-fabric")
def apply_fabric_endpoint(
    panel_name: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    swatch: UploadFile = File(...),
):
    if panel_name not in OUTPUT_INDEX or panel_name == "background":
        valid = [name for name in OUTPUT_INDEX if name != "background"]
        raise HTTPException(status_code=400, detail=f"panel_name must be one of {valid}")

    image_array = _read_image(image, "RGB")
    mask_array = _read_image(mask, "L")
    swatch_array = _read_image(swatch, "RGB")

    if image_array.shape[:2] != mask_array.shape[:2]:
        raise HTTPException(status_code=400, detail="image and mask dimensions must match")

    result = apply_fabric(image_array, mask_array, panel_name, swatch_array)
    return _png_response(result)