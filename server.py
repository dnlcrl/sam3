import io
import os
from typing import Tuple

import torch
from flask import Flask, jsonify, request
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def _load_model_and_processor() -> Tuple[torch.nn.Module, Sam3Processor, str]:
    """Instantiate the SAM3 image model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    checkpoint_path = os.environ.get("SAM3_CHECKPOINT")
    load_from_hf = checkpoint_path is None

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
    )
    processor = Sam3Processor(model)
    return model, processor, device


app = Flask(__name__)
model, processor, DEVICE = _load_model_and_processor()


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """Lightweight health endpoint."""
    return jsonify({"status": "ok", "device": DEVICE}), 200


@app.route("/segment", methods=["POST"])
def segment():
    """
    Run text-prompted segmentation on an uploaded image.
    Request must contain multipart form-data with:
      - image: the image file
      - prompt: text prompt describing the target concept
    """
    if "image" not in request.files:
        return jsonify({"error": "missing image file"}), 400
    prompt = request.form.get("prompt")
    if not prompt:
        return jsonify({"error": "missing prompt"}), 400

    image_bytes = request.files["image"].read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"invalid image: {exc}"}), 400

    with torch.no_grad():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    boxes = output.get("boxes")
    scores = output.get("scores")
    masks = output.get("masks")
    response = {
        "boxes": boxes.cpu().tolist() if boxes is not None else [],
        "scores": scores.cpu().tolist() if scores is not None else [],
        "num_masks": int(masks.shape[0]) if masks is not None else 0,
    }
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
