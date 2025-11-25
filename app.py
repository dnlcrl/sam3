"""Flask server wrapping SAM3 video predictor with point-based prompts."""

import io
import json
import os
import tempfile
import uuid
import zipfile
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import pydicom
import torch
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Flask, jsonify, request, send_file
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor

inference_states: Dict[str, Dict] = {}
scheduler = BackgroundScheduler()
scheduler.start()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024  # 1000 MB

# Build the SAM3 video predictor once at startup.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_PREDICTOR = build_sam3_video_predictor()


# Helper function to delete session
def delete_session(session_id: str) -> None:
    """Remove cached state for an expired session."""
    if session_id in inference_states:
        del inference_states[session_id]
        print(f"Session {session_id} deleted due to timeout.")


# Function to set or reset a session timer
def set_or_reset_timer(session_id, timeout_seconds=3000):
    job_id = f"session_cleanup_{session_id}"
    # Remove the existing job for this session if it exists
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    # Schedule a new job to delete the session after 'timeout_seconds'
    scheduler.add_job(
        delete_session,
        trigger=IntervalTrigger(seconds=timeout_seconds),
        id=job_id,
        args=[session_id],
        replace_existing=True,
    )


@app.route("/get_server_status", methods=["POST"])
def get_server_status():
    return {"status": "happily running", "device": _DEVICE}, 200


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File is too large"}), 413


@app.route("/initialize_video", methods=["POST"])
def initialize_video():
    meta = request.form.to_dict()
    session_id = meta.get("session_id", str(uuid.uuid4()))

    zip_file = request.files.get("data_binary")
    print("received params", meta)
    # get ww/wl values
    ww = meta.get("ww", None)
    wl = meta.get("wl", None)
    if ww:
        print(f"found WW value: ${ww}")
    if wl:
        print(f"found WW value: ${wl}")

    if not zip_file:
        return jsonify({"error": "No zip file provided"}), 400

    print(f"Received file: {zip_file.filename}")

    # Save the uploaded file to a temporary location
    temp_zip_path = os.path.join(tempfile.gettempdir(), zip_file.filename)
    zip_file.save(temp_zip_path)

    # Check if the file was saved correctly and has a non-zero size
    if os.path.getsize(temp_zip_path) == 0:
        print("File size is 0 after saving!")
        return jsonify({"error": "File size is 0"}), 400

    temp_dir = tempfile.mkdtemp()
    dcm_dir = os.path.join(temp_dir, "dicoms")
    jpg_dir = os.path.join(temp_dir, "jpgs")
    os.makedirs(dcm_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file) as zip_ref:
        zip_ref.extractall(dcm_dir)

    dicom_filenames = [
        fname for fname in os.listdir(dcm_dir) if fname.lower().endswith(".dcm")
    ]
    dicom_paths = [os.path.join(dcm_dir, f) for f in dicom_filenames]
    files = [pydicom.dcmread(fname) for fname in dicom_paths]

    # Sort slices using SliceLocation if available, otherwise fallback to InstanceNumber or ImagePositionPatient
    def sort_key(s):
        if hasattr(s, "SliceLocation"):
            return s.SliceLocation
        elif hasattr(s, "InstanceNumber"):
            return s.InstanceNumber
        elif hasattr(s, "ImagePositionPatient"):
            return s.ImagePositionPatient[2]
        return float("inf")

    slices = sorted(files, key=sort_key)

    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img2d = s.pixel_array.astype(np.float32)
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)
        img2d = img2d * slope + intercept

        if ww and wl:
            ww_f = float(ww)
            wl_f = float(wl)
            min_val = wl_f - ww_f / 2
            max_val = wl_f + ww_f / 2
            img2d = np.clip(img2d, min_val, max_val)
            img2d = (img2d - min_val) / (max_val - min_val) * 255

        img3d[:, :, i] = img2d

    for idx in range(img3d.shape[2]):
        image_array = img3d[:, :, idx]
        image = Image.fromarray(image_array).convert("L")
        image.save(os.path.join(jpg_dir, f"{idx}.jpg"), quality=100)

    start_resp = _PREDICTOR.handle_request(
        {"type": "start_session", "resource_path": jpg_dir, "session_id": session_id}
    )
    session_id = start_resp["session_id"]
    inference_states[session_id] = {
        "temp_dir": temp_dir,
        "n_frames": len(os.listdir(jpg_dir)),
    }
    set_or_reset_timer(session_id)
    return jsonify({"session_id": session_id, "n_frames": len(os.listdir(jpg_dir))})


@app.route("/add_points", methods=["POST"])
def add_points():
    data = request.form.to_dict()
    if data is None:
        return jsonify({"error": "No data provided"}), 400
    session_id = data.get("session_id")
    frame_idx = int(data.get("frame_idx"))
    obj_id = int(data.get("obj_id"))
    points = json.loads(data.get("points"))
    labels = json.loads(data.get("labels"))
    if (
        session_id is None
        or frame_idx is None
        or obj_id is None
        or points is None
        or labels is None
    ):
        return jsonify({"error": "All fields are required"}), 400

    if session_id not in inference_states:
        return jsonify({"error": "Invalid session_id"}), 400

    points = np.array(points, dtype=np.float32).tolist()
    labels = np.array(labels, dtype=np.int32).tolist()

    response = _PREDICTOR.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "points": points,
            "point_labels": labels,
            "obj_id": obj_id,
        }
    )

    outputs = response["outputs"]
    masks = outputs.get("out_binary_masks")
    out_obj_ids = outputs.get("out_obj_ids")
    boxes = outputs.get("out_boxes_xywh")
    scores = outputs.get("out_probs")

    set_or_reset_timer(session_id)

    return jsonify(
        {
            "frame_index": response.get("frame_index", frame_idx),
            "obj_ids": out_obj_ids.tolist() if out_obj_ids is not None else [],
            "boxes": boxes.tolist() if boxes is not None else [],
            "scores": scores.tolist() if scores is not None else [],
            "num_masks": int(masks.shape[0]) if masks is not None else 0,
        }
    )


@app.route("/propagate_masks", methods=["POST"])
def propagate_masks():
    data = request.form.to_dict()
    if data is None:
        return jsonify({"error": "No data provided"}), 400
    session_id = data.get("session_id")
    if session_id is None:
        return jsonify({"error": "session_id is required"}), 400

    state_info = inference_states.get(session_id)
    if state_info is None:
        return jsonify({"error": "Invalid session_id"}), 400

    video_segments = {}
    stream = _PREDICTOR.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
        }
    )

    for frame_idx, outputs in stream:
        obj_ids = outputs.get("out_obj_ids", [])
        masks = outputs.get("out_binary_masks", [])
        video_segments[frame_idx] = {
            int(obj_ids[i]): masks[i] for i in range(len(obj_ids))
        }

    set_or_reset_timer(session_id)
    return jsonify(
        {
            "num_frames": len(video_segments),
            "frame_indices": sorted(video_segments.keys()),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
