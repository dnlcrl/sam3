"""Flask server wrapping SAM3 video tracker (SAM2-style example 1)."""

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import nullcontext
from io import BytesIO
from typing import Dict

import nibabel as nib
import numpy as np
import pydicom
import torch
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Flask, jsonify, request, send_file
from PIL import Image

from sam3.model_builder import build_sam3_video_model

inference_states: Dict[str, Dict] = {}
scheduler = BackgroundScheduler()
scheduler.start()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024  # 1000 MB


def _select_device() -> torch.device:
    """Select computation device following the notebook logic."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire process
        # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        print("autocast set")
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and "
            "might give numerically different outputs and sometimes degraded "
            "performance on MPS."
        )
    return device


# Build the SAM3 video model and tracker once at startup (example 1 style).
DEVICE = _select_device()
checkpoint_path = os.environ.get("SAM3_CHECKPOINT")
load_from_hf = checkpoint_path is None
sam3_model = build_sam3_video_model(
    checkpoint_path=checkpoint_path,
    load_from_HF=load_from_hf,
    device=DEVICE,
)
PREDICTOR = sam3_model.tracker
PREDICTOR.backbone = sam3_model.detector.backbone
print(torch.__version__)


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


def convert_masks_to_nii(
    video_segments: Dict[int, Dict[int, np.ndarray]], num_frames: int
) -> nib.Nifti1Image:
    """Convert propagated video masks into a 4D NIfTI volume.

    The output layout is (num_frames, 1, H, W) so that downstream
    consumers can index frames as in the example notebook/tests.
    """
    if not video_segments:
        # No masks propagated; return an empty volume with a single 1x1 frame.
        volume = np.zeros((1, 1, 1, 1), dtype=np.float32)
        affine = np.eye(4, dtype=np.float32)
        return nib.Nifti1Image(volume, affine)

    # Infer spatial dimensions from the first mask.
    first_frame_masks = next(iter(video_segments.values()))
    first_mask = next(iter(first_frame_masks.values()))
    _, height, width = first_mask.shape

    volume = np.zeros((int(num_frames), 1, height, width), dtype=np.float32)

    # Encode each object id as a distinct integer label per frame.
    for frame_idx, obj_dict in video_segments.items():
        if frame_idx >= num_frames:
            continue
        for obj_id, mask in obj_dict.items():
            mask_arr = mask.astype(np.float32)
            label_value = 1.0 + float(obj_id)
            volume[frame_idx, 0] = np.maximum(
                volume[frame_idx, 0],
                mask_arr * label_value,
            )

    affine = np.eye(4, dtype=np.float32)
    return nib.Nifti1Image(volume, affine)


@app.route("/get_server_status", methods=["POST"])
def get_server_status():
    return {"status": "happily running", "device": str(DEVICE)}, 200


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

    # Initialize tracker state on this JPEG folder, following the notebook flow.
    inference_state = PREDICTOR.init_state(video_path=jpg_dir)
    PREDICTOR.clear_all_points_in_video(inference_state)

    inference_states[session_id] = {
        "temp_dir": temp_dir,
        "n_frames": len(os.listdir(jpg_dir)),
        "inference_state": inference_state,
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

    session_state = inference_states.get(session_id)
    if session_state is None:
        return jsonify({"error": "Invalid session_id"}), 400

    # Normalize points to relative coordinates as in the notebook.
    jpg_dir = os.path.join(session_state["temp_dir"], "jpgs")
    frame_path = os.path.join(jpg_dir, f"{frame_idx}.jpg")
    with Image.open(frame_path) as img:
        width, height = img.size

    points_arr = np.array(points, dtype=np.float32).reshape(-1, 2)
    rel_points = [[float(x) / width, float(y) / height] for x, y in points_arr]
    labels_arr = np.array(labels, dtype=np.int32).reshape(-1)

    inference_state = session_state["inference_state"]
    (
        frame_idx_out,
        out_obj_ids,
        _,
        video_res_masks,
    ) = PREDICTOR.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=torch.tensor(rel_points, dtype=torch.float32),
        labels=torch.tensor(labels_arr, dtype=torch.int32),
        clear_old_points=False,
        rel_coordinates=True,
        use_prev_mem_frame=False,
    )

    # Build a mask for the current frame and return it as a zipped NIfTI file.
    if video_res_masks is not None and len(video_res_masks) > 0:
        mask_logits = video_res_masks.detach().cpu().numpy()  # (N, 1, H, W)
        _, _, H, W = mask_logits.shape
        mask = np.zeros((H, W), dtype=np.float32)

        for i, oid in enumerate(out_obj_ids):
            mask += (mask_logits[i, 0] > 0).astype(np.float32) * (1 + int(oid))

        non_zero_count = np.count_nonzero(mask)
        print(f"Number of non-zero values in the mask: {non_zero_count}")
        print("logits_sum: ", np.sum(mask_logits))

        # Create a 3D volume with a singleton first dimension for NIfTI.
        mask_volume = mask[np.newaxis, ...].astype(np.float32)
    else:
        # Fallback: create an empty mask based on the current frame size.
        jpg_dir = os.path.join(session_state["temp_dir"], "jpgs")
        frame_path = os.path.join(jpg_dir, f"{frame_idx}.jpg")
        with Image.open(frame_path) as img:
            width, height = img.size
        mask_volume = np.zeros((1, height, width), dtype=np.float32)

    affine = np.eye(4, dtype=np.float32)
    nii_img = nib.Nifti1Image(mask_volume, affine)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
        temp_file_path = temp_file.name
        nib.save(nii_img, temp_file_path)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip_file:
        temp_zip_file_path = temp_zip_file.name
        with zipfile.ZipFile(temp_zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(temp_file_path, arcname="masks.nii.gz")

    with open(temp_zip_file_path, "rb") as f:
        zip_file_content = f.read()

    os.remove(temp_file_path)
    os.remove(temp_zip_file_path)

    set_or_reset_timer(session_id)

    return send_file(
        BytesIO(zip_file_content),
        download_name="masks.zip",
        as_attachment=True,
        mimetype="application/octet-stream",
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
    inference_state = state_info["inference_state"]
    num_frames = inference_state.get("num_frames", state_info["n_frames"])

    # Run full-video propagation using the tracker, mirroring example 1.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE.type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        for (
            frame_idx,
            obj_ids,
            _,
            video_res_masks,
            obj_scores,
        ) in PREDICTOR.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=num_frames,
            reverse=False,
            tqdm_disable=True,
            propagate_preflight=True,
        ):
            # Store masks per frame and object id for downstream use.
            video_segments[frame_idx] = {
                int(obj_id): (video_res_masks[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

    # Clean up temporary files associated with this session.
    temp_dir = state_info.get("temp_dir")
    if temp_dir:
        try:
            shutil.rmtree(temp_dir)
        except FileNotFoundError:
            pass

    # Convert masks to NIfTI and return as zipped binary content.
    nii_img = convert_masks_to_nii(video_segments, state_info["n_frames"])

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
        temp_file_path = temp_file.name
        nib.save(nii_img, temp_file_path)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip_file:
        temp_zip_file_path = temp_zip_file.name
        with zipfile.ZipFile(temp_zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(temp_file_path, arcname="masks.nii.gz")

    with open(temp_zip_file_path, "rb") as f:
        zip_file_content = f.read()

    os.remove(temp_file_path)
    os.remove(temp_zip_file_path)

    set_or_reset_timer(session_id)

    return send_file(
        BytesIO(zip_file_content),
        download_name="masks.nii.gz",
        as_attachment=True,
        mimetype="application/octet-stream",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
