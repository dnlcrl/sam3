import glob
import os
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import requests
from PIL import Image

# Server URL (adjust if needed)
server_url = "http://localhost:5000"  # change port number if needed

# Path to the video images directory
video_images_dir = "examples/images/video/"
output_images_dir = "examples/images/output/"  # Directory to save overlaid images
wwwl = {"ww": 300, "wl": 40}
output_nii_dir = "./output_nii_files"


def convert_dicoms_to_jpgs(zip_file_path, video_images_dir, ww=None, wl=None):
    # Check if the file exists and has a non-zero size
    if not os.path.isfile(zip_file_path) or os.path.getsize(zip_file_path) == 0:
        raise ValueError("Invalid file or file size is 0!")

    # Create temporary directories for extracted DICOMs
    temp_dir = tempfile.mkdtemp()
    dcm_dir = os.path.join(temp_dir, "dicoms")
    os.makedirs(dcm_dir, exist_ok=True)
    os.makedirs(video_images_dir, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path) as zip_ref:
        zip_ref.extractall(dcm_dir)

    # Load DICOM files
    dicom_filenames = glob.glob(os.path.join(dcm_dir, "*.dcm"))
    files = [
        pydicom.dcmread(fname) for fname in dicom_filenames if fname.endswith(".dcm")
    ]

    # Sort slices based on DICOM metadata
    def sort_key(s):
        if hasattr(s, "SliceLocation"):
            return s.SliceLocation
        elif hasattr(s, "InstanceNumber"):
            return s.InstanceNumber
        elif hasattr(s, "ImagePositionPatient"):
            return s.ImagePositionPatient[2]
        else:
            return float("inf")  # Put files with no relevant tag at the end

    slices = sorted(files, key=sort_key)

    # Prepare 3D array of pixel data
    img_shape = list(slices[0].pixel_array.shape) + [len(slices)]
    img3d = np.zeros(img_shape)

    # Fill the 3D array with pixel data, applying normalization and WW/WL adjustments
    for i, s in enumerate(slices):
        img2d = s.pixel_array.astype(np.float32)
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)
        img2d = img2d * slope + intercept

        if ww and wl:
            ww = float(ww)
            wl = float(wl)
            min_val = wl - ww / 2
            max_val = wl + ww / 2
            img2d = np.clip(img2d, min_val, max_val)
            img2d = (img2d - min_val) / (max_val - min_val) * 255

        img3d[:, :, i] = img2d

    # Normalize the 3D array
    non_zero_values = img3d[img3d != 0]
    min_val = int(np.min(non_zero_values)) + 100
    max_val = int(0.67 * np.max(non_zero_values))
    img3d_normalized = np.clip(img3d, min_val, max_val)
    img3d_normalized = 255 * (img3d_normalized - min_val) / (max_val - min_val)
    img3d_normalized = img3d_normalized.astype(np.uint8)

    # Convert slices to JPG and save in video_images_dir
    for idx in range(img3d_normalized.shape[2]):
        image_array = img3d_normalized[:, :, idx]
        image = Image.fromarray(image_array).convert("L")
        image.save(os.path.join(video_images_dir, f"{idx}.jpg"), quality=100)

    # Cleanup temporary directories
    print(f"Converted DICOM files to JPGs in {video_images_dir}")


def initialize_video():
    # Specify the zip file containing the DICOM data
    zip_file_path = os.path.join(os.getcwd(), "test_data.zip")

    # Ensure the file exists
    if not os.path.exists(zip_file_path):
        print(f"Zip file not found at {zip_file_path}")
        return None

    # Print file size before sending to ensure it's non-zero
    file_size = os.path.getsize(zip_file_path)
    print(f"File size before sending: {file_size}")

    if file_size == 0:
        print("The zip file is empty!")
        return None

    # Prepare the file to be sent in the request
    try:
        with open(zip_file_path, "rb") as f:
            files = {"data_binary": f}
            # Send the request to the server
            response = requests.post(
                f"{server_url}/initialize_video", files=files, data=wwwl
            )
        print(response.content)
        # Handle the response
        if response.status_code == 200:
            print("Video initialized successfully!")
            session_id = response.json()["session_id"]
            if session_id is None:
                raise Exception("session id not found")
            return session_id
        else:
            print(f"Failed to initialize video: {response.text}")
            return None

    except Exception as e:
        print(f"Error while opening or sending file: {e}")
        return None


def visualize_overlay_from_nii(nii_file_path, frame_idx):
    """
    Loads and visualizes the overlay mask from a NIfTI file on the input image.
    """
    # Load the NIfTI image
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()

    mask_slice = nii_data[0, :, :]
    # Load the input image
    input_image_path = os.path.join(
        video_images_dir, f"{frame_idx}.jpg"
    )  # Assuming first frame is '0.jpg'
    input_image = Image.open(input_image_path)
    input_image_np = np.array(input_image)

    # Overlay the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(input_image_np)
    plt.imshow(mask_slice, alpha=0.5, cmap="jet")  # Overlay with transparency
    plt.title("Input Image with Overlayed Mask from NIfTI")
    plt.axis("off")
    plt.show()


import io
import zipfile


# Step 2: Test the '/add_points' endpoint (positive and negative clicks)
def add_points(session_id, input_points, labels, object_id=0, frame_idx=0):

    data = {
        "session_id": session_id,
        "points": [[input_points]],  # Format: [[x1, y1], [x2, y2], ...]
        "labels": [
            [labels]
        ],  # Format: [label1, label2, ...] where 1 = positive, 0 = negative
        "obj_id": object_id,
        "frame_idx": frame_idx,
    }

    response = requests.post(f"{server_url}/add_points", data=data)

    if response.status_code == 200:
        # Set up paths
        zip_bytes = io.BytesIO(response.content)
        output_nii_dir = f"output/{session_id}/niftis"  # Example output directory
        os.makedirs(output_nii_dir, exist_ok=True)

        # Extract and save NIfTI files sequentially
        nii_file_paths = []
        with zipfile.ZipFile(zip_bytes, "r") as zip_file:
            nifti_files_namelist = zip_file.namelist()
            print(nifti_files_namelist)
            for nifti_filename in nifti_files_namelist:
                nii_file_path = os.path.join(output_nii_dir, nifti_filename)
                with zip_file.open(nifti_filename) as nifti_data:
                    with open(nii_file_path, "wb") as f:
                        f.write(nifti_data.read())
                nii_file_paths.append(nii_file_path)

        # Visualize overlay mask from the NIfTI file
        for p in nii_file_paths:
            visualize_overlay_from_nii(p, frame_idx)

    else:
        print(f"Failed to add points: {response.text}")


# Function to display masks over the frame image
def show_mask(mask, ax, obj_id=None):
    """Display a mask over the current frame."""
    mask = np.array(mask[0])  # Convert mask to numpy array if needed
    ax.imshow(mask, alpha=0.5, cmap="jet")  # Overlay mask with some transparency
    if obj_id is not None:
        ax.text(10, 10, f"Object {obj_id}", bbox=dict(facecolor="yellow", alpha=0.5))


def create_overlay_video(session_id, nii_files_dir):
    nifti_files = sorted(
        [f for f in os.listdir(nii_files_dir) if f.endswith(".nii.gz")]
    )
    overlay_images = []

    for i, nii_filename in enumerate(nifti_files):
        nii_file_path = os.path.join(nii_files_dir, nii_filename)
        nii_img = nib.load(nii_file_path)
        nii_data = nii_img.get_fdata()

        # Apply transformations to the NIfTI data for correct orientation
        nii_data = nii_data[:, 0, :, :]  # Reversing and selecting slice
        nii_data = np.transpose(
            nii_data, (1, 2, 0)
        )  # Transpose for [height, width, slices]
        num_slices = nii_data.shape[-1]
        print("number of sliices: ", num_slices)

        for slice_idx in range(num_slices):
            input_image_path = os.path.join(video_images_dir, f"{slice_idx}.jpg")
            input_image = Image.open(input_image_path)
            input_image_np = np.array(input_image)

            # Extract mask for the current slice
            mask_slice = nii_data[:, :, slice_idx]

            # Create overlay
            plt.figure(figsize=(10, 10))
            plt.imshow(input_image_np)
            plt.imshow(mask_slice, alpha=0.5, cmap="jet")  # Overlay with transparency
            plt.axis("off")

            # Save the overlay image
            output_overlay_path = os.path.join(
                output_images_dir, f"overlay_{slice_idx}.png"
            )
            plt.savefig(output_overlay_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            overlay_images.append(output_overlay_path)

    # Create video from overlay images
    save_overlay_video(overlay_images)


def save_overlay_video(overlay_images):
    images = [imageio.imread(img) for img in overlay_images]

    # Save as video
    video_path = "output_overlay_video.mp4"
    imageio.mimsave(video_path, images, fps=10)  # Save video with 10 fps
    print(f"Overlay video saved at {video_path}")


# Step 3: Test the '/propagate_masks' endpoint
def propagate_masks(session_id):
    data = {"session_id": session_id}

    # Send request to propagate masks
    response = requests.post(f"{server_url}/propagate_masks", data=data)

    if response.status_code == 200:
        # Initialize zip handling with response content
        zip_bytes = io.BytesIO(response.content)
        nii_files_dir = os.path.join(output_nii_dir, session_id)
        os.makedirs(nii_files_dir, exist_ok=True)

        with zipfile.ZipFile(zip_bytes, "r") as zip_file:
            nifti_files_namelist = zip_file.namelist()
            for file_idx, nifti_filename in enumerate(nifti_files_namelist):
                output_path = os.path.join(nii_files_dir, nifti_filename)
                with zip_file.open(nifti_filename) as nifti_data:
                    with open(output_path, "wb") as f:
                        f.write(nifti_data.read())

        print(f"Masks propagated successfully and saved to {nii_files_dir}")

        # Create overlay video from the extracted NIfTI files
        create_overlay_video(session_id, nii_files_dir)

    else:
        print(f"Failed to propagate masks: {response.text}")


# Natural sort function using regex to extract numbers from filenames
def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text for text in re.split(r"(\d+)", filename)
    ]


# Test flow
if __name__ == "__main__":
    # clear output files
    convert_dicoms_to_jpgs("test_data.zip", video_images_dir, wwwl["ww"], wwwl["wl"])
    image_files = [
        os.path.join(video_images_dir, f)
        for f in sorted(
            [f for f in os.listdir(video_images_dir) if "jpg" in f],
            key=lambda x: int(x.split(".")[0]),
        )
        if f.endswith(".jpg")
    ]

    [
        os.remove(f)
        for f in glob.glob(os.path.join(output_nii_dir, "*"))
        if os.path.isfile(f)
    ]
    if os.path.isfile("output_video.mp4"):
        os.remove("output_video.mp4")

    # Initialize the video
    session_id = initialize_video()

    if session_id:
        # Add points (Example: Adding two points with positive and negative labels)
        input_points = [[200, 250]]
        labels = [1]  # Positive (1) and Negative (0)
        add_points(session_id, input_points, labels, object_id=0, frame_idx=3)
        # input_points = [[260, 50]]
        # labels = [1]  # Positive (1) and Negative (0)
        # add_points(session_id, input_points, labels,  frame_idx=3)
        # breakpoint()
        # Propagate masks
        propagate_masks(session_id)

        # After propagating masks, you can use the saved voerlaid images to create a video using imageio

        # Load image files
        image_files = sorted(
            glob.glob(os.path.join(output_nii_dir, "overlay_*.png")),
            key=natural_sort_key,
        )

        # Ensure images exist
        if image_files:
            images = [imageio.imread(filename) for filename in image_files]

            # Save as video
            video_path = "output_video.mp4"
            imageio.mimsave(video_path, images, fps=10)  # Save video with 10 fps
            print(f"Video saved at {video_path}")
        else:
            print("No images found to create the video.")
