"""
TFRecord Conversion for Cholec80 Dataset
Author: ChatGPT
Authorised: 22/01/2026
"""

import os
import glob
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Constants ---
PHASE_NAMES = [
    'Preparation',
    'CalotTriangleDissection',
    'ClippingCutting',
    'GallbladderDissection',
    'GallbladderRetraction',
    'CleaningCoagulation',
    'GallbladderPackaging'
]
NUM_PHASES = len(PHASE_NAMES)
NUM_TOOLS = 7
phase_to_id = {name: i for i, name in enumerate(PHASE_NAMES)}

# Paths (adjust as needed)
DATASET_DIR = r"F:\2026 vibes\MPHY0043CW2\Dataset\cholec80"
FRAMES_DIR = os.path.join(DATASET_DIR, "frames")
PHASE_DIR = os.path.join(DATASET_DIR, "phase_annotations")
TOOL_DIR = os.path.join(DATASET_DIR, "tool_annotations")
OUTPUT_DIR = os.path.join(DATASET_DIR, "tfrecords")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def load_image(path):
    """Load an image and return bytes"""
    img = Image.open(path).convert("RGB")
    return img.tobytes()

def load_phase_annotations(file_path, num_frames):
    """Load phase IDs per frame (skip header)"""
    phase_ids = np.zeros(num_frames, dtype=np.int64)
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            frame_no, phase_name = parts
            frame_idx = int(frame_no)
            if frame_idx >= num_frames:
                continue  # safety check
            phase_id = phase_to_id.get(phase_name)
            if phase_id is None:
                raise ValueError(f"Unknown phase name: {phase_name}")
            phase_ids[frame_idx] = phase_id
    return phase_ids

def load_tool_annotations(file_path, num_frames):
    """Load tool usage per frame with forward-fill interpolation (skip header)"""
    tools_array = np.zeros((num_frames, NUM_TOOLS), dtype=np.int64)
    last_state = np.zeros(NUM_TOOLS, dtype=np.int64)
    last_frame = 0
    with open(file_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split("\t")
        if len(header) != NUM_TOOLS + 1:
            raise ValueError(f"Unexpected tool annotation header: {header}")
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) != NUM_TOOLS + 1:
                continue
            frame_idx = int(parts[0])
            state = np.array([int(x) for x in parts[1:]], dtype=np.int64)
            # Fill frames from last_frame up to current frame
            tools_array[last_frame:frame_idx] = last_state
            last_state = state
            last_frame = frame_idx
        # Fill remaining frames
        tools_array[last_frame:] = last_state
    return tools_array

def create_tfrecord(video_id, frames, phases, instruments, output_dir):
    """Write a single video TFRecord"""
    tfrecord_path = os.path.join(output_dir, f"video{video_id:02d}.tfrecord")
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, frame_path in enumerate(frames):
            img_bytes = load_image(frame_path)
            phase_id = int(phases[i])
            tools = instruments[i].tolist()
            feature = {
                "frame": bytes_feature(img_bytes),
                "video_id": int64_feature(video_id),
                "frame_id": int64_feature(i),
                "total_frames": int64_feature(len(frames)),
                "phase": int64_feature(phase_id),
                "instruments": int64_list_feature(tools)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"Saved TFRecord for video {video_id:02d} -> {tfrecord_path}")

# --- Main Execution ---
if __name__ == "__main__":
    for video_folder in sorted(os.listdir(FRAMES_DIR)):
        if not video_folder.startswith("video"):
            continue
        video_id = int(video_folder.replace("video", ""))
        frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, video_folder, "*.png")))
        num_frames = len(frame_files)

        phase_file = os.path.join(PHASE_DIR, f"{video_folder}-phase.txt")
        tool_file = os.path.join(TOOL_DIR, f"{video_folder}-tool.txt")

        # Skip videos with missing frames or annotations
        if num_frames == 0:
            print(f"Skipping video {video_id:02d}: no frames found")
            continue
        if not os.path.exists(phase_file) or not os.path.exists(tool_file):
            print(f"Skipping video {video_id:02d}: missing annotation")
            continue

        phases = load_phase_annotations(phase_file, num_frames)
        instruments = load_tool_annotations(tool_file, num_frames)

        create_tfrecord(video_id, frame_files, phases, instruments, OUTPUT_DIR)

    print("All videos converted to TFRecords successfully!")
