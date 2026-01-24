"""
Dataset loader for MPHY0043 Coursework.

Loads Cholec80 data from raw format (PNG frames + text annotations).

Expected directory structure:
    cholec80/
    ├── frames/
    │   ├── video01/
    │   │   ├── video01_000001.png
    │   │   ├── video01_000002.png
    │   │   └── ...
    │   ├── video02/
    │   └── ...
    ├── phase_annotations/
    │   ├── video01-phase.txt  (or video03-phase.txt, etc.)
    │   ├── video02-phase.txt
    │   └── ...
    └── tool_annotations/
        ├── video01-tool.txt
        ├── video02-tool.txt
        └── ...

Frame naming: videoXX_NNNNNN.png (sequential frame numbers starting at 1)
Phase annotations: "Frame\\tPhase" where Frame is 0, 1, 2, ... (1fps sequential)
Tool annotations: "Frame\\tGrasper\\t..." where Frame is 0, 25, 50, ... (25fps intervals)
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_PHASES = 7
NUM_TOOLS = 7

PHASE_NAMES = [
    'Preparation',
    'CalotTriangleDissection',
    'ClippingCutting',
    'GallbladderDissection',
    'GallbladderRetraction',
    'CleaningCoagulation',
    'GallbladderPackaging'
]

# Map phase names to indices
PHASE_TO_IDX = {name: idx for idx, name in enumerate(PHASE_NAMES)}

TOOL_NAMES = [
    'Grasper',
    'Bipolar',
    'Hook',
    'Scissors',
    'Clipper',
    'Irrigator',
    'SpecimenBag'
]

# Frame dimensions
FRAME_HEIGHT = 480
FRAME_WIDTH = 854
FRAME_CHANNELS = 3

# Standard data splits
TRAIN_VIDEO_IDS = list(range(1, 33))    # Videos 01-32 (1-indexed in filenames)
VAL_VIDEO_IDS = list(range(33, 41))     # Videos 33-40
TEST_VIDEO_IDS = list(range(41, 81))    # Videos 41-80


# ============================================================================
# ANNOTATION PARSING
# ============================================================================

def parse_phase_annotations(annotation_path: str) -> Dict[int, int]:
    """
    Parse phase annotation file.

    Args:
        annotation_path: Path to videoXX-phase.txt

    Returns:
        Dictionary mapping frame_id -> phase_idx

    File format:
        Frame	Phase
        0	Preparation
        25	Preparation
        ...
    """
    frame_to_phase = {}

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 2:
            frame_id = int(parts[0])
            phase_name = parts[1]

            # Convert phase name to index
            if phase_name in PHASE_TO_IDX:
                frame_to_phase[frame_id] = PHASE_TO_IDX[phase_name]
            else:
                print(f"Warning: Unknown phase '{phase_name}' at frame {frame_id}")

    return frame_to_phase


def parse_tool_annotations(annotation_path: str) -> Dict[int, np.ndarray]:
    """
    Parse tool annotation file.

    Args:
        annotation_path: Path to videoXX-tool.txt

    Returns:
        Dictionary mapping frame_id -> tool_array (7 binary values)

    File format:
        Frame	Grasper	Bipolar	Hook	Scissors	Clipper	Irrigator	SpecimenBag
        0	1	0	0	0	0	0	0
        25	1	0	0	0	0	0	0
        ...
    """
    frame_to_tools = {}

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) >= 8:  # Frame + 7 tools
            frame_id = int(parts[0])
            tools = np.array([int(parts[i]) for i in range(1, 8)], dtype=np.int32)
            frame_to_tools[frame_id] = tools

    return frame_to_tools


# ============================================================================
# VIDEO DATA LOADING
# ============================================================================

def get_video_frames(frames_dir: str, video_id: int) -> List[Tuple[int, str]]:
    """
    Get list of frame files for a video.

    Args:
        frames_dir: Base frames directory
        video_id: Video ID (1-80)

    Returns:
        List of (frame_id, frame_path) tuples, sorted by frame_id

    Handles filename format: videoXX_NNNNNN.png (e.g., video01_000001.png)
    The frame_id is the sequential number (1, 2, 3, ...) extracted from filename.
    """
    video_folder = os.path.join(frames_dir, f'video{video_id:02d}')

    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder not found: {video_folder}")

    frames = []
    video_prefix = f'video{video_id:02d}_'

    for filename in os.listdir(video_folder):
        if filename.endswith('.png'):
            # Handle format: video01_000001.png -> extract 000001 -> 1
            if filename.startswith(video_prefix):
                # Extract the number part after "videoXX_"
                num_part = filename[len(video_prefix):-4]  # Remove prefix and .png
                try:
                    frame_id = int(num_part)
                    frame_path = os.path.join(video_folder, filename)
                    frames.append((frame_id, frame_path))
                except ValueError:
                    print(f"Warning: Could not parse frame number from {filename}")
            else:
                # Fallback: try old format (just number.png)
                try:
                    frame_id = int(filename.replace('.png', ''))
                    frame_path = os.path.join(video_folder, filename)
                    frames.append((frame_id, frame_path))
                except ValueError:
                    print(f"Warning: Could not parse frame number from {filename}")

    # Sort by frame ID
    frames.sort(key=lambda x: x[0])
    return frames


def load_video_data(
    data_dir: str,
    video_id: int
) -> Dict[str, np.ndarray]:
    """
    Load all data for a single video.

    Args:
        data_dir: Base Cholec80 directory
        video_id: Video ID (1-80)

    Returns:
        Dictionary with:
            - frame_ids: array of frame IDs (sequential: 1, 2, 3, ...)
            - frame_paths: array of frame file paths
            - phases: array of phase indices
            - tools: array of tool vectors (N, 7)

    Note on annotation mapping:
        - Frame files: video01_000001.png, video01_000002.png, ... (1-indexed, at 1fps)
        - Phase annotations: at 25fps (indices 0, 1, 2, ..., ~43000 for a ~30min video)
        - Tool annotations: at 1fps intervals (0, 25, 50, ... in annotation indices)
        - Frame file N corresponds to annotation index (N-1) * 25
    """
    frames_dir = os.path.join(data_dir, 'frames')
    phase_dir = os.path.join(data_dir, 'phase_annotations')
    tool_dir = os.path.join(data_dir, 'tool_annotations')

    # Get frame list (frame_ids are 1-indexed from filenames)
    frames = get_video_frames(frames_dir, video_id)
    frame_ids = [f[0] for f in frames]
    frame_paths = [f[1] for f in frames]

    # Load annotations
    phase_file = os.path.join(phase_dir, f'video{video_id:02d}-phase.txt')
    tool_file = os.path.join(tool_dir, f'video{video_id:02d}-tool.txt')

    frame_to_phase = parse_phase_annotations(phase_file)
    frame_to_tools = parse_tool_annotations(tool_file)

    # Build arrays
    # Frame files are at 1fps (subsampled from 25fps video)
    # Frame file N (1-indexed) corresponds to annotation index (N-1) * 25
    # Example: frame 1 -> annotation 0, frame 2 -> annotation 25, frame 3 -> annotation 50
    phases = []
    tools = []

    for frame_id in frame_ids:
        # Map 1-indexed frame file to 25fps annotation index
        # Frame 1 -> index 0, Frame 2 -> index 25, Frame 3 -> index 50, etc.
        annotation_idx = (frame_id - 1) * 25

        # Get phase (default to 0 if missing)
        phase = frame_to_phase.get(annotation_idx, 0)
        phases.append(phase)

        # Tool annotations are also at these 25-frame intervals
        # so annotation_idx should directly match tool annotation indices
        tool = frame_to_tools.get(annotation_idx, np.zeros(NUM_TOOLS, dtype=np.int32))
        tools.append(tool)

    return {
        'video_id': video_id,
        'frame_ids': np.array(frame_ids, dtype=np.int32),
        'frame_paths': np.array(frame_paths),
        'phases': np.array(phases, dtype=np.int32),
        'tools': np.array(tools, dtype=np.int32)
    }


# ============================================================================
# TENSORFLOW DATASET
# ============================================================================

def create_dataset(
    data_dir: str,
    video_ids: List[int],
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    include_timing: bool = True,
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from raw Cholec80 data.

    Args:
        data_dir: Path to Cholec80 directory
        video_ids: List of video IDs to include
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        include_timing: Whether to include timing labels
        timing_labels: Pre-computed timing labels (from preprocessing.py)

    Returns:
        tf.data.Dataset yielding batches of:
            {
                'frame': (batch, H, W, 3) uint8
                'video_id': (batch,) int32
                'frame_id': (batch,) int32
                'phase': (batch,) int32
                'instruments': (batch, 7) int32
                'remaining_phase': (batch,) int32 (if include_timing)
                'remaining_surgery': (batch,) int32 (if include_timing)
                'elapsed_phase': (batch,) int32 (if include_timing)
                'phase_progress': (batch,) float32 (if include_timing)
            }
    """
    # Collect all samples
    all_samples = []

    for video_id in video_ids:
        try:
            video_data = load_video_data(data_dir, video_id)

            for i, frame_id in enumerate(video_data['frame_ids']):
                sample = {
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'frame_path': video_data['frame_paths'][i],
                    'phase': video_data['phases'][i],
                    'tools': video_data['tools'][i]
                }

                # Add timing labels if available
                if include_timing and timing_labels is not None:
                    if video_id in timing_labels:
                        labels = timing_labels[video_id]
                        # Find index for this frame
                        idx = i  # Assuming same ordering
                        if idx < len(labels['remaining_phase']):
                            sample['remaining_phase'] = labels['remaining_phase'][idx]
                            sample['remaining_surgery'] = labels['remaining_surgery'][idx]
                            sample['elapsed_phase'] = labels['elapsed_phase'][idx]
                            sample['phase_progress'] = labels['phase_progress'][idx]
                            sample['future_phase_starts'] = labels['future_phase_starts'][idx]

                all_samples.append(sample)

        except Exception as e:
            print(f"Warning: Could not load video {video_id}: {e}")
            continue

    print(f"Loaded {len(all_samples)} samples from {len(video_ids)} videos")

    # Create dataset from samples
    def generator():
        samples = all_samples.copy()
        if shuffle:
            np.random.shuffle(samples)
        for sample in samples:
            yield sample

    # Define output signature
    output_signature = {
        'video_id': tf.TensorSpec(shape=(), dtype=tf.int32),
        'frame_id': tf.TensorSpec(shape=(), dtype=tf.int32),
        'frame_path': tf.TensorSpec(shape=(), dtype=tf.string),
        'phase': tf.TensorSpec(shape=(), dtype=tf.int32),
        'tools': tf.TensorSpec(shape=(NUM_TOOLS,), dtype=tf.int32)
    }

    if include_timing and timing_labels is not None:
        output_signature.update({
            'remaining_phase': tf.TensorSpec(shape=(), dtype=tf.int32),
            'remaining_surgery': tf.TensorSpec(shape=(), dtype=tf.int32),
            'elapsed_phase': tf.TensorSpec(shape=(), dtype=tf.int32),
            'phase_progress': tf.TensorSpec(shape=(), dtype=tf.float32),
            'future_phase_starts': tf.TensorSpec(shape=(NUM_PHASES,), dtype=tf.int32)
        })

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Load and decode images
    def load_image(sample):
        # Read image file
        image_data = tf.io.read_file(sample['frame_path'])
        image = tf.io.decode_png(image_data, channels=3)
        image = tf.ensure_shape(image, [FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS])

        # Build output dictionary
        output = {
            'frame': image,
            'video_id': sample['video_id'],
            'frame_id': sample['frame_id'],
            'phase': sample['phase'],
            'instruments': sample['tools']  # Rename to match TF-Cholec80 convention
        }

        if 'remaining_phase' in sample:
            output['remaining_phase'] = sample['remaining_phase']
            output['remaining_surgery'] = sample['remaining_surgery']
            output['elapsed_phase'] = sample['elapsed_phase']
            output['phase_progress'] = sample['phase_progress']
            output['future_phase_starts'] = sample['future_phase_starts']

        return output

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    ds = ds.batch(batch_size)

    # Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ============================================================================
# SEQUENCE DATASET (For SSM Time Predictor)
# ============================================================================

def create_sequence_dataset(
    data_dir: str,
    video_ids: List[int],
    sequence_length: int = 64,
    batch_size: int = 4,
    shuffle: bool = True,
    stride: int = 32,
    include_timing: bool = True,
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset with sequences of consecutive frames.

    This is designed for the SSM-based time predictor which expects:
        - frames: (batch, seq_len, H, W, C)
        - elapsed_phase: (batch, seq_len, 1)
        - phase: (batch, seq_len)

    Args:
        data_dir: Path to Cholec80 directory
        video_ids: List of video IDs to include
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle sequences
        stride: Step size between sequence start positions (for overlap control)
        include_timing: Whether to include timing labels
        timing_labels: Pre-computed timing labels (from preprocessing.py)

    Returns:
        tf.data.Dataset yielding batches of:
            {
                'frames': (batch, seq_len, H, W, 3) uint8
                'elapsed_phase': (batch, seq_len, 1) float32
                'phase': (batch, seq_len) int32
                'remaining_phase': (batch, seq_len, 1) float32 (if include_timing)
                'future_phase_starts': (batch, seq_len, 6) float32 (if include_timing)
                'video_id': (batch,) int32
            }
    """
    # Collect all sequences
    all_sequences = []

    for video_id in video_ids:
        try:
            video_data = load_video_data(data_dir, video_id)
            n_frames = len(video_data['frame_ids'])

            # Skip if video is too short
            if n_frames < sequence_length:
                print(f"Warning: Video {video_id} has only {n_frames} frames, skipping")
                continue

            # Get timing labels for this video
            video_timing = None
            if include_timing and timing_labels is not None and video_id in timing_labels:
                video_timing = timing_labels[video_id]

            # Create sequences with stride
            for start_idx in range(0, n_frames - sequence_length + 1, stride):
                end_idx = start_idx + sequence_length

                sequence = {
                    'video_id': video_id,
                    'start_idx': start_idx,
                    'frame_paths': video_data['frame_paths'][start_idx:end_idx].tolist(),
                    'phases': video_data['phases'][start_idx:end_idx].tolist(),
                }

                # Add timing labels for the sequence
                if video_timing is not None:
                    sequence['elapsed_phase'] = video_timing['elapsed_phase'][start_idx:end_idx].tolist()
                    sequence['remaining_phase'] = video_timing['remaining_phase'][start_idx:end_idx].tolist()
                    sequence['future_phase_starts'] = video_timing['future_phase_starts'][start_idx:end_idx].tolist()

                all_sequences.append(sequence)

        except Exception as e:
            print(f"Warning: Could not load video {video_id}: {e}")
            continue

    print(f"Created {len(all_sequences)} sequences from {len(video_ids)} videos")
    print(f"  Sequence length: {sequence_length}, Stride: {stride}")

    # Create dataset from sequences
    def generator():
        sequences = all_sequences.copy()
        if shuffle:
            np.random.shuffle(sequences)
        for seq in sequences:
            yield seq

    # Define output signature
    output_signature = {
        'video_id': tf.TensorSpec(shape=(), dtype=tf.int32),
        'start_idx': tf.TensorSpec(shape=(), dtype=tf.int32),
        'frame_paths': tf.TensorSpec(shape=(sequence_length,), dtype=tf.string),
        'phases': tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
    }

    if include_timing and timing_labels is not None:
        output_signature.update({
            'elapsed_phase': tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
            'remaining_phase': tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
            'future_phase_starts': tf.TensorSpec(shape=(sequence_length, NUM_PHASES), dtype=tf.int32),
        })

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Load and decode images for entire sequence
    def load_sequence(seq):
        # Load all frames in sequence
        def load_single_frame(path):
            image_data = tf.io.read_file(path)
            image = tf.io.decode_png(image_data, channels=3)
            image = tf.ensure_shape(image, [FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS])
            return image

        # Map over all frame paths in sequence
        frames = tf.map_fn(
            load_single_frame,
            seq['frame_paths'],
            fn_output_signature=tf.TensorSpec(
                shape=[FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
                dtype=tf.uint8
            )
        )  # (seq_len, H, W, C)

        # Build output dictionary
        output = {
            'frames': frames,  # (seq_len, H, W, C)
            'video_id': seq['video_id'],
            'phase': seq['phases'],  # (seq_len,)
        }

        if 'elapsed_phase' in seq:
            # Reshape to (seq_len, 1) and cast to float32
            output['elapsed_phase'] = tf.cast(
                tf.expand_dims(seq['elapsed_phase'], axis=-1),
                tf.float32
            )
            output['remaining_phase'] = tf.cast(
                tf.expand_dims(seq['remaining_phase'], axis=-1),
                tf.float32
            )
            # Take phases 1-6 (not phase 0) for future starts
            output['future_phase_starts'] = tf.cast(
                seq['future_phase_starts'][:, 1:],  # (seq_len, 6)
                tf.float32
            )

        return output

    ds = ds.map(load_sequence, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    ds = ds.batch(batch_size)

    # Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_train_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """Get training dataset (videos 01-32)."""
    return create_dataset(
        data_dir=data_dir,
        video_ids=TRAIN_VIDEO_IDS,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        include_timing=True,
        timing_labels=timing_labels
    )


def get_val_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """Get validation dataset (videos 33-40)."""
    return create_dataset(
        data_dir=data_dir,
        video_ids=VAL_VIDEO_IDS,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        include_timing=True,
        timing_labels=timing_labels
    )


def get_test_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """Get test dataset (videos 41-80)."""
    return create_dataset(
        data_dir=data_dir,
        video_ids=TEST_VIDEO_IDS,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        include_timing=True,
        timing_labels=timing_labels
    )


# Sequence dataset convenience functions
def get_train_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 32,
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """Get training sequence dataset (videos 01-32)."""
    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=TRAIN_VIDEO_IDS,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True,
        stride=stride,
        include_timing=True,
        timing_labels=timing_labels
    )


def get_val_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 64,  # No overlap for validation
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """Get validation sequence dataset (videos 33-40)."""
    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=VAL_VIDEO_IDS,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        stride=stride,
        include_timing=True,
        timing_labels=timing_labels
    )


def get_test_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 64,  # No overlap for testing
    timing_labels: Optional[Dict] = None
) -> tf.data.Dataset:
    """Get test sequence dataset (videos 41-80)."""
    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=TEST_VIDEO_IDS,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        stride=stride,
        include_timing=True,
        timing_labels=timing_labels
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Cholec80 dataset loader')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Cholec80 directory')
    parser.add_argument('--video_id', type=int, default=1,
                        help='Video ID to test (default: 1)')

    args = parser.parse_args()

    print("Testing Cholec80 dataset loader...")
    print("=" * 60)

    # Test single video loading
    print(f"\n1. Loading video {args.video_id}...")
    video_data = load_video_data(args.data_dir, args.video_id)

    print(f"   Video ID: {video_data['video_id']}")
    print(f"   Number of frames: {len(video_data['frame_ids'])}")
    print(f"   Frame ID range: {video_data['frame_ids'][0]} - {video_data['frame_ids'][-1]}")
    print(f"   Phases present: {np.unique(video_data['phases'])}")
    print(f"   Tool presence (sum): {video_data['tools'].sum(axis=0)}")

    # Test dataset creation
    print(f"\n2. Creating dataset from videos [1, 2]...")
    ds = create_dataset(
        data_dir=args.data_dir,
        video_ids=[1, 2],
        batch_size=4,
        shuffle=False,
        include_timing=False
    )

    # Get one batch
    print("\n3. Loading first batch...")
    for batch in ds.take(1):
        print(f"   Batch keys: {list(batch.keys())}")
        print(f"   Frame shape: {batch['frame'].shape}")
        print(f"   Frame dtype: {batch['frame'].dtype}")
        print(f"   Video IDs: {batch['video_id'].numpy()}")
        print(f"   Frame IDs: {batch['frame_id'].numpy()}")
        print(f"   Phases: {batch['phase'].numpy()}")
        print(f"   Instruments shape: {batch['instruments'].shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
