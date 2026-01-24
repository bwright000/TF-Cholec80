"""
Preprocessing module for MPHY0043 Coursework.
Extracts phase timing information from Cholec80 dataset.

Works with RAW data format (PNG frames + text annotations).

Expected directory structure:
    cholec80/
    ├── frames/
    │   └── videoXX/  (contains PNG frames: 0.png, 25.png, ...)
    ├── phase_annotations/
    │   └── videoXX-phase.txt
    └── tool_annotations/
        └── videoXX-tool.txt

This module processes the raw Cholec80 data to compute:
- Remaining time in current phase
- Remaining time in surgery
- Future phase start times
- Phase statistics (mean durations, transitions)
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Constants
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

# Standard data splits (1-indexed video IDs)
TRAIN_VIDEO_IDS = list(range(1, 33))    # Videos 01-32
VAL_VIDEO_IDS = list(range(33, 41))     # Videos 33-40
TEST_VIDEO_IDS = list(range(41, 81))    # Videos 41-80


# ============================================================================
# RAW DATA LOADING
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


def get_video_frame_ids(frames_dir: str, video_id: int) -> List[int]:
    """
    Get list of frame IDs for a video by scanning the frames directory.

    Args:
        frames_dir: Base frames directory
        video_id: Video ID (1-80)

    Returns:
        Sorted list of frame IDs (1-indexed, from filenames like video01_000001.png)
    """
    video_folder = os.path.join(frames_dir, f'video{video_id:02d}')

    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder not found: {video_folder}")

    frame_ids = []
    video_prefix = f'video{video_id:02d}_'

    for filename in os.listdir(video_folder):
        if filename.endswith('.png'):
            # Handle format: video01_000001.png -> extract 000001 -> 1
            if filename.startswith(video_prefix):
                num_part = filename[len(video_prefix):-4]  # Remove prefix and .png
                try:
                    frame_id = int(num_part)
                    frame_ids.append(frame_id)
                except ValueError:
                    pass
            else:
                # Fallback: try old format (just number.png)
                try:
                    frame_id = int(filename.replace('.png', ''))
                    frame_ids.append(frame_id)
                except ValueError:
                    pass

    return sorted(frame_ids)


def extract_video_data(data_dir: str, video_id: int) -> Dict:
    """
    Extract all frame data from a single video using raw data format.

    Args:
        data_dir: Path to Cholec80 directory
        video_id: Video ID (1-80, 1-indexed)

    Returns:
        Dictionary containing:
            - frame_ids: array of frame IDs (1-indexed from filenames)
            - phases: array of phase indices
            - instruments: array of tool vectors (N, 7)

    Note on annotation mapping:
        - Frame files: video01_000001.png, video01_000002.png, ... (1-indexed, at 1fps)
        - Phase annotations: at 25fps (indices 0, 1, 2, ..., ~43000 for a ~30min video)
        - Tool annotations: at 1fps intervals (0, 25, 50, ... in annotation indices)
        - Frame file N corresponds to annotation index (N-1) * 25
    """
    frames_dir = os.path.join(data_dir, 'frames')
    phase_dir = os.path.join(data_dir, 'phase_annotations')
    tool_dir = os.path.join(data_dir, 'tool_annotations')

    # Get frame IDs from the frames directory (1-indexed from filenames)
    frame_ids = get_video_frame_ids(frames_dir, video_id)

    # Load annotations
    phase_file = os.path.join(phase_dir, f'video{video_id:02d}-phase.txt')
    tool_file = os.path.join(tool_dir, f'video{video_id:02d}-tool.txt')

    frame_to_phase = parse_phase_annotations(phase_file)
    frame_to_tools = parse_tool_annotations(tool_file)

    # Build arrays aligned with frame_ids
    # Frame files are at 1fps (subsampled from 25fps video)
    # Frame file N (1-indexed) corresponds to annotation index (N-1) * 25
    # Example: frame 1 -> annotation 0, frame 2 -> annotation 25, frame 3 -> annotation 50
    phases = []
    instruments = []

    for frame_id in frame_ids:
        # Map 1-indexed frame file to 25fps annotation index
        # Frame 1 -> index 0, Frame 2 -> index 25, Frame 3 -> index 50, etc.
        annotation_idx = (frame_id - 1) * 25

        # Get phase (default to 0 if missing)
        phase = frame_to_phase.get(annotation_idx, 0)
        phases.append(phase)

        # Tool annotations are also at these 25-frame intervals
        tool = frame_to_tools.get(annotation_idx, np.zeros(NUM_TOOLS, dtype=np.int32))
        instruments.append(tool)

    return {
        'frame_ids': np.array(frame_ids, dtype=np.int32),
        'phases': np.array(phases, dtype=np.int32),
        'instruments': np.array(instruments, dtype=np.int32)
    }


# ============================================================================
# PHASE BOUNDARY COMPUTATION
# ============================================================================

def compute_phase_boundaries(phases: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find where each phase starts and ends in a video.

    Args:
        phases: Array of phase labels for each frame (e.g., [0,0,0,1,1,1,2,2,...])

    Returns:
        List of tuples: (phase_id, start_frame_idx, end_frame_idx)

    Example:
        If phases = [0,0,0,1,1,2,2,2,2]
        Returns: [(0, 0, 2), (1, 3, 4), (2, 5, 8)]

        This means:
        - Phase 0 runs from index 0 to index 2
        - Phase 1 runs from index 3 to index 4
        - Phase 2 runs from index 5 to index 8
    """
    if len(phases) == 0:
        return []

    current_phase = phases[0]
    boundaries = []
    start_idx = 0

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            # Transition detected
            boundaries.append((int(current_phase), start_idx, i - 1))
            # Start tracking the new phase
            current_phase = phases[i]
            start_idx = i

    # Add the final phase segment
    boundaries.append((int(current_phase), start_idx, len(phases) - 1))

    return boundaries


# ============================================================================
# TIMING LABEL COMPUTATION
# ============================================================================

def compute_timing_labels(phases: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute timing labels for each frame in a video.
    These are the TARGET values our model will learn to predict.

    Args:
        phases: Array of phase labels for each frame

    Returns:
        Dictionary containing:
        - remaining_phase: frames until current phase ends
        - remaining_surgery: frames until video ends
        - elapsed_phase: frames since current phase started
        - phase_progress: normalized progress in current phase (0.0 to 1.0)
        - future_phase_starts: frames until each phase starts (-1 if already passed)
    """
    n_frames = len(phases)

    # Get phase boundaries
    boundaries = compute_phase_boundaries(phases)

    # Create a lookup table: frame_idx -> boundary_idx
    frame_to_boundary = np.zeros(n_frames, dtype=int)
    for b_idx, (phase_id, start, end) in enumerate(boundaries):
        frame_to_boundary[start:end + 1] = b_idx

    # Initialize output arrays
    remaining_phase = np.zeros(n_frames, dtype=np.int32)
    remaining_surgery = np.zeros(n_frames, dtype=np.int32)
    elapsed_phase = np.zeros(n_frames, dtype=np.int32)
    phase_progress = np.zeros(n_frames, dtype=np.float32)
    future_phase_starts = np.full((n_frames, NUM_PHASES), -1, dtype=np.int32)

    # Compute labels for each frame
    for frame_idx in range(n_frames):
        # Remaining frames left in surgery
        remaining_surgery[frame_idx] = (n_frames - 1) - frame_idx

        # Find the boundary for this frame
        b_idx = frame_to_boundary[frame_idx]
        phase_id, start, end = boundaries[b_idx]

        # Frames left in current phase
        remaining_phase[frame_idx] = end - frame_idx

        # Frames since the phase started
        elapsed_phase[frame_idx] = frame_idx - start

        # Phase progress (0.0 at start, 1.0 at end)
        phase_duration = end - start + 1
        if phase_duration > 1:
            phase_progress[frame_idx] = (frame_idx - start) / (phase_duration - 1)
        else:
            phase_progress[frame_idx] = 1.0  # Single-frame phase

        # Calculate future phase start times
        for future_b_idx in range(b_idx + 1, len(boundaries)):
            future_phase_id, future_start, _ = boundaries[future_b_idx]
            # Record only the first instance of a future phase
            if future_phase_starts[frame_idx, future_phase_id] == -1:
                future_phase_starts[frame_idx, future_phase_id] = future_start - frame_idx

    return {
        'remaining_phase': remaining_phase,
        'remaining_surgery': remaining_surgery,
        'elapsed_phase': elapsed_phase,
        'phase_progress': phase_progress,
        'future_phase_starts': future_phase_starts
    }


# ============================================================================
# PHASE STATISTICS
# ============================================================================

def compute_phase_statistics(all_video_data: Dict[int, Dict]) -> Dict:
    """
    Compute statistics about phase durations across all videos.
    Useful for understanding the dataset and for normalization.

    Args:
        all_video_data: Dictionary mapping video_id to video data
                        Each video data has 'phases' array

    Returns:
        Dictionary containing statistics for each phase
    """
    # Collect all phase durations
    phase_durations = {p: [] for p in range(NUM_PHASES)}

    for video_id, data in all_video_data.items():
        boundaries = compute_phase_boundaries(data['phases'])

        for phase_id, start, end in boundaries:
            duration = end - start + 1  # +1 because end is inclusive
            phase_durations[phase_id].append(duration)

    # Compute statistics for each phase
    stats = {
        'phase_names': PHASE_NAMES,
        'num_videos': len(all_video_data),
        'durations': {}
    }

    for phase_id in range(NUM_PHASES):
        phase_name = PHASE_NAMES[phase_id]
        durations = phase_durations[phase_id]

        if len(durations) > 0:
            stats['durations'][phase_name] = {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': int(np.min(durations)),
                'max': int(np.max(durations)),
                'count': len(durations)  # How many times this phase appeared
            }
        else:
            stats['durations'][phase_name] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0,
                'count': 0
            }

    return stats


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def preprocess_dataset(
    data_dir: str,
    video_ids: Optional[List[int]] = None,
    output_dir: str = 'mphy0043_cw/results'
) -> Tuple[str, str]:
    """
    Main function to preprocess the entire dataset.
    Extracts timing labels for all videos and saves them.

    Args:
        data_dir: Path to Cholec80 directory (with frames/, phase_annotations/, tool_annotations/)
        video_ids: List of video IDs to process (default: all 80, 1-indexed)
        output_dir: Directory to save output files

    Returns:
        Tuple of (timing_labels_path, statistics_path)
    """
    # Default to all 80 videos (1-indexed)
    if video_ids is None:
        video_ids = list(range(1, 81))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Preprocessing {len(video_ids)} videos...")
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Step 1: Load data from all videos
    all_video_data = {}
    all_timing_labels = {}

    for video_id in tqdm(video_ids, desc="Processing videos"):
        try:
            # Extract raw data
            video_data = extract_video_data(data_dir, video_id)
            all_video_data[video_id] = video_data

            # Compute timing labels
            timing = compute_timing_labels(video_data['phases'])

            # Also store original data for reference
            timing['phases'] = video_data['phases']
            timing['instruments'] = video_data['instruments']
            timing['frame_ids'] = video_data['frame_ids']

            all_timing_labels[video_id] = timing

        except Exception as e:
            print(f"\nError processing video {video_id}: {e}")
            continue

    # Step 2: Compute statistics across all videos
    print("\nComputing phase statistics...")
    stats = compute_phase_statistics(all_video_data)

    # Step 3: Save timing labels to .npz file
    # Flatten the nested structure for numpy saving
    save_dict = {}
    for video_id, labels in all_timing_labels.items():
        for key, value in labels.items():
            save_dict[f"video_{video_id}_{key}"] = value

    timing_path = os.path.join(output_dir, 'timing_labels.npz')
    np.savez_compressed(timing_path, **save_dict)
    print(f"Saved timing labels to: {timing_path}")

    # Step 4: Save statistics to JSON file
    stats_path = os.path.join(output_dir, 'phase_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("PHASE DURATION SUMMARY (in frames, at 1 fps)")
    print("=" * 60)
    for phase_name in PHASE_NAMES:
        d = stats['durations'][phase_name]
        if d['count'] > 0:
            mean_min = d['mean'] / 60  # Convert to minutes
            std_min = d['std'] / 60
            print(f"{phase_name:30s}: {d['mean']:7.1f} ± {d['std']:6.1f} frames "
                  f"({mean_min:5.1f} ± {std_min:4.1f} min)")
        else:
            print(f"{phase_name:30s}: No instances found")

    print(f"\nTotal videos processed: {len(all_timing_labels)}")

    return timing_path, stats_path


def load_timing_labels(timing_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load preprocessed timing labels from a .npz file.

    Args:
        timing_path: Path to timing_labels.npz

    Returns:
        Dictionary mapping video_id to timing labels

    Example usage:
        labels = load_timing_labels('results/timing_labels.npz')
        video_1_remaining = labels[1]['remaining_phase']
    """
    data = np.load(timing_path)

    # Reconstruct the nested dictionary structure
    all_labels = {}

    for key in data.files:
        # Key format: "video_{id}_{field}"
        # Example: "video_1_remaining_phase"
        parts = key.split('_')
        video_id = int(parts[1])
        field = '_'.join(parts[2:])  # Handle fields with underscores

        if video_id not in all_labels:
            all_labels[video_id] = {}

        all_labels[video_id][field] = data[key]

    return all_labels


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess Cholec80 dataset for timing prediction'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to Cholec80 directory (with frames/, phase_annotations/, tool_annotations/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='mphy0043_cw/results',
        help='Directory to save output files (default: mphy0043_cw/results)'
    )
    parser.add_argument(
        '--video_ids',
        type=int,
        nargs='+',
        default=None,
        help='Specific video IDs to process (default: all 80)'
    )

    args = parser.parse_args()

    # Run preprocessing
    preprocess_dataset(
        data_dir=args.data_dir,
        video_ids=args.video_ids,
        output_dir=args.output_dir
    )

# Terminal script to run this: python -m mphy0043_cw.data.preprocessing --data_dir /path/to/cholec80 --output_dir mphy0043_cw/results
