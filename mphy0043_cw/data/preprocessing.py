"""
Preprocessing module for MPHY0043 Coursework.
Extracts phase timing information from Cholec80 dataset.

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
import tensorflow as tf
from tqdm import tqdm
# Add parent directory to the path to alloow for import tf_cholec80
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tf_cholec80.dataset import make_cholec80

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

TOOL_NAMES = [
    'Grasper',
    'Bipolar',
    'Hook',
    'Scissors',
    'Clipper',
    'Irrigator',
    'SpecimenBag'
]
# Data Extraction from a Single Video
def extract_video_data(video_id, batch_size=64):
    """
    Extracts all frame data from a single video

    Args:
        video_id (int): ID of the video to process.
        batch_size (int): Batch size for data extraction.
    Returns:
        dict: Dictionary containing extracted data arrays. 'frame_ids', 'phases', instruments'
    """
# Use infer to extract data sequentially (i.e., no shuffling)
    ds = make_cholec80(
        n_minibatch=batch_size,
        video_ids=[video_id],
        mode='INFER'
    )

# Collect data from each batch - First initialise arrays
    frame_ids =[]
    phases = []
    instruments = []

    for batch in ds:
    # Batch is a dictionary with keys: frame, video_id, frame_id,
    # total_frames, instruments, phase, end_flag
        frame_ids.extend(batch['frame_id'].numpy().tolist())
        phases.extend(batch['phase'].numpy().tolist())
        instruments.extend(batch['instruments'].numpy().tolist())

    return {
        'frame_ids': np.array(frame_ids),
        'phases': np.array(phases),
        'instruments': np.array(instruments)
    }

# Finding Phase Boundaries

def compute_phase_boundaries(phases):
    """
    Find where each phase starts and ends in a video.
    
    Args:
        phases: Array of phase labels for each frame (e.g., [0,0,0,1,1,1,2,2,...])
        
    Returns:
        List of tuples: (phase_id, start_frame, end_frame)
        
    Example:
        If phases = [0,0,0,1,1,2,2,2,2]
        Returns: [(0, 0, 2), (1, 3, 4), (2, 5, 8)]
        
        This means:
        - Phase 0 runs from frame 0 to frame 2
        - Phase 1 runs from frame 3 to frame 4
        - Phase 2 runs from frame 5 to frame 8
    """
    current_phase = phases[0]
    boundaries = []
    start_frame = 0
    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            # Transition detected
            boundaries.append((current_phase, start_frame, i - 1))
            # start tracking the new phase
            current_phase = phases[i]
            start_frame = i
    boundaries.append((current_phase, start_frame, len(phases) -1))
    return boundaries

# Compute frame timing labels

def compute_timing_labels(phases):
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
    """
    n_frames = len(phases)
    # Get phase boundaries
    boundaries = compute_phase_boundaries(phases)
    # Create a lookupu table to find which boundary each frame belongs to
    frame_to_boundary = np.zeros(n_frames, dtype=int)
    for b_idx, (phase_id, start, end) in enumerate(boundaries):
        frame_to_boundary[start:end + 1] = b_idx

    # Initalise outputu arrays
    remaining_phase = np.zeros(n_frames, dtype=int)
    remaining_surgery = np.zeros(n_frames, dtype=int)
    elapsed_phase = np.zeros(n_frames, dtype=int)
    phase_progress = np.zeros(n_frames, dtype=float)
    future_phase_starts = np.full((n_frames, NUM_PHASES), -1, dtype=np.int32)
    # Compute Labels for each frame
    for frame_idx in range(n_frames):
        # Remaining frames left in surgery
        remaining_surgery[frame_idx] = ((n_frames - 1) - frame_idx)
        # find the boundary for this frame
        b_idx = frame_to_boundary[frame_idx]
        phase_id, start, end = boundaries[b_idx]
        # Frames left in phase
        remaining_phase[frame_idx] = end - frame_idx
        # Frames since the phase started
        elapsed_phase[frame_idx] = frame_idx - start
        phase_duration = end - start + 1
        if phase_duration > 1:
            phase_progress[frame_idx] = (frame_idx - start)/(phase_duration -1)
        else:
            phase_progress[frame_idx] = 1 # phase only has 1 frame

        # calculate future phase start times
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
    
def compute_phase_statistics(all_video_data):
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

def preprocess_dataset(video_ids=None, output_dir='results', batch_size=64):
    """
    Main function to preprocess the entire dataset.
    Extracts timing labels for all videos and saves them.
    
    Args:
        video_ids: List of video IDs to process (default: all 80)
        output_dir: Directory to save output files
        batch_size: Batch size for loading data
        
    Returns:
        Tuple of (timing_labels_path, statistics_path)
    """
    # Default to all 80 videos
    if video_ids is None:
        video_ids = list(range(80))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Preprocessing {len(video_ids)} videos...")
    print("="*50)
    
    # Step 1: Extract data from all videos
    all_video_data = {}
    all_timing_labels = {}
    
    for video_id in tqdm(video_ids, desc="Processing videos"):
        try:
            # Extract raw data
            video_data = extract_video_data(video_id, batch_size)
            all_video_data[video_id] = video_data
            
            # Compute timing labels
            timing = compute_timing_labels(video_data['phases'])
            
            # Add the original data too (we'll need phases and instruments later)
            timing['phases'] = video_data['phases']
            timing['instruments'] = video_data['instruments']
            
            all_timing_labels[video_id] = timing
            
        except Exception as e:
            print(f"\nError processing video {video_id}: {e}")
            continue
    
    # Step 2: Compute statistics across all videos
    print("\nComputing phase statistics...")
    stats = compute_phase_statistics(all_video_data)
    
    # Step 3: Save timing labels to .npz file
    # We flatten the nested structure for numpy saving
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
    print("\n" + "="*50)
    print("PHASE DURATION SUMMARY (in frames, 1 fps)")
    print("="*50)
    for phase_name in PHASE_NAMES:
        d = stats['durations'][phase_name]
        mean_min = d['mean'] / 60  # Convert to minutes
        std_min = d['std'] / 60
        print(f"{phase_name:30s}: {d['mean']:7.1f} ± {d['std']:6.1f} frames "
              f"({mean_min:5.1f} ± {std_min:4.1f} min)")
    
    return timing_path, stats_path

def load_timing_labels(timing_path):
    """
    Load preprocessed timing labels from a .npz file.
    
    Args:
        timing_path: Path to timing_labels.npz
        
    Returns:
        Dictionary mapping video_id to timing labels
        
    Example usage:
        labels = load_timing_labels('results/timing_labels.npz')
        video_0_remaining = labels[0]['remaining_phase']
    """
    data = np.load(timing_path)
    
    # Reconstruct the nested dictionary structure
    all_labels = {}
    
    for key in data.files:
        # Key format: "video_{id}_{field}"
        # Example: "video_0_remaining_phase"
        parts = key.split('_')
        video_id = int(parts[1])
        field = '_'.join(parts[2:])  # Handle fields with underscores
        
        if video_id not in all_labels:
            all_labels[video_id] = {}
        
        all_labels[video_id][field] = data[key]
    
    return all_labels

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess Cholec80 dataset for timing prediction'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='mphy0043_cw/results',
        help='Directory to save output files (default: mphy0043_cw/results)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Batch size for data loading (default: 64)'
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
        video_ids=args.video_ids,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
