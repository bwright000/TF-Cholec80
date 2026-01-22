"""
DataLoader for MPHY0043 Coursework.
Combines core Cholec80 dataset with preprocessed timing labels.
"""

import sys
import os
import pathlib

# Add parent directory to path for importing tf_cholec80
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tf_cholec80.dataset import make_cholec80
from preprocessing import load_timing_labels
import tensorflow as tf
import numpy as np


# DATASET SPLITS (Standard Cholec80 protocol)
TRAIN_VIDEO_IDS = list(range(0, 32))   # Videos 0-31 (training)
VAL_VIDEO_IDS = list(range(32, 40))    # Videos 32-39 (validation)
TEST_VIDEO_IDS = list(range(40, 80))   # Videos 40-79 (testing)

def make_cholec80_with_timing(n_minibatch, timing_labels_path, config_path=None, video_ids=None, mode="FRAME"):
    """
    Creates Cholec80 dataset with timing labels merged in.

    TODO:
    1. Call make_cholec80() to get base dataset
    2. Load timing labels from .npz file
    3. Map a function to add timing labels to each batch
    4. Return the enhanced dataset
    """
    ds = make_cholec80(
        n_minibatch = n_minibatch,
        config_path = config_path,
        video_ids = video_ids,
        mode = mode,
    )
    all_timing_labels = load_timing_labels(timing_labels_path)

    def add_timing_labels(batch):
        def lookup_timings(video_ids, frame_ids):
            batch_size = len(video_ids)
            remaining_phase = np.zeros(batch_size, dtype = np.int32)
            remaining_surgery = np.zeros(batch_size, dtype = np.int32)
            elapsed_phase = np.zeros(batch_size, dtype = np.int32)
            phase_progress = np.zeros(batch_size, dtype = np.float32)
            future_phase_starts = np.zeros((batch_size, 7), dtype = np.int32)
            for i in range(batch_size):
                frame_id = int(frame_ids[i])
                video_id_str = video_ids[i].decode('utf-8')
                video_num = int(video_id_str.replace('video', ''))
                labels = all_timing_labels[video_num]
                remaining_phase[i] = labels['remaining_phase'][frame_id]
                remaining_surgery[i] = labels['remaining_surgery'][frame_id]
                elapsed_phase[i] = labels['elapsed_phase'][frame_id]
                phase_progress[i] = labels['phase_progress'][frame_id]
                future_phase_starts[i] = labels['future_phase_starts'][frame_id]
            return (remaining_phase, 
                    remaining_surgery, 
                    elapsed_phase, 
                    phase_progress, 
                    future_phase_starts
                    )
        (remaining_phase,
        remaining_surgery,
        elapsed_phase,
        phase_progress,
        future_phase_starts
        ) = tf.py_function(
            func = lookup_timings,
            inp = [batch['video_id'], batch['frame_id']],
            Tout = [tf.int32, tf.int32, tf.int32, tf.float32, tf.int32]
            )
        remaining_phase.set_shape([None])
        remaining_surgery.set_shape([None])
        elapsed_phase.set_shape([None])
        phase_progress.set_shape([None])
        future_phase_starts.set_shape([None, 7])

        batch['remaining_phase'] = remaining_phase
        batch['remaining_surgery'] = remaining_surgery
        batch['elapsed_phase'] = elapsed_phase
        batch['phase_progress'] = phase_progress
        batch['future_phase_starts'] = future_phase_starts

        return batch
    ds = ds.map(add_timing_labels, num_parallel_calls = tf.data.AUTOTUNE)
    return ds



def get_train_dataset(batch_size, timing_labels_path, config_path=None):
    """TODO: Return training dataset (videos 0-63) with FRAME mode"""
    return make_cholec80_with_timing(
        n_minibatch=batch_size,
        timing_labels_path=timing_labels_path,
        config_path=config_path,
        video_ids=TRAIN_VIDEO_IDS,
        mode='FRAME'
    )


def get_val_dataset(batch_size, timing_labels_path, config_path=None):
    """Return validation dataset (videos 32-39) with INFER mode for consistent evaluation"""
    return make_cholec80_with_timing(
        n_minibatch=batch_size,
        timing_labels_path=timing_labels_path,
        config_path=config_path,
        video_ids=VAL_VIDEO_IDS,
        mode='INFER'
    )


def get_test_dataset(batch_size, timing_labels_path, config_path=None):
    """TODO: Return test dataset (videos 72-79) with INFER mode"""
    return make_cholec80_with_timing(
        n_minibatch=batch_size,
        timing_labels_path=timing_labels_path,
        config_path=config_path,
        video_ids=TEST_VIDEO_IDS,
        mode='INFER'
    )



if __name__ == '__main__':
    """TODO: Test dataloader with a small subset of videos and print first batch"""
    test_ds = make_cholec80_with_timing(
        n_minibatch=4,
        timing_labels_path='mphy0043_cw/results/timing_labels.npz',
        config_path='tf_cholec80/configs/config.json',
        video_ids=[0, 1],
        mode="FRAME"
    )
    # Print first batch
    print("Testing dataloader...")
    for batch in test_ds.take(1):
        print("Batch keys:", batch.keys())
        print("Frame shape:", batch['frame'].shape)
        print("Video IDs:", batch['video_id'].numpy())
        print("Frame IDs:", batch['frame_id'].numpy())
        print("Phases:", batch['phase'].numpy())
        print("Remaining phase:", batch['remaining_phase'].numpy())
        print("Remaining surgery:", batch['remaining_surgery'].numpy())
        print("Phase progress:", batch['phase_progress'].numpy())
        break
    print("Test complete!")
