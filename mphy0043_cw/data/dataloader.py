"""
DataLoader for MPHY0043 Coursework.
Provides training, validation, and test datasets using raw Cholec80 data.

This module wraps dataset.py and preprocessing.py to provide convenient
functions for loading datasets with timing labels.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import tensorflow as tf
import numpy as np

# Import from our dataset module
from mphy0043_cw.data.dataset import (
    create_dataset,
    create_sequence_dataset,
    load_video_data,
    TRAIN_VIDEO_IDS,
    VAL_VIDEO_IDS,
    TEST_VIDEO_IDS,
    NUM_PHASES,
    NUM_TOOLS,
    FRAME_HEIGHT,
    FRAME_WIDTH
)
from mphy0043_cw.data.preprocessing import load_timing_labels


def get_train_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None,
    shuffle: bool = True,
    augment: bool = True
) -> tf.data.Dataset:
    """
    Get training dataset (videos 01-32).

    Args:
        data_dir: Path to Cholec80 data directory
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation

    Returns:
        tf.data.Dataset with training data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_dataset(
        data_dir=data_dir,
        video_ids=TRAIN_VIDEO_IDS,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=augment,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_val_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None
) -> tf.data.Dataset:
    """
    Get validation dataset (videos 33-40).

    Args:
        data_dir: Path to Cholec80 data directory
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)

    Returns:
        tf.data.Dataset with validation data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_dataset(
        data_dir=data_dir,
        video_ids=VAL_VIDEO_IDS,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_test_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None
) -> tf.data.Dataset:
    """
    Get test dataset (videos 41-80).

    Args:
        data_dir: Path to Cholec80 data directory
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)

    Returns:
        tf.data.Dataset with test data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_dataset(
        data_dir=data_dir,
        video_ids=TEST_VIDEO_IDS,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_dataset_for_videos(
    data_dir: str,
    video_ids: List[int],
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None,
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Get dataset for specific video IDs.

    Args:
        data_dir: Path to Cholec80 data directory
        video_ids: List of video IDs (1-indexed)
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)
        shuffle: Whether to shuffle data

    Returns:
        tf.data.Dataset with specified videos
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_dataset(
        data_dir=data_dir,
        video_ids=video_ids,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=False,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


# ============================================================================
# SEQUENCE DATASETS (For SSM Time Predictor)
# ============================================================================

def get_train_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 32,
    timing_labels_path: Optional[str] = None
) -> tf.data.Dataset:
    """
    Get training sequence dataset (videos 01-32).

    Returns sequences of consecutive frames for SSM-based models.

    Args:
        data_dir: Path to Cholec80 data directory
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        stride: Step size between sequence starts (controls overlap)
        timing_labels_path: Path to timing_labels.npz

    Returns:
        tf.data.Dataset with:
            - frames: (batch, seq_len, H, W, C)
            - elapsed_phase: (batch, seq_len, 1)
            - phase: (batch, seq_len)
            - remaining_phase: (batch, seq_len, 1)
            - future_phase_starts: (batch, seq_len, 6)
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=TRAIN_VIDEO_IDS,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True,
        stride=stride,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_val_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 64,
    timing_labels_path: Optional[str] = None
) -> tf.data.Dataset:
    """
    Get validation sequence dataset (videos 33-40).

    Args:
        data_dir: Path to Cholec80 data directory
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        stride: Step size between sequence starts
        timing_labels_path: Path to timing_labels.npz

    Returns:
        tf.data.Dataset with sequence data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=VAL_VIDEO_IDS,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        stride=stride,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_test_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 64,
    timing_labels_path: Optional[str] = None
) -> tf.data.Dataset:
    """
    Get test sequence dataset (videos 41-80).

    Args:
        data_dir: Path to Cholec80 data directory
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        stride: Step size between sequence starts
        timing_labels_path: Path to timing_labels.npz

    Returns:
        tf.data.Dataset with sequence data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=TEST_VIDEO_IDS,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        stride=stride,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Cholec80 directory')
    parser.add_argument('--timing_labels', type=str, default=None,
                        help='Path to timing_labels.npz')
    args = parser.parse_args()

    print("Testing dataloader...")
    print("=" * 60)

    # Test train dataset
    print("\n1. Loading training dataset (first 2 videos only for speed)...")
    train_ds = get_dataset_for_videos(
        data_dir=args.data_dir,
        video_ids=[1, 2],
        batch_size=4,
        timing_labels_path=args.timing_labels
    )

    # Get one batch
    for batch in train_ds.take(1):
        print(f"   Batch keys: {list(batch.keys())}")
        print(f"   Frame shape: {batch['frame'].shape}")
        print(f"   Phase values: {batch['phase'].numpy()}")
        print(f"   Instruments shape: {batch['instruments'].shape}")
        if 'remaining_phase' in batch:
            print(f"   Remaining phase: {batch['remaining_phase'].numpy()}")

    print("\n" + "=" * 60)
    print("Dataloader test complete!")
