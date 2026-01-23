"""
Data module for MPHY0043 Coursework.

This module contains:
- dataset.py: Raw data loader for Cholec80 PNG frames
- dataloader.py: High-level dataset creation functions
- preprocessing.py: Timing label extraction
- augmentation.py: Data augmentation functions
"""

from .dataset import (
    create_dataset,
    load_video_data,
    TRAIN_VIDEO_IDS,
    VAL_VIDEO_IDS,
    TEST_VIDEO_IDS,
    NUM_PHASES,
    NUM_TOOLS,
    PHASE_NAMES,
    TOOL_NAMES
)
from .dataloader import (
    get_train_dataset,
    get_val_dataset,
    get_test_dataset,
    get_dataset_for_videos
)
from .preprocessing import (
    load_timing_labels,
    preprocess_dataset,
    compute_timing_labels
)
