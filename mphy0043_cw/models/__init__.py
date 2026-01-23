"""
Models package for MPHY0043 Coursework.

This package contains:
- backbone.py: ResNet-50 feature extractor
- time_predictor.py: Task A - remaining time prediction (SSM-based)
- tool_detector.py: Task B baseline - visual-only tool detection
- timed_tool_detector.py: Task B timed - tool detection with timing info
"""

from .backbone import create_backbone, get_backbone_output_dim
from .time_predictor import (
    create_time_predictor,
    SSMLayer,
    SSMBlock,
    weighted_huber_loss,
    future_phase_loss,
    combined_time_loss
)
from .tool_detector import (
    create_tool_detector,
    focal_loss,
    FocalLoss,
    weighted_binary_crossentropy,
    compute_tool_metrics,
    NUM_TOOLS,
    TOOL_NAMES
)
from .timed_tool_detector import (
    create_timed_tool_detector,
    create_attention_timed_tool_detector,
    prepare_timing_inputs_from_predictions,
    prepare_timing_inputs_from_ground_truth
)
