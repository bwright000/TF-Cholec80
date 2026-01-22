"""
Evaluation module for MPHY0043 Coursework.

Provides metrics and evaluation pipelines for:
- Task A: Time prediction (MAE, wMAE, within-threshold accuracy)
- Task B: Tool detection (mAP, per-tool AP, precision, recall, F1)
"""

from .metrics import (
    compute_time_prediction_metrics,
    compute_tool_detection_metrics,
    compute_improvement,
    compute_per_tool_improvement,
    paired_t_test,
    format_metrics_table
)

__all__ = [
    'compute_time_prediction_metrics',
    'compute_tool_detection_metrics',
    'compute_improvement',
    'compute_per_tool_improvement',
    'paired_t_test',
    'format_metrics_table'
]
