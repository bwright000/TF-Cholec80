"""
Figure generation for MPHY0043 Coursework report.

Generates publication-quality figures for:
- Task A: Time prediction results
- Task B: Tool detection results and comparison
- Research question analysis

Usage:
    python -m mphy0043_cw.visualization.generate_figures --results_path results.json --output_dir figures/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional

# Use a clean style for academic figures
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette (colorblind-friendly)
COLORS = {
    'baseline': '#1f77b4',      # Blue
    'oracle': '#2ca02c',        # Green
    'predicted': '#ff7f0e',     # Orange
    'error': '#d62728',         # Red
    'neutral': '#7f7f7f'        # Gray
}

# Tool names for axis labels
TOOL_NAMES = [
    'Grasper', 'Bipolar', 'Hook', 'Scissors',
    'Clipper', 'Irrigator', 'SpecimenBag'
]

PHASE_NAMES = [
    'Preparation', 'Calot Triangle\nDissection', 'Clipping\nCutting',
    'Gallbladder\nDissection', 'Gallbladder\nRetraction',
    'Cleaning\nCoagulation', 'Gallbladder\nPackaging'
]

SHORT_PHASE_NAMES = [
    'Prep', 'CTD', 'CC', 'GD', 'GR', 'ClCo', 'GP'
]


# ============================================================================
# TASK A: TIME PREDICTION FIGURES
# ============================================================================

def plot_time_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = "Time Prediction: Predicted vs Actual"
):
    """
    Create scatter plot of predicted vs actual remaining time.

    Args:
        y_true: Ground truth remaining time (frames)
        y_pred: Predicted remaining time (frames)
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Convert to minutes
    y_true_min = y_true / 60
    y_pred_min = y_pred / 60

    # Scatter plot with alpha for density
    ax.scatter(y_true_min, y_pred_min, alpha=0.1, s=5, c=COLORS['baseline'])

    # Perfect prediction line
    max_val = max(y_true_min.max(), y_pred_min.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    # Error bands
    ax.fill_between([0, max_val], [0, max_val - 5], [0, max_val + 5],
                    alpha=0.2, color=COLORS['oracle'], label='±5 min')

    ax.set_xlabel('Actual Remaining Time (minutes)', fontsize=12)
    ax.set_ylabel('Predicted Remaining Time (minutes)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_horizon_mae(
    horizon_mae: Dict,
    output_path: str,
    title: str = "MAE by Prediction Horizon"
):
    """
    Bar chart showing MAE at different prediction horizons.

    Args:
        horizon_mae: Dictionary with horizon -> {'mae_minutes': float, 'n_samples': int}
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = list(horizon_mae.keys())
    maes = [horizon_mae[h]['mae_minutes'] for h in horizons if horizon_mae[h]['mae_minutes'] is not None]
    samples = [horizon_mae[h]['n_samples'] for h in horizons if horizon_mae[h]['mae_minutes'] is not None]
    valid_horizons = [h for h in horizons if horizon_mae[h]['mae_minutes'] is not None]

    x = np.arange(len(valid_horizons))
    bars = ax.bar(x, maes, color=COLORS['baseline'], edgecolor='black', linewidth=1)

    # Add sample count annotations
    for i, (bar, n) in enumerate(zip(bars, samples)):
        ax.annotate(f'n={n:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (minutes)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_horizons, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_phase_mae(
    per_phase_mae: Dict,
    output_path: str,
    title: str = "MAE by Surgical Phase"
):
    """
    Bar chart showing MAE for each surgical phase.

    Args:
        per_phase_mae: Dictionary with phase_id -> MAE
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    phases = list(range(7))
    maes = [per_phase_mae.get(i, 0) / 60 for i in phases]  # Convert to minutes

    # Color gradient based on MAE
    colors = plt.cm.RdYlGn_r(np.array(maes) / max(maes) if max(maes) > 0 else np.zeros_like(maes))

    bars = ax.bar(phases, maes, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Surgical Phase', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (minutes)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(phases)
    ax.set_xticklabels(SHORT_PHASE_NAMES, fontsize=10, rotation=0)

    # Add MAE values on bars
    for bar, mae in zip(bars, maes):
        if mae > 0:
            ax.annotate(f'{mae:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_within_threshold_accuracy(
    metrics: Dict,
    output_path: str,
    title: str = "Prediction Accuracy Within Time Thresholds"
):
    """
    Bar chart showing percentage of predictions within various time thresholds.

    Args:
        metrics: Dictionary with within_X_min accuracy values
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    thresholds = ['2 min', '5 min', '10 min']
    accuracies = [
        metrics['within_2_min'],
        metrics['within_5_min'],
        metrics['within_10_min']
    ]

    colors = [COLORS['error'], COLORS['predicted'], COLORS['oracle']]
    bars = ax.bar(thresholds, accuracies, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Error Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Predictions (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 100)

    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# TASK B: TOOL DETECTION FIGURES
# ============================================================================

def plot_tool_ap_comparison(
    baseline_ap: Dict,
    timed_ap: Dict,
    output_path: str,
    title: str = "Tool Detection: AP Comparison"
):
    """
    Grouped bar chart comparing per-tool AP between baseline and timed models.

    Args:
        baseline_ap: Per-tool AP from baseline model
        timed_ap: Per-tool AP from timed model
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(TOOL_NAMES))
    width = 0.35

    baseline_vals = [baseline_ap[tool]['ap'] for tool in TOOL_NAMES]
    timed_vals = [timed_ap[tool]['ap'] for tool in TOOL_NAMES]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Visual Only)',
                   color=COLORS['baseline'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, timed_vals, width, label='Timed (With Timing)',
                   color=COLORS['oracle'], edgecolor='black', linewidth=1)

    ax.set_xlabel('Surgical Tool', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(TOOL_NAMES, fontsize=10, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)

    # Add improvement annotations
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        improvement = timed_vals[i] - baseline_vals[i]
        color = COLORS['oracle'] if improvement > 0 else COLORS['error']
        ax.annotate(f'{improvement:+.3f}',
                    xy=(x[i] + width/2, b2.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_map_comparison(
    results: Dict,
    output_path: str,
    title: str = "mAP Comparison: Baseline vs Timed Models"
):
    """
    Bar chart comparing overall mAP between models.

    Args:
        results: Dictionary with baseline, oracle, and predicted mAP values
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['Baseline\n(Visual Only)', 'Timed\n(Oracle)', 'Timed\n(Predicted)']
    maps = [
        results.get('baseline', 0),
        results.get('timed_oracle', 0),
        results.get('timed_predicted', results.get('timed_oracle', 0))
    ]
    colors_list = [COLORS['baseline'], COLORS['oracle'], COLORS['predicted']]

    bars = ax.bar(models, maps, color=colors_list, edgecolor='black', linewidth=1)

    ax.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)

    # Add mAP values
    for bar, mAP in zip(bars, maps):
        ax.annotate(f'{mAP:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tool_improvement_heatmap(
    improvements: Dict,
    output_path: str,
    title: str = "Per-Tool AP Improvement"
):
    """
    Horizontal bar chart showing AP improvement for each tool.

    Args:
        improvements: Dictionary with tool -> improvement value
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    tools = list(improvements.keys())
    imps = [improvements[t] for t in tools]

    # Sort by improvement
    sorted_idx = np.argsort(imps)
    tools = [tools[i] for i in sorted_idx]
    imps = [imps[i] for i in sorted_idx]

    # Color based on positive/negative
    colors = [COLORS['oracle'] if i > 0 else COLORS['error'] for i in imps]

    bars = ax.barh(tools, imps, color=colors, edgecolor='black', linewidth=1)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('AP Improvement (Timed - Baseline)', fontsize=12)
    ax.set_ylabel('Surgical Tool', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add value labels
    for bar, imp in zip(bars, imps):
        x_pos = bar.get_width() + 0.005 if imp >= 0 else bar.get_width() - 0.005
        ha = 'left' if imp >= 0 else 'right'
        ax.annotate(f'{imp:+.3f}',
                    xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                    va='center', ha=ha, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# COMBINED/SUMMARY FIGURES
# ============================================================================

def plot_research_summary(
    task_a_metrics: Dict,
    task_b_comparison: Dict,
    output_path: str
):
    """
    Create a summary figure addressing all research questions.

    Args:
        task_a_metrics: Task A evaluation metrics
        task_b_comparison: Task B comparison results
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(15, 10))

    # RQ1: Does timing improve tool detection?
    ax1 = fig.add_subplot(2, 2, 1)
    models = ['Baseline', 'Timed\n(Predicted)']
    maps = [
        task_b_comparison['mAP']['baseline'],
        task_b_comparison['mAP']['timed_predicted']
    ]
    colors_list = [COLORS['baseline'], COLORS['predicted']]
    bars = ax1.bar(models, maps, color=colors_list, edgecolor='black')
    ax1.set_ylabel('mAP')
    ax1.set_title('RQ1: Does Timing Improve Detection?', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.8, 1.0)
    for bar, val in zip(bars, maps):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')

    # RQ2: How does prediction degrade with horizon?
    ax2 = fig.add_subplot(2, 2, 2)
    if 'horizon_mae' in task_a_metrics:
        horizons = list(task_a_metrics['horizon_mae'].keys())
        maes = [task_a_metrics['horizon_mae'][h]['mae_minutes']
                for h in horizons if task_a_metrics['horizon_mae'][h]['mae_minutes'] is not None]
        valid_horizons = [h.replace('min', '') for h in horizons
                          if task_a_metrics['horizon_mae'][h]['mae_minutes'] is not None]
        ax2.bar(valid_horizons, maes, color=COLORS['baseline'], edgecolor='black')
        ax2.set_xlabel('Horizon (minutes)')
        ax2.set_ylabel('MAE (minutes)')
    ax2.set_title('RQ2: Prediction Degradation with Horizon', fontsize=12, fontweight='bold')

    # RQ3: Which tools benefit most?
    ax3 = fig.add_subplot(2, 2, 3)
    if 'per_tool' in task_b_comparison:
        tools = list(task_b_comparison['per_tool'].keys())
        imps = [task_b_comparison['per_tool'][t]['improvement_predicted'] for t in tools]
        sorted_idx = np.argsort(imps)[::-1]
        tools = [tools[i] for i in sorted_idx]
        imps = [imps[i] for i in sorted_idx]
        colors_list = [COLORS['oracle'] if i > 0 else COLORS['error'] for i in imps]
        ax3.barh(tools, imps, color=colors_list, edgecolor='black')
        ax3.axvline(x=0, color='black', linewidth=1)
        ax3.set_xlabel('AP Improvement')
    ax3.set_title('RQ3: Which Tools Benefit Most?', fontsize=12, fontweight='bold')

    # Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = f"""
    SUMMARY STATISTICS

    Task A: Time Prediction
    ─────────────────────
    MAE: {task_a_metrics.get('mae_minutes', 0):.2f} minutes
    Within 5 min: {task_a_metrics.get('within_5_min', 0):.1f}%

    Task B: Tool Detection
    ─────────────────────
    Baseline mAP: {task_b_comparison['mAP']['baseline']:.3f}
    Timed mAP: {task_b_comparison['mAP']['timed_predicted']:.3f}
    Improvement: {task_b_comparison['mAP']['improvement_predicted']:+.3f}
    """

    if 'statistical_test' in task_b_comparison:
        p_val = task_b_comparison['statistical_test']['p_value']
        sig = "Yes" if p_val < 0.05 else "No"
        summary_text += f"""
    Statistical Significance
    ─────────────────────
    p-value: {p_val:.4f}
    Significant (α=0.05): {sig}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def generate_all_figures(results_path: str, output_dir: str):
    """
    Generate all figures from evaluation results.

    Args:
        results_path: Path to JSON file with evaluation results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Task A figures
    if 'task_a' in results:
        print("\nGenerating Task A figures...")
        task_a = results['task_a']

        if 'horizon_mae' in task_a:
            plot_horizon_mae(
                task_a['horizon_mae'],
                os.path.join(output_dir, 'task_a_horizon_mae.png'),
                title="Time Prediction: MAE by Prediction Horizon"
            )

        if 'per_phase_mae' in task_a:
            # Convert string keys to int
            per_phase = {int(k): v for k, v in task_a['per_phase_mae'].items() if v is not None}
            plot_per_phase_mae(
                per_phase,
                os.path.join(output_dir, 'task_a_phase_mae.png'),
                title="Time Prediction: MAE by Surgical Phase"
            )

        if all(k in task_a for k in ['within_2_min', 'within_5_min', 'within_10_min']):
            plot_within_threshold_accuracy(
                task_a,
                os.path.join(output_dir, 'task_a_threshold_accuracy.png'),
                title="Time Prediction: Accuracy Within Error Thresholds"
            )

    # Task B figures
    if 'task_b' in results:
        print("\nGenerating Task B figures...")
        task_b = results['task_b']

        if 'comparison' in task_b:
            comparison = task_b['comparison']

            # mAP comparison
            plot_map_comparison(
                {
                    'baseline': comparison['mAP']['baseline'],
                    'timed_oracle': comparison['mAP']['timed_oracle'],
                    'timed_predicted': comparison['mAP']['timed_predicted']
                },
                os.path.join(output_dir, 'task_b_map_comparison.png'),
                title="Tool Detection: mAP Comparison"
            )

            # Per-tool AP comparison
            if 'baseline' in task_b and 'timed_predicted' in task_b:
                plot_tool_ap_comparison(
                    task_b['baseline']['per_tool'],
                    task_b.get('timed_predicted', task_b.get('timed_oracle', {})).get('per_tool', {}),
                    os.path.join(output_dir, 'task_b_tool_ap_comparison.png'),
                    title="Tool Detection: Per-Tool AP Comparison"
                )

            # Improvement chart
            if 'per_tool' in comparison:
                improvements = {t: comparison['per_tool'][t]['improvement_predicted']
                                for t in comparison['per_tool']}
                plot_tool_improvement_heatmap(
                    improvements,
                    os.path.join(output_dir, 'task_b_tool_improvement.png'),
                    title="Tool Detection: AP Improvement by Tool"
                )

    # Summary figure
    if 'task_a' in results and 'task_b' in results and 'comparison' in results['task_b']:
        print("\nGenerating summary figure...")
        plot_research_summary(
            results['task_a'],
            results['task_b']['comparison'],
            os.path.join(output_dir, 'summary_research_questions.png')
        )

    print(f"\nAll figures saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate figures for MPHY0043 report')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to evaluation results JSON')
    parser.add_argument('--output_dir', type=str, default='mphy0043_cw/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    generate_all_figures(args.results_path, args.output_dir)


if __name__ == '__main__':
    # Test with dummy data if no arguments
    if len(sys.argv) == 1:
        print("Testing figure generation with dummy data...")

        # Create dummy results
        dummy_results = {
            'task_a': {
                'mae_frames': 450,
                'mae_minutes': 7.5,
                'weighted_mae': 380,
                'within_2_min': 25.3,
                'within_5_min': 55.7,
                'within_10_min': 78.2,
                'horizon_mae': {
                    '0-5min': {'mae_frames': 120, 'mae_minutes': 2.0, 'n_samples': 5000},
                    '5-10min': {'mae_frames': 300, 'mae_minutes': 5.0, 'n_samples': 4000},
                    '10-30min': {'mae_frames': 540, 'mae_minutes': 9.0, 'n_samples': 8000},
                    '30-infmin': {'mae_frames': 900, 'mae_minutes': 15.0, 'n_samples': 3000}
                },
                'per_phase_mae': {
                    '0': 180, '1': 420, '2': 300,
                    '3': 600, '4': 360, '5': 240, '6': 120
                }
            },
            'task_b': {
                'baseline': {
                    'mAP': 0.892,
                    'per_tool': {t: {'ap': 0.85 + np.random.uniform(0, 0.1)} for t in TOOL_NAMES}
                },
                'timed_oracle': {
                    'mAP': 0.921,
                    'per_tool': {t: {'ap': 0.88 + np.random.uniform(0, 0.1)} for t in TOOL_NAMES}
                },
                'timed_predicted': {
                    'mAP': 0.908,
                    'per_tool': {t: {'ap': 0.86 + np.random.uniform(0, 0.1)} for t in TOOL_NAMES}
                },
                'comparison': {
                    'mAP': {
                        'baseline': 0.892,
                        'timed_oracle': 0.921,
                        'timed_predicted': 0.908,
                        'improvement_oracle': 0.029,
                        'improvement_predicted': 0.016
                    },
                    'per_tool': {
                        t: {
                            'baseline_ap': 0.85 + i * 0.02,
                            'oracle_ap': 0.88 + i * 0.02,
                            'predicted_ap': 0.86 + i * 0.02,
                            'improvement_oracle': 0.03,
                            'improvement_predicted': 0.01 + (i - 3) * 0.005
                        }
                        for i, t in enumerate(TOOL_NAMES)
                    },
                    'statistical_test': {
                        't_statistic': 2.45,
                        'p_value': 0.023,
                        'significant_at_005': True
                    }
                }
            }
        }

        # Save dummy results
        os.makedirs('mphy0043_cw/results', exist_ok=True)
        dummy_path = 'mphy0043_cw/results/dummy_results.json'
        with open(dummy_path, 'w') as f:
            json.dump(dummy_results, f, indent=2)

        # Generate figures
        generate_all_figures(dummy_path, 'mphy0043_cw/figures')

        print("\nTest complete! Check mphy0043_cw/figures/ for output.")
    else:
        main()

# Terminal script to run this visualization: python -m mphy0043_cw.visualization.generate_figures
