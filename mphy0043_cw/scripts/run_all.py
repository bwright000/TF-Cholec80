"""
Full experiment pipeline for MPHY0043 Coursework.

Runs all steps in sequence:
1. Preprocessing - Extract timing labels
2. Training Task A - Time prediction model
3. Training Task B - Tool detection baseline
4. Training Task B - Tool detection with timing
5. Evaluation - All models
6. Visualization - Generate report figures

Usage:
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --config mphy0043_cw/config.yaml

For individual steps:
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step preprocess
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step train_time
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step train_tools
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step train_timed_tools
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step evaluate
    python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step visualize
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def run_preprocessing(data_dir: str, config_path: str):
    """Run preprocessing to extract timing labels."""
    cmd = [
        sys.executable, "-m", "mphy0043_cw.data.preprocessing",
        "--data_dir", data_dir,
        "--output_dir", "mphy0043_cw/results"
    ]
    return run_command(cmd, "Preprocessing - Extract Timing Labels")


def run_train_time(data_dir: str, config_path: str):
    """Train the time prediction model (Task A)."""
    cmd = [
        sys.executable, "-m", "mphy0043_cw.training.train_time",
        "--config", config_path,
        "--data_dir", data_dir
    ]
    return run_command(cmd, "Training Task A - Time Predictor")


def run_train_tools(data_dir: str, config_path: str):
    """Train the tool detection baseline (Task B)."""
    cmd = [
        sys.executable, "-m", "mphy0043_cw.training.train_tools",
        "--config", config_path,
        "--data_dir", data_dir
    ]
    return run_command(cmd, "Training Task B - Tool Detector Baseline")


def run_train_timed_tools(data_dir: str, config_path: str):
    """Train the timed tool detector (Task B with timing)."""
    cmd = [
        sys.executable, "-m", "mphy0043_cw.training.train_timed_tools",
        "--config", config_path,
        "--data_dir", data_dir
    ]
    return run_command(cmd, "Training Task B - Timed Tool Detector")


def run_evaluation(data_dir: str, config_path: str):
    """Run full evaluation on all models."""
    cmd = [
        sys.executable, "-m", "mphy0043_cw.evaluation.evaluate",
        "--config", config_path,
        "--data_dir", data_dir,
        "--task", "all",
        "--output", "mphy0043_cw/results/evaluation_results.json"
    ]
    return run_command(cmd, "Evaluation - All Models")


def run_visualization(results_path: str = "mphy0043_cw/results/evaluation_results.json"):
    """Generate figures for the report."""
    cmd = [
        sys.executable, "-m", "mphy0043_cw.visualization.generate_figures",
        "--results_path", results_path,
        "--output_dir", "mphy0043_cw/figures"
    ]
    return run_command(cmd, "Visualization - Generate Figures")


def main():
    parser = argparse.ArgumentParser(description='MPHY0043 Full Experiment Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Cholec80 data directory')
    parser.add_argument('--config', type=str, default='mphy0043_cw/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'preprocess', 'train_time', 'train_tools',
                                 'train_timed_tools', 'evaluate', 'visualize'],
                        help='Which step to run (default: all)')
    args = parser.parse_args()

    start_time = datetime.now()

    print("=" * 60)
    print("MPHY0043 COURSEWORK - EXPERIMENT PIPELINE")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config}")
    print(f"Step: {args.step}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    os.makedirs("mphy0043_cw/results", exist_ok=True)
    os.makedirs("mphy0043_cw/results/checkpoints", exist_ok=True)
    os.makedirs("mphy0043_cw/figures", exist_ok=True)

    success = True

    # Run selected steps
    if args.step in ['all', 'preprocess']:
        success = run_preprocessing(args.data_dir, args.config) and success
        if not success and args.step == 'all':
            print("\nStopping due to preprocessing failure.")
            return 1

    if args.step in ['all', 'train_time']:
        success = run_train_time(args.data_dir, args.config) and success
        if not success and args.step == 'all':
            print("\nWarning: Time predictor training failed. Continuing...")

    if args.step in ['all', 'train_tools']:
        success = run_train_tools(args.data_dir, args.config) and success
        if not success and args.step == 'all':
            print("\nWarning: Tool detector training failed. Continuing...")

    if args.step in ['all', 'train_timed_tools']:
        success = run_train_timed_tools(args.data_dir, args.config) and success
        if not success and args.step == 'all':
            print("\nWarning: Timed tool detector training failed. Continuing...")

    if args.step in ['all', 'evaluate']:
        success = run_evaluation(args.data_dir, args.config) and success

    if args.step in ['all', 'visualize']:
        success = run_visualization() and success

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Status: {'SUCCESS' if success else 'COMPLETED WITH ERRORS'}")

    if success:
        print("\nOutputs:")
        print("  - Timing labels: mphy0043_cw/results/timing_labels.npz")
        print("  - Checkpoints: mphy0043_cw/results/checkpoints/")
        print("  - Evaluation: mphy0043_cw/results/evaluation_results.json")
        print("  - Figures: mphy0043_cw/figures/")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
