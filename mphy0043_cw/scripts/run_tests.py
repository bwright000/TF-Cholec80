"""
Test runner for MPHY0043 Coursework.

Runs all module test scripts to verify code correctness.

Usage:
    # Run tests that don't require data (models, metrics, augmentation)
    python -m mphy0043_cw.scripts.run_tests

    # Run all tests including data-dependent ones
    python -m mphy0043_cw.scripts.run_tests --data_dir /path/to/cholec80

    # Run specific test categories
    python -m mphy0043_cw.scripts.run_tests --only models
    python -m mphy0043_cw.scripts.run_tests --only data --data_dir /path/to/cholec80
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from typing import List, Tuple


# Test categories and their modules
TESTS_NO_DATA = {
    'models': [
        ('Backbone', 'mphy0043_cw.models.backbone'),
        ('Tool Detector', 'mphy0043_cw.models.tool_detector'),
        ('Timed Tool Detector', 'mphy0043_cw.models.timed_tool_detector'),
        ('Time Predictor', 'mphy0043_cw.models.time_predictor'),
    ],
    'evaluation': [
        ('Metrics', 'mphy0043_cw.evaluation.metrics'),
    ],
    'augmentation': [
        ('Augmentation', 'mphy0043_cw.data.augmentation'),
    ],
    'visualization': [
        ('Figure Generation', 'mphy0043_cw.visualization.generate_figures'),
    ],
}

TESTS_WITH_DATA = {
    'data': [
        ('Dataset', 'mphy0043_cw.data.dataset', ['--video_id', '1']),
        ('Dataloader', 'mphy0043_cw.data.dataloader', []),
    ],
}


def run_test(name: str, module: str, extra_args: List[str] = None, data_dir: str = None) -> Tuple[bool, str]:
    """
    Run a single test module.

    Args:
        name: Display name for the test
        module: Python module path
        extra_args: Additional command-line arguments
        data_dir: Path to Cholec80 data directory (if required)

    Returns:
        (success, output) tuple
    """
    cmd = [sys.executable, '-m', module]

    if data_dir:
        cmd.extend(['--data_dir', data_dir])

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"

        print(output)

        if result.returncode == 0:
            print(f"\n[PASS] {name}")
            return True, output
        else:
            print(f"\n[FAIL] {name} (exit code: {result.returncode})")
            return False, output

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {name} - exceeded 5 minute limit")
        return False, "Test timed out"
    except Exception as e:
        print(f"\n[ERROR] {name} - {str(e)}")
        return False, str(e)


def run_test_category(category: str, tests: List[Tuple], data_dir: str = None) -> Tuple[int, int]:
    """
    Run all tests in a category.

    Args:
        category: Category name
        tests: List of (name, module, [extra_args]) tuples
        data_dir: Path to Cholec80 data directory

    Returns:
        (passed, failed) counts
    """
    print(f"\n{'#'*60}")
    print(f"# CATEGORY: {category.upper()}")
    print(f"{'#'*60}")

    passed = 0
    failed = 0

    for test_info in tests:
        if len(test_info) == 2:
            name, module = test_info
            extra_args = []
        else:
            name, module, extra_args = test_info

        success, _ = run_test(name, module, extra_args, data_dir)

        if success:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description='Run MPHY0043 test suite')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to Cholec80 data directory (enables data-dependent tests)')
    parser.add_argument('--only', type=str, default=None,
                        choices=['models', 'evaluation', 'augmentation', 'visualization', 'data', 'all'],
                        help='Run only specific test category')
    parser.add_argument('--timing_labels', type=str, default=None,
                        help='Path to timing_labels.npz for dataloader tests')

    args = parser.parse_args()

    start_time = datetime.now()

    print("="*60)
    print("MPHY0043 COURSEWORK - TEST SUITE")
    print("="*60)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir or 'Not provided (skipping data tests)'}")

    total_passed = 0
    total_failed = 0

    # Run tests that don't require data
    if args.only is None or args.only in TESTS_NO_DATA or args.only == 'all':
        for category, tests in TESTS_NO_DATA.items():
            if args.only is None or args.only == category or args.only == 'all':
                passed, failed = run_test_category(category, tests)
                total_passed += passed
                total_failed += failed

    # Run tests that require data (if data_dir provided)
    if args.data_dir and (args.only is None or args.only in TESTS_WITH_DATA or args.only == 'all'):
        for category, tests in TESTS_WITH_DATA.items():
            if args.only is None or args.only == category or args.only == 'all':
                # Add timing_labels to dataloader test if provided
                if category == 'data' and args.timing_labels:
                    tests = [
                        (name, module, extra + ['--timing_labels', args.timing_labels])
                        if module == 'mphy0043_cw.data.dataloader' else (name, module, extra)
                        for name, module, extra in tests
                    ]
                passed, failed = run_test_category(category, tests, args.data_dir)
                total_passed += passed
                total_failed += failed
    elif not args.data_dir and args.only in TESTS_WITH_DATA:
        print(f"\n[SKIP] Category '{args.only}' requires --data_dir argument")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Duration: {duration}")
    print(f"Status: {'ALL TESTS PASSED' if total_failed == 0 else 'SOME TESTS FAILED'}")
    print("="*60)

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
