#!/usr/bin/env python3
"""
Main orchestrator script for running all benchmarks.

Usage:
    python run_all_benchmarks.py --all          # Run everything
    python run_all_benchmarks.py --benchmarks   # Only benchmarks
    python run_all_benchmarks.py --tests        # Only failure tests
    python run_all_benchmarks.py --report       # Only report generation
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))


def run_recovery_benchmark():
    """Run recovery time benchmark."""
    print("\n" + "=" * 60)
    print("Running Recovery Time Benchmark")
    print("=" * 60)
    from benchmark_recovery_time import main
    return main()


def run_checkpoint_benchmark():
    """Run checkpoint overhead benchmark."""
    print("\n" + "=" * 60)
    print("Running Checkpoint Overhead Benchmark")
    print("=" * 60)
    from benchmark_checkpoint_overhead import main
    return main()


def run_scalability_benchmark():
    """Run scalability benchmark."""
    print("\n" + "=" * 60)
    print("Running Scalability Benchmark")
    print("=" * 60)
    from benchmark_scalability import main
    return main()


def run_network_benchmark():
    """Run network overhead benchmark."""
    print("\n" + "=" * 60)
    print("Running Network Overhead Benchmark")
    print("=" * 60)
    from benchmark_network_overhead import main
    return main()


def run_failure_tests():
    """Run failure scenario tests."""
    print("\n" + "=" * 60)
    print("Running Failure Scenario Tests")
    print("=" * 60)
    from test_failures import run_all_tests
    return run_all_tests()




def run_report():
    """Generate benchmark report."""
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)
    from generate_report import generate_report
    return generate_report()


def main():
    parser = argparse.ArgumentParser(
        description="Run distributed training orchestrator benchmarks"
    )
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks, tests, and report')
    parser.add_argument('--benchmarks', action='store_true',
                       help='Run only performance benchmarks')
    parser.add_argument('--tests', action='store_true',
                       help='Run only failure scenario tests')
    parser.add_argument('--report', action='store_true',
                       help='Generate only the report')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with reduced trials')

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.all, args.benchmarks, args.tests, args.report]):
        parser.print_help()
        print("\n\nExample: python run_all_benchmarks.py --all")
        return

    start_time = time.time()

    print("=" * 60)
    print("Distributed Training Orchestrator Benchmark Suite")
    print("=" * 60)

    if args.all or args.benchmarks:
        try:
            run_recovery_benchmark()
        except Exception as e:
            print(f"Recovery benchmark failed: {e}")

        try:
            run_checkpoint_benchmark()
        except Exception as e:
            print(f"Checkpoint benchmark failed: {e}")

        try:
            run_scalability_benchmark()
        except Exception as e:
            print(f"Scalability benchmark failed: {e}")

        try:
            run_network_benchmark()
        except Exception as e:
            print(f"Network benchmark failed: {e}")

    if args.all or args.tests:
        try:
            run_failure_tests()
        except Exception as e:
            print(f"Failure tests failed: {e}")

    if args.all or args.report:
        try:
            run_report()
        except Exception as e:
            print(f"Report generation failed: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Benchmark suite completed in {elapsed:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
