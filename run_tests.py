#!/usr/bin/env python3
"""
Test runner for Stage1 mesh optimization tests.
Provides convenient interface to run different test suites.
"""
import sys
import subprocess
from pathlib import Path
import argparse


def run_tests(test_type="unit", verbose=False, coverage=False):
    """
    Run specified test suite
    
    Args:
        test_type: Type of tests to run ("unit", "integration", "performance", "all")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=mesh_optim", "--cov-report=html", "--cov-report=term"])
    
    # Add test selection based on type
    if test_type == "unit":
        cmd.append("tests/stage1/")
    elif test_type == "integration":
        cmd.append("tests/test_integration.py")
    elif test_type == "performance":
        cmd.extend(["tests/test_performance.py", "-m", "not slow"])
    elif test_type == "performance-full":
        cmd.append("tests/test_performance.py")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Add common pytest options
    cmd.extend([
        "--tb=short",           # Shorter traceback format
        "--strict-markers",     # Strict marker checking
        "-ra"                   # Show summary of all outcomes
    ])
    
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def check_dependencies():
    """Check if required test dependencies are available"""
    required_packages = ["pytest", "pytest-cov"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required test dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage1 mesh optimization tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  unit         - Run unit tests for core modules
  integration  - Run integration tests  
  performance  - Run performance tests (fast subset)
  performance-full - Run all performance tests including slow ones
  all          - Run all tests

Examples:
  python run_tests.py unit -v
  python run_tests.py integration --coverage
  python run_tests.py performance
  python run_tests.py all -v --coverage
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "performance-full", "all"],
        default="unit",
        nargs="?",
        help="Type of tests to run (default: unit)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies and exit"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("All test dependencies are available")
            return 0
        else:
            return 1
    
    # Check dependencies before running tests
    if not check_dependencies():
        return 1
    
    # Run tests
    success = run_tests(args.test_type, args.verbose, args.coverage)
    
    if success:
        print("\n✅ Tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())