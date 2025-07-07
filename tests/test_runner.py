# VideoLingo Test Runner
# Comprehensive test execution with coverage reporting

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type="all", coverage=True, verbose=False, parallel=False):
    """
    Run VideoLingo tests with specified parameters
    
    Args:
        test_type: Type of tests to run (unit, integration, all)
        coverage: Whether to generate coverage report
        verbose: Verbose output
        parallel: Run tests in parallel
    """
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Test selection
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
        cmd.extend(["-m", "integration"])
    elif test_type == "ui":
        cmd.extend(["-m", "ui"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "network":
        cmd.extend(["-m", "network"])
    elif test_type == "gpu":
        cmd.extend(["-m", "gpu"])
    else:  # all
        cmd.append("tests/")
    
    # Coverage options
    if coverage:
        cmd.extend([
            "--cov=core",
            "--cov-report=html:tests/reports/coverage_html",
            "--cov-report=xml:tests/reports/coverage.xml",
            "--cov-report=term-missing",
            "--cov-branch",
            "--cov-fail-under=90"
        ])
    
    # Parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("Warning: pytest-xdist not installed, running serially")
    
    # Verbose output
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Additional options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--junitxml=tests/reports/junit.xml"
    ])
    
    # Create reports directory
    os.makedirs("tests/reports", exist_ok=True)
    
    print(f"Running tests: {' '.join(cmd)}")
    
    # Execute tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if coverage and result.returncode == 0:
        print(f"\nCoverage report generated:")
        print(f"  HTML: tests/reports/coverage_html/index.html")
        print(f"  XML:  tests/reports/coverage.xml")
    
    return result.returncode

def run_specific_test(test_path, coverage=False):
    """Run a specific test file or test function"""
    cmd = ["pytest", test_path]
    
    if coverage:
        cmd.extend([
            "--cov=core",
            "--cov-report=term-missing"
        ])
    
    cmd.extend(["-v", "--tb=short"])
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode

def generate_coverage_report():
    """Generate detailed coverage report"""
    print("Generating comprehensive coverage report...")
    
    cmd = [
        "pytest",
        "tests/",
        "--cov=core",
        "--cov-report=html:tests/reports/coverage_html",
        "--cov-report=xml:tests/reports/coverage.xml",
        "--cov-report=term",
        "--cov-branch",
        "--cov-config=tests/.coveragerc"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print(f"\nDetailed coverage report generated:")
        print(f"  HTML: tests/reports/coverage_html/index.html")
        print(f"  XML:  tests/reports/coverage.xml")
        
        # Print coverage summary
        try:
            from coverage import Coverage
            cov = Coverage()
            cov.load()
            print(f"\nCoverage Summary:")
            cov.report()
        except ImportError:
            print("Install coverage package for detailed summary")
    
    return result.returncode

def run_performance_tests():
    """Run performance benchmarks"""
    print("Running performance tests...")
    
    cmd = [
        "pytest",
        "tests/",
        "-m", "slow",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-json=tests/reports/benchmark.json"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print(f"Performance report: tests/reports/benchmark.json")
    
    return result.returncode

def check_test_dependencies():
    """Check if all test dependencies are installed"""
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-asyncio",
        "coverage"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing test dependencies: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="VideoLingo Test Runner")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "ui", "slow", "network", "gpu"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--specific",
        help="Run specific test file or function"
    )
    
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Generate coverage report only"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    if args.check_deps:
        if check_test_dependencies():
            print("All test dependencies are installed")
            return 0
        else:
            return 1
    
    if args.coverage_only:
        return generate_coverage_report()
    
    if args.performance:
        return run_performance_tests()
    
    if args.specific:
        return run_specific_test(args.specific, coverage=not args.no_coverage)
    
    # Check dependencies before running tests
    if not check_test_dependencies():
        return 1
    
    return run_tests(
        test_type=args.test_type,
        coverage=not args.no_coverage,
        verbose=args.verbose,
        parallel=args.parallel
    )

if __name__ == "__main__":
    sys.exit(main())