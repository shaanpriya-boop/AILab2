"""
Test Runner for Portfolio Advisor Testing Suite
Runs all tests with proper configuration
"""

import pytest
import sys
from pathlib import Path


def run_all_tests():
    """Run all tests with comprehensive reporting"""
    
    args = [
        "-v",                          # Verbose
        "--tb=short",                  # Short traceback
        "--cov=../",                   # Coverage
        "--cov-report=html",           # HTML coverage report
        "--cov-report=term-missing",   # Terminal coverage
        "--junit-xml=test_results.xml", # JUnit XML for CI/CD
        "-m", "not slow",              # Skip slow tests by default
    ]
    
    print("="*60)
    print("RUNNING PORTFOLIO ADVISOR TEST SUITE")
    print("="*60)
    print()
    
    return pytest.main(args)


def run_unit_tests():
    """Run only unit tests"""
    
    print("Running Unit Tests...")
    return pytest.main([
        "test_unit.py",
        "-v",
        "--tb=short"
    ])


def run_integration_tests():
    """Run only integration tests"""
    
    print("Running Integration Tests...")
    return pytest.main([
        "test_integration.py",
        "-v",
        "--tb=short",
        "-m", "not slow"
    ])


def run_e2e_tests():
    """Run only end-to-end tests"""
    
    print("Running End-to-End Tests...")
    return pytest.main([
        "test_e2e.py",
        "-v",
        "--tb=short",
        "-m", "e2e"
    ])


def run_fast_tests():
    """Run fast tests only (excluding slow and e2e)"""
    
    print("Running Fast Tests...")
    return pytest.main([
        "-v",
        "--tb=short",
        "-m", "not slow and not e2e"
    ])


def run_with_coverage():
    """Run tests with detailed coverage report"""
    
    print("Running Tests with Coverage Analysis...")
    return pytest.main([
        "-v",
        "--tb=short",
        "--cov=../",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-branch",
        "-m", "not slow"
    ])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "unit":
            exit_code = run_unit_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "e2e":
            exit_code = run_e2e_tests()
        elif command == "fast":
            exit_code = run_fast_tests()
        elif command == "coverage":
            exit_code = run_with_coverage()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: unit, integration, e2e, fast, coverage")
            exit_code = 1
    else:
        # Run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
