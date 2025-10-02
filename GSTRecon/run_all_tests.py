"""
Comprehensive Test Runner for AI/ML Enhanced GST Reconciliation System

This script runs all unit tests, integration tests, and performance tests
for the AI/ML enhanced GST reconciliation system.
"""

import unittest
import sys
import os
import time
import warnings
from pathlib import Path

# Suppress warnings during testing
warnings.filterwarnings('ignore')

def discover_and_run_tests():
    """Discover and run all tests"""
    
    print("="*80)
    print("AI/ML ENHANCED GST RECONCILIATION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Test categories and their files
    test_categories = {
        "Unit Tests": [
            "test_smart_matching_unit.py",
            "test_anomaly_detector_unit.py", 
            "test_data_quality_unit.py",
            "test_predictive_scoring_unit.py"
        ],
        "Integration Tests": [
            "test_ai_ml_integration.py",
            "test_fallback_mechanisms.py",
            "test_configuration_integration.py"
        ],
        "Performance Tests": [
            "test_performance_stress.py"
        ]
    }
    
    overall_results = {
        'total_tests': 0,
        'total_failures': 0,
        'total_errors': 0,
        'category_results': {}
    }
    
    start_time = time.time()
    
    for category, test_files in test_categories.items():
        print(f"\n{'-'*60}")
        print(f"RUNNING {category.upper()}")
        print(f"{'-'*60}")
        
        category_results = {
            'tests': 0,
            'failures': 0,
            'errors': 0,
            'time': 0
        }
        
        category_start_time = time.time()
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\nExecuting: {test_file}")
                print("-" * 40)
                
                # Discover tests in the file
                loader = unittest.TestLoader()
                
                try:
                    # Import the test module
                    module_name = test_file[:-3]  # Remove .py extension
                    spec = __import__(module_name)
                    
                    # Load tests from module
                    suite = loader.loadTestsFromModule(spec)
                    
                    # Run tests
                    runner = unittest.TextTestRunner(
                        verbosity=2,
                        stream=sys.stdout,
                        buffer=True
                    )
                    
                    result = runner.run(suite)
                    
                    # Update category results
                    category_results['tests'] += result.testsRun
                    category_results['failures'] += len(result.failures)
                    category_results['errors'] += len(result.errors)
                    
                    # Print individual test file results
                    print(f"\nResults for {test_file}:")
                    print(f"  Tests: {result.testsRun}")
                    print(f"  Failures: {len(result.failures)}")
                    print(f"  Errors: {len(result.errors)}")
                    print(f"  Success: {result.wasSuccessful()}")
                    
                    if result.failures:
                        print(f"  Failure Details:")
                        for test, traceback in result.failures:
                            print(f"    - {test}")
                    
                    if result.errors:
                        print(f"  Error Details:")
                        for test, traceback in result.errors:
                            print(f"    - {test}")
                
                except ImportError as e:
                    print(f"  ERROR: Could not import {test_file}: {e}")
                    category_results['errors'] += 1
                
                except Exception as e:
                    print(f"  ERROR: Failed to run {test_file}: {e}")
                    category_results['errors'] += 1
            
            else:
                print(f"  WARNING: Test file {test_file} not found")
        
        category_end_time = time.time()
        category_results['time'] = category_end_time - category_start_time
        
        # Print category summary
        print(f"\n{category} Summary:")
        print(f"  Total Tests: {category_results['tests']}")
        print(f"  Failures: {category_results['failures']}")
        print(f"  Errors: {category_results['errors']}")
        print(f"  Time: {category_results['time']:.2f} seconds")
        print(f"  Success Rate: {((category_results['tests'] - category_results['failures'] - category_results['errors']) / max(1, category_results['tests']) * 100):.1f}%")
        
        # Update overall results
        overall_results['total_tests'] += category_results['tests']
        overall_results['total_failures'] += category_results['failures']
        overall_results['total_errors'] += category_results['errors']
        overall_results['category_results'][category] = category_results
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL TEST SUMMARY")
    print(f"{'='*80}")
    
    print(f"Total Tests Run: {overall_results['total_tests']}")
    print(f"Total Failures: {overall_results['total_failures']}")
    print(f"Total Errors: {overall_results['total_errors']}")
    print(f"Total Time: {total_time:.2f} seconds")
    
    success_count = (overall_results['total_tests'] - 
                    overall_results['total_failures'] - 
                    overall_results['total_errors'])
    
    if overall_results['total_tests'] > 0:
        success_rate = (success_count / overall_results['total_tests']) * 100
        print(f"Overall Success Rate: {success_rate:.1f}%")
    else:
        print("No tests were executed")
        return False
    
    # Print category breakdown
    print(f"\nCategory Breakdown:")
    for category, results in overall_results['category_results'].items():
        if results['tests'] > 0:
            category_success_rate = ((results['tests'] - results['failures'] - results['errors']) / results['tests']) * 100
            print(f"  {category}: {results['tests']} tests, {category_success_rate:.1f}% success, {results['time']:.2f}s")
        else:
            print(f"  {category}: No tests executed")
    
    # Print recommendations
    print(f"\nRecommendations:")
    
    if overall_results['total_failures'] > 0:
        print(f"  - Review and fix {overall_results['total_failures']} test failures")
    
    if overall_results['total_errors'] > 0:
        print(f"  - Investigate and resolve {overall_results['total_errors']} test errors")
    
    if success_rate < 95:
        print(f"  - Improve test success rate (currently {success_rate:.1f}%)")
    
    if total_time > 300:  # 5 minutes
        print(f"  - Consider optimizing test execution time (currently {total_time:.1f}s)")
    
    # Performance-specific recommendations
    performance_results = overall_results['category_results'].get('Performance Tests', {})
    if performance_results.get('failures', 0) > 0 or performance_results.get('errors', 0) > 0:
        print(f"  - Address performance issues identified in performance tests")
    
    # Return success status
    return (overall_results['total_failures'] == 0 and 
            overall_results['total_errors'] == 0 and 
            overall_results['total_tests'] > 0)


def run_specific_test_category(category):
    """Run tests for a specific category"""
    
    test_categories = {
        "unit": [
            "test_smart_matching_unit.py",
            "test_anomaly_detector_unit.py", 
            "test_data_quality_unit.py",
            "test_predictive_scoring_unit.py"
        ],
        "integration": [
            "test_ai_ml_integration.py",
            "test_fallback_mechanisms.py",
            "test_configuration_integration.py"
        ],
        "performance": [
            "test_performance_stress.py"
        ]
    }
    
    if category not in test_categories:
        print(f"Unknown test category: {category}")
        print(f"Available categories: {', '.join(test_categories.keys())}")
        return False
    
    print(f"Running {category} tests...")
    
    test_files = test_categories[category]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nExecuting: {test_file}")
            
            # Run the test file
            result = os.system(f"python {test_file}")
            
            if result != 0:
                print(f"Test file {test_file} failed with exit code {result}")
                return False
        else:
            print(f"Test file {test_file} not found")
    
    return True


def main():
    """Main test runner function"""
    
    # Check if specific category was requested
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        success = run_specific_test_category(category)
    else:
        # Run all tests
        success = discover_and_run_tests()
    
    # Exit with appropriate code
    if success:
        print(f"\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed or encountered errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()