#!/usr/bin/env python3

"""
Runs all tests for the backward induction algorithm implementation.
"""

import os
import sys
import time
import subprocess
from termcolor import colored

def run_test(filename):
    """Run a test script and return the result."""
    print(f"Executing: {filename}")
    
    # Start timer
    start_time = time.time()
    
    # Run the test
    completed_process = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True
    )
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print results
    print(f"Completed in {elapsed_time:.2f} seconds with exit code: {completed_process.returncode}")
    
    if completed_process.stdout:
        print("\nOutput:")
        shortened_output = completed_process.stdout
        if len(shortened_output.splitlines()) > 10:
            output_lines = shortened_output.splitlines()
            shortened_output = '\n'.join(['...'] + output_lines[-10:])
        print(shortened_output)
    
    return completed_process.returncode == 0, filename

def main():
    print("Testing the backward induction algorithm implementation...\n")
    
    # List of test files to run
    test_files = [
        "simple_test.py",
        "memoization_test.py"
    ]
    
    # Track results
    results = []
    
    # Run each test
    for test_file in test_files:
        success, name = run_test(test_file)
        results.append((success, name))
        print()  # Add a blank line between tests
    
    # Print summary
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    
    all_passed = True
    for success, name in results:
        status = colored("PASSED", "green") if success else colored("FAILED", "red")
        if name == "simple_test.py":
            test_name = "Simple Game Tree Test"
        elif name == "memoization_test.py":
            test_name = "Memoization Performance Test"
        else:
            test_name = name
            
        print(f"{status}: {test_name} ({name})")
        all_passed = all_passed and success
    
    print("\nAll tests", end=" ")
    if all_passed:
        print(colored("PASSED!", "green"))
    else:
        print(colored("FAILED!", "red"))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 