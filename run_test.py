#!/usr/bin/env python3
"""
Simple test runner
"""

import subprocess
import sys

def main():
    """Run the final fixes test"""
    print("🚀 Running Final Fixes Test...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "test_final_fixes.py"], 
                              capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nExit code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    main()