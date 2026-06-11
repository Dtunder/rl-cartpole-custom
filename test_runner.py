import subprocess
import sys
import json

def run_tests():
    print("="*60)
    print("🚀 RUNNING MASTER TEST SUITE")
    print("="*60)
    
    result = subprocess.run(
        [
            "python", "-m", "pytest", 
            "--cov=custom_cartpole", 
            "--cov=main", 
            "--cov=resilience", 
            "--cov=config", 
            "--cov-report=json",
            "--cov-report=term-missing",
            "tests/"
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Stderr Output:")
        print(result.stderr)

    print("="*60)
    print("📊 TEST SUMMARY REPORT")
    print("="*60)

    if result.returncode != 0 and result.returncode != 5:
        print("❌ Test Suite Status: FAILED")
    else:
        print("✅ Test Suite Status: PASSED")

    try:
        with open("coverage.json") as f:
            data = json.load(f)
            total_coverage = data["totals"]["percent_covered"]
            print(f"📈 Total Test Coverage: {total_coverage:.2f}%")
            if total_coverage < 80:
                print("❌ Coverage Requirement: FAILED (Must be >= 80%)")
                sys.exit(1)
            else:
                print("✅ Coverage Requirement: PASSED")
    except Exception as e:
        print(f"❌ Failed to read coverage report: {e}")
        sys.exit(1)

    print("="*60)

    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_tests()
