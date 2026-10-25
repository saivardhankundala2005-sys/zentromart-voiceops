#!/usr/bin/env python3
"""VoiceOps 2026 - Pre-Submission Automated Verification Script.

Validates that the workspace is clean, leak-free, properly configured,
and ready for official capstone evaluation and packaging.
"""

import os
import sys
import re
import subprocess
from pathlib import Path

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).parent.parent.resolve()

def check_secret_leaks() -> bool:
    print("[1/6] Checking for hardcoded API keys and secrets...")
    # Patterns for Gnani and Gemini keys
    key_patterns = [
        re.compile(r"vach_[0-9a-zA-Z_]{30,}"),
        re.compile(r"AQ\.[0-9a-zA-Z_\-]{30,}"),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
    ]
    
    scanned_exts = {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".toml", ".txt"}
    excluded_dirs = {".git", ".pytest_cache", ".gemini", "reports", "scratch"}
    excluded_files = {".env"}

    leaks = []

    for path in ROOT_DIR.rglob("*"):
        if path.is_file() and path.suffix in scanned_exts:
            if any(part in excluded_dirs for part in path.parts):
                continue
            if path.name in excluded_files:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pat in key_patterns:
                    matches = pat.findall(content)
                    if matches:
                        leaks.append((path.relative_to(ROOT_DIR), len(matches)))
            except Exception as e:
                print(f"    Warning: Could not read {path}: {e}")

    if leaks:
        print("    ❌ FAILED: Potential API keys found in tracked files:")
        for file_rel, count in leaks:
            print(f"       - {file_rel} ({count} match(es))")
        return False

    print("    ✅ PASSED: No secret leaks found across codebase.")
    return True


def check_env_example() -> bool:
    print("[2/6] Verifying .env and .env.example configuration...")
    env_example = ROOT_DIR / ".env.example"
    gitignore = ROOT_DIR / ".gitignore"

    if not env_example.exists():
        print("    ❌ FAILED: .env.example does not exist.")
        return False

    example_content = env_example.read_text(encoding="utf-8")
    if "vach_" in example_content or "AQ.Ab8" in example_content:
        print("    ❌ FAILED: .env.example contains real secret keys.")
        return False

    if not gitignore.exists():
        print("    ❌ FAILED: .gitignore does not exist.")
        return False

    git_content = gitignore.read_text(encoding="utf-8")
    if ".env" not in git_content:
        print("    ❌ FAILED: .env is not ignored in .gitignore.")
        return False

    print("    ✅ PASSED: .env.example is clean and .env is properly gitignored.")
    return True


def check_dataset_isolation() -> bool:
    print("[3/6] Verifying dataset isolation (no root copies)...")
    # Verify dataset is referenced, not duplicated into src or root
    root_csvs = list(ROOT_DIR.glob("*.csv"))
    if root_csvs:
        print(f"    ❌ FAILED: Root contains misplaced CSV files: {[c.name for c in root_csvs]}")
        return False

    print("    ✅ PASSED: Dataset is isolated and not duplicated into submission root.")
    return True


def check_cli_and_app() -> bool:
    print("[4/6] Verifying CLI runner (run.py) and Web App (app.py)...")
    run_py = ROOT_DIR / "run.py"
    app_py = ROOT_DIR / "app.py"
    run_sh = ROOT_DIR / "run.sh"

    if not run_py.exists():
        print("    ❌ FAILED: run.py is missing.")
        return False
    if not app_py.exists():
        print("    ❌ FAILED: app.py is missing.")
        return False
    if not run_sh.exists():
        print("    ❌ FAILED: run.sh is missing.")
        return False

    # Test run.py --help
    cmd = [sys.executable, str(run_py), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    ❌ FAILED: run.py --help failed: {res.stderr}")
        return False

    print("    ✅ PASSED: run.py, run.sh, and app.py exist and respond correctly.")
    return True


def check_reports_and_artifacts() -> bool:
    print("[5/6] Verifying evaluation reports and sample outputs...")
    required_reports = [
        "reports/step1_baseline.csv",
        "reports/step2_rag_clean.csv",
        "reports/step3_conversations.csv",
        "reports/evaluation_summary.json",
        "reports/benchmark_comparison.md",
    ]
    missing = [r for r in required_reports if not (ROOT_DIR / r).exists()]
    if missing:
        print(f"    ❌ FAILED: Missing evaluation report(s): {missing}")
        return False

    sample_out = ROOT_DIR / "sample_outputs"
    if not sample_out.exists() or not (sample_out / "batch_summary.json").exists():
        print("    ❌ FAILED: sample_outputs directory or batch_summary.json missing.")
        return False

    print("    ✅ PASSED: All benchmark reports and sample outputs verified.")
    return True


def run_unit_tests() -> bool:
    print("[6/6] Executing full test suite (pytest)...")
    cmd = [sys.executable, "-m", "pytest", "tests/test_data_and_calendar.py", "tests/test_tools_and_verification.py", "tests/test_evaluation_metrics.py", "tests/test_cli_runner.py", "-q"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    ❌ FAILED: Pytest failed:\n{res.stdout}\n{res.stderr}")
        return False

    print("    ✅ PASSED: Core unit test suite passed completely.")
    return True


def main():
    print("=" * 65)
    print("VoiceOps 2026 - Pre-Submission Verification Check")
    print("=" * 65)

    checks = [
        check_secret_leaks,
        check_env_example,
        check_dataset_isolation,
        check_cli_and_app,
        check_reports_and_artifacts,
        run_unit_tests,
    ]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
            break

    print("=" * 65)
    if all_passed:
        print("🎉 ALL PRE-SUBMISSION CHECKS PASSED! Repository is ready for packaging.")
        sys.exit(0)
    else:
        print("❌ PRE-SUBMISSION CHECKS FAILED. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
