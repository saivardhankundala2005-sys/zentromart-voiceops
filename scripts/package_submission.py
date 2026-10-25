#!/usr/bin/env python3
"""VoiceOps 2026 - Clean Submission Packaging Script.

Creates a compliant submission zip archive containing all source code, tests,
scripts, evaluation reports, sample outputs, and documentation, while strictly
excluding secrets (.env), the raw dataset, build caches, and git metadata.
"""

import os
import sys
import zipfile
from pathlib import Path

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).parent.parent.resolve()

INCLUDED_PATHS = [
    "src",
    "tests",
    "scripts",
    "reports",
    "sample_outputs",
    "run.py",
    "run.sh",
    "app.py",
    "README.md",
    "report.md",
    "report.pdf",
    "demo_script.md",
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
]

EXCLUDED_NAMES = {
    ".env",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data_index",
    "scratch",
    "Capstone_Project_Dataset_FTTT_2026",
}


def should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_NAMES or part.endswith(".pyc") or part.endswith(".zip"):
            return False
    return True


def package_submission(output_zip: str = "voiceops_submission.zip"):
    zip_path = ROOT_DIR / output_zip
    print(f"Creating submission package at: {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item_name in INCLUDED_PATHS:
            item_path = ROOT_DIR / item_name
            if not item_path.exists():
                print(f"    Warning: '{item_name}' does not exist, skipping.")
                continue

            if item_path.is_file():
                if should_include(item_path):
                    zipf.write(item_path, arcname=item_name)
                    print(f"  + Added file: {item_name}")
            elif item_path.is_dir():
                for file_path in item_path.rglob("*"):
                    if file_path.is_file() and should_include(file_path):
                        arcname = file_path.relative_to(ROOT_DIR)
                        zipf.write(file_path, arcname=str(arcname))
                print(f"  + Added directory: {item_name}/")

    size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)
    print("=" * 65)
    print(f"✅ Package created successfully: {zip_path.name} ({size_mb} MB)")
    print("=" * 65)


if __name__ == "__main__":
    out_name = sys.argv[1] if len(sys.argv) > 1 else "voiceops_submission.zip"
    package_submission(out_name)
