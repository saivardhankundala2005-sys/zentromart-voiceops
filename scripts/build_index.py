#!/usr/bin/env python3
"""Build and cache the persistent vector & BM25 knowledge index."""
import argparse
import sys
from pathlib import Path

# Ensure src is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from voiceops.knowledge.indexer import get_indexer

def main():
    parser = argparse.ArgumentParser(description="Build ZentroMart Knowledge Index")
    parser.add_argument("--force", action="store_true", help="Force rebuild of vector embeddings")
    args = parser.parse_args()

    print("Initializing Knowledge Indexer...", flush=True)
    indexer = get_indexer()
    result = indexer.build_index(force_rebuild=args.force)
    print("Index Build Result:", flush=True)
    for k, v in result.items():
        print(f"  {k}: {v}", flush=True)

    report = indexer.health_report()
    print("\nIndex Health Report:", flush=True)
    for k, v in report.items():
        print(f"  {k}: {v}", flush=True)
    print("\nIndex is ready for retrieval!", flush=True)

if __name__ == "__main__":
    main()
