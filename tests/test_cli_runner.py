"""Test batch CLI runner functionality."""
import json
import pytest
from pathlib import Path
from run import discover_audio_files


def test_discover_audio_files_grouping(tmp_path):
    # Create test files
    (tmp_path / "conv01_turn1.wav").write_bytes(b"RIFFdummy")
    (tmp_path / "conv01_turn2.wav").write_bytes(b"RIFFdummy")
    (tmp_path / "standalone_q1.wav").write_bytes(b"RIFFdummy")
    (tmp_path / "readme.txt").write_text("not audio")

    items = discover_audio_files(tmp_path)
    assert len(items) == 3

    # conv01 turn 1 should come before turn 2
    conv_items = [item for item in items if item[0] == "conv01"]
    assert len(conv_items) == 2
    assert conv_items[0][1] == 1
    assert conv_items[1][1] == 2
    assert conv_items[0][2].name == "conv01_turn1.wav"
    assert conv_items[1][2].name == "conv01_turn2.wav"

    standalone = [item for item in items if "standalone" in item[0]]
    assert len(standalone) == 1
    assert standalone[0][1] == 1
