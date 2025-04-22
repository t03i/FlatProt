# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from flatprot.core.types import SecondaryStructureType
from flatprot.io.dssp import parse_dssp

# Expected raw segments from DSSP parser itself based on tests/data/test.dssp
EXPECTED_DSSP_SEGMENTS_RAW = sorted(
    [
        (SecondaryStructureType.SHEET, 2, 5),
        (SecondaryStructureType.SHEET, 6, 6),
        (SecondaryStructureType.HELIX, 18, 19),  # 3-10 Helix
        (SecondaryStructureType.SHEET, 14, 17),
        (SecondaryStructureType.SHEET, 24, 30),
        (SecondaryStructureType.SHEET, 38, 44),
        (SecondaryStructureType.HELIX, 46, 54),  # Alpha Helix
        (SecondaryStructureType.SHEET, 60, 65),
    ],
    key=lambda x: x[1],
)  # Sort by start index for consistent comparison


def test_dssp_parser() -> None:
    """Test parsing of a raw DSSP file into secondary structure segments."""
    dssp_file = Path("tests/data/test.dssp")
    segments = parse_dssp(dssp_file)

    # Verify the segments are correctly parsed
    assert isinstance(segments, list)
    # Sort parsed segments for comparison
    sorted_segments = sorted(segments, key=lambda x: x[1])

    assert len(sorted_segments) == len(EXPECTED_DSSP_SEGMENTS_RAW)
    assert all(isinstance(s, tuple) and len(s) == 3 for s in sorted_segments)
    assert all(isinstance(s[0], SecondaryStructureType) for s in sorted_segments)
    assert all(isinstance(s[1], int) and isinstance(s[2], int) for s in sorted_segments)

    # Compare with expected raw segments from DSSP
    assert sorted_segments == EXPECTED_DSSP_SEGMENTS_RAW
