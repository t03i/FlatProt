# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


from typing import NamedTuple
from pathlib import Path

import numpy as np

from flatprot.transformation import TransformationMatrix
from .db import AlignmentDatabase
from .foldseek import FoldseekAligner
from .errors import (
    NoSignificantAlignmentError,
    DatabaseEntryNotFoundError,
)


class AlignmentResult(NamedTuple):
    """Results from a structural family alignment."""

    db_id: str
    probability: float
    aligned_region: np.ndarray
    alignment_scores: np.ndarray
    rotation_matrix: TransformationMatrix


def get_aligned_rotation_database(
    alignment: AlignmentResult, db: AlignmentDatabase
) -> TransformationMatrix:
    """Combines alignment rotation with database rotation.

    Args:
        alignment: Alignment result from align_structure_database
        db: Initialized AlignmentDatabase instance

    Returns:
        Combined transformation matrix

    Raises:
        DatabaseEntryNotFoundError: If matched database entry is missing
    """
    if not db.contains_entry_id(alignment.db_id):
        raise DatabaseEntryNotFoundError(f"Database entry {alignment.db_id} not found")

    db_entry = db.get_by_entry_id(alignment.db_id)
    return alignment.rotation_matrix.combined_rotation(db_entry.rotation_matrix)


def align_structure_database(
    structure_file: Path,
    foldseek_db_path: Path,
    foldseek_command: str = "foldseek",
    min_probability: float = 0.5,
) -> AlignmentResult:
    """Calculate the alignment result for structural alignment using FoldSeek.

    Args:
        structure_file: Path to input protein structure file (PDB/mmCIF).
        foldseek_db_path: Path to FoldSeek-specific database.
        foldseek_command: FoldSeek executable name/path.
        min_probability: Minimum alignment probability threshold.

    Returns:
        AlignmentResult: Result containing alignment details and rotation matrix.

    Raises:
        AlignmentError: General alignment failures
        NoSignificantAlignmentError: No alignment meets probability threshold
    """
    aligner = FoldseekAligner(
        foldseek_executable=foldseek_command, database_path=foldseek_db_path
    )

    alignment_result = aligner.align_structure(
        structure_path=structure_file, min_probability=min_probability
    )

    if alignment_result is None:
        raise NoSignificantAlignmentError(
            f"No alignment found above {min_probability} probability threshold"
        )

    return alignment_result
