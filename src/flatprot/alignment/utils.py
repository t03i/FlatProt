# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Callable, Optional

from flatprot.transformation import TransformationMatrix
from .db import AlignmentDatabase, AlignmentDBEntry
from .foldseek import FoldseekAligner, AlignmentResult
from .errors import (
    NoSignificantAlignmentError,
    DatabaseEntryNotFoundError,
)


def _foldseek_id_to_db_id(foldseek_id: str) -> str:
    """Convert FoldSeek ID to database ID.

    Args:
        foldseek_id: FoldSeek ID
    """
    return f"{foldseek_id}".upper()


def get_aligned_rotation_database(
    alignment: AlignmentResult,
    db: AlignmentDatabase,
    id_transform: Callable[[str], str] = _foldseek_id_to_db_id,
) -> tuple[TransformationMatrix, AlignmentDBEntry]:
    """Combines alignment rotation with database rotation.

    Args:
        alignment: Alignment result from align_structure_database
        db: Initialized AlignmentDatabase instance
        id_transform: Function to transform FoldSeek IDs to database IDs

    Returns:
        tuple: A tuple containing:
            - Combined transformation matrix
            - Database entry object for the matched alignment

    Raises:
        DatabaseEntryNotFoundError: If matched database entry is missing
    """
    with db:
        db_id = id_transform(alignment.db_id)
        if not db.contains_entry_id(db_id):
            raise DatabaseEntryNotFoundError(
                f"Database entry {alignment.db_id} not found"
            )

        db_entry = db.get_by_entry_id(db_id)

    alignment_transform: TransformationMatrix = alignment.rotation_matrix
    db_transform: TransformationMatrix = db_entry.rotation_matrix

    # Combine the transformations: apply query_to_target_transform first, then db_transform.
    # T_final = T_db ∘ T_inv_align
    # Since T2.before(T1) applies T1 then T2, we use:
    final_transform = db_transform.before(alignment_transform)

    return final_transform, db_entry


def align_structure_database(
    structure_file: Path,
    foldseek_db_path: Path,
    foldseek_command: str = "foldseek",
    min_probability: float = 0.5,
    target_db_id: Optional[str] = None,
) -> AlignmentResult:
    """Calculate the alignment result for structural alignment using FoldSeek.

    Args:
        structure_file: Path to input protein structure file (PDB/mmCIF).
        foldseek_db_path: Path to FoldSeek-specific database.
        foldseek_command: FoldSeek executable name/path.
        min_probability: Minimum alignment probability threshold.
        target_db_id: If provided, force alignment to this specific FoldSeek target ID.

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
        structure_path=structure_file,
        min_probability=min_probability,
        fixed_alignment_id=target_db_id,
    )

    if alignment_result is None:
        raise NoSignificantAlignmentError(
            f"No alignment found above {min_probability} probability threshold"
        )

    return alignment_result
