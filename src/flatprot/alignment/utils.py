# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


from typing import NamedTuple

import numpy as np

from flatprot.alignment.db import AlignmentDatabase
from flatprot.transformation import TransformationMatrix


class AlignmentResult(NamedTuple):
    """Results from a structural family alignment."""

    db_id: str
    probability: float
    aligned_region: np.ndarray
    alignment_scores: np.ndarray
    rotation_matrix: TransformationMatrix


def alignment_to_db_rotation(
    alignment: AlignmentResult, db: AlignmentDatabase
) -> TransformationMatrix:
    """Converts an alignment result to a combined rotation matrix.

    Args:
        alignment: The alignment result containing rotation and translation
        db: Database containing the target rotation matrix

    Returns:
        Combined rotation matrix that first applies the alignment transform,
        then the database rotation
    """
    # Get database entry rotation
    with db:
        if not db.contains_entry_id(alignment.db_id):
            raise ValueError(f"Alignment database entry {alignment.db_id} not found")
        db_entry = db.get_by_entry_id(alignment.db_id)

    return alignment.rotation_matrix.combined_rotation(db_entry.rotation_matrix)
