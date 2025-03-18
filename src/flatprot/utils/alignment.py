# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from pathlib import Path
import json

import numpy as np

from flatprot.alignment import AlignmentResult, AlignmentDBEntry


# Save alignment results
def save_alignment_results(
    result: AlignmentResult,
    db_entry: AlignmentDBEntry,
    output_path: Optional[Path],
    structure_file: Path,
) -> None:
    """
    Save alignment results in JSON format.

    Args:
        result: Alignment result object containing match details
        db_entry: Database entry with metadata about the matched family
        output_path: Path to save results (None prints to stdout)
        structure_file: Original input structure file path
    """
    aligned_region = {
        "start": result.query_start,
        "end": result.query_end,
        "length": result.query_end - result.query_start + 1,
    }

    data = {
        "structure_file": str(structure_file.resolve()),
        "matched_family": {
            "scop_id": db_entry.scop_id,
            "family": db_entry.family,
            "superfamily": db_entry.superfamily,
            "fold": db_entry.fold,
        },
        "probability": round(result.probability, 3),
        "aligned_region": aligned_region,
        "rotation_matrix": result.rotation_matrix.tolist(),
        "message": (
            f"Structure aligned to {db_entry.superfamily} (SCOP {db_entry.scop_id}) "
            f"with probability {result.probability:.1%}. "
            f"Aligned region: residues {result.query_start}-{result.query_end}."
        ),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
    else:
        print(json.dumps(data, indent=4))


def save_alignment_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """
    Save alignment matrix to a file.

    Args:
        matrix: Alignment matrix to save
        output_path: Path to save the matrix
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
