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
        "start": float(result.aligned_region[0]),
        "end": float(result.aligned_region[1]),
        "length": float(result.aligned_region[1] - result.aligned_region[0] + 1),
    }

    data = {
        "structure_file": str(structure_file.resolve()),
        "matched_family": {
            "superfamily": db_entry.entry_id,
        },
        "probability": round(result.probability, 3),
        "aligned_region": aligned_region,
        "rotation_matrix": list(
            map(float, result.rotation_matrix.to_array().flatten())
        ),
        "message": (
            f"Structure aligned to {db_entry.entry_id} "
            f"with probability {result.probability:.1%}. "
            f"Aligned region: residues {result.aligned_region[0]}-{result.aligned_region[1]}."
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
    np.save(output_path, matrix.to_array(), allow_pickle=False)
