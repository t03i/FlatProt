# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts",
# ]
# ///
"""Aligns a structure to a family and then projects it."""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App()


@app.default
def main(
    structure: Annotated[
        Path, Parameter(help="Path to the PDB/mmCIF structure file to project.")
    ],
    output_file: Annotated[Path, Parameter(help="Path to save the output SVG file.")],
) -> None:
    """Aligns a structure to a family and then creates a 2D projection.

    This script first aligns the given protein structure to the best matching family.
    It then uses this matrix to create a standardized 2D projection of the structure.

    Parameters
    ----------
    structure
        Path to the PDB/mmCIF structure file to project.
    output_file
        Path to save the output SVG file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        matrix_path = Path(temp_dir) / "alignment.npy"

        # Step 1: Align the structure to the family to get a rotational matrix.
        align_cmd = [
            "flatprot",
            "align",
            str(structure),
            "-m",
            str(matrix_path),
        ]
        subprocess.run(
            align_cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        # Step 2: Create a 2D projection using the generated alignment matrix.
        project_cmd = [
            "flatprot",
            "project",
            str(structure),
            "--matrix",
            str(matrix_path),
            "--output",
            str(output_file),
        ]
        subprocess.run(
            project_cmd,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    app()
