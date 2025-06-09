# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts"
# ]
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Annotated, List

from cyclopts import App, Parameter
from pymol import cmd

app = App()


@app.default
def main(
    output_file: Annotated[Path, Parameter(help="Path to save the output PNG image.")],
    structure_files: Annotated[
        List[Path],
        Parameter(
            exists=True,
            help="One or more paths to input structure files (e.g., CIF, PDB).",
        ),
    ],
) -> None:
    """Load multiple structures, align them, and save a PNG image using PyMOL.

    This script is intended to be run with PyMOL's Python interpreter, e.g.:
    `pymol -cr pymol_family.py <path/to/output.png> <path/to/structure1.cif> ...`

    Parameters
    ----------
    output_file
        Path to save the output PNG image.
    structure_files
        Paths to the input structure files. At least one must be provided.
    """
    if not structure_files:
        print("Error: No structure files provided.", file=sys.stderr)
        cmd.quit(1)

    # Load all structures
    for structure_file in structure_files:
        cmd.load(str(structure_file))

    # Align all to the first loaded structure
    first_structure_name = structure_files[0].stem
    cmd.align("all", first_structure_name)

    # Style the scene
    getattr(cmd, "as")("cartoon")
    cmd.color("spectrum", "all")
    cmd.bg_color("white")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the image
    cmd.png(str(output_file), dpi=300, quiet=1)


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        cmd.quit(1)
