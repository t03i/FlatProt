# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts"
# ]
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from pymol import cmd

app = App()


@app.default
def main(
    structure_file: Annotated[Path, Parameter(exists=True)],
    output_file: Path,
) -> None:
    """Load a structure, orient it, and save a PNG image using PyMOL.

    This script is intended to be run with PyMOL's Python interpreter, e.g.:
    `pymol -cr pymol_single.py <path/to/structure.cif> <path/to/output.png>`

    Parameters
    ----------
    structure_file
        Path to the input structure file (e.g., CIF, PDB). Must exist.
    output_file
        Path to save the output PNG image.
    """
    cmd.load(str(structure_file))
    cmd.orient("all")
    getattr(cmd, "as")("cartoon")
    cmd.color("spectrum", "all")
    cmd.bg_color("white")
    cmd.png(str(output_file), dpi=300, quiet=1)


if __name__ == "__main__":
    # When run with `pymol -cr`, PyMOL handles argument parsing for its own
    # options, and the remaining arguments are passed to the script.
    try:
        app()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Make sure PyMOL quits on error to prevent hanging.
        cmd.quit(1)
