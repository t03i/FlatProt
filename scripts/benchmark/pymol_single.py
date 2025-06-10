# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts"
# ]
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from pymol import cmd


def main(
    structure_file: Path,
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
    cmd.do("as cartoon")
    cmd.bg_color("white")
    cmd.png(str(output_file), dpi=300, quiet=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a structure, orient it, and save a PNG image using PyMOL."
    )
    parser.add_argument(
        "structure_file",
        type=Path,
        help="Path to the input structure file (e.g., CIF, PDB).",
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to save the output PNG image."
    )
    # When run with `pymol -cr`, PyMOL handles argument parsing for its own
    # options, and the remaining arguments are passed to the script.
    try:
        args = parser.parse_args()
        if not args.structure_file.exists():
            print(f"Error: structure file not found: {args.structure_file}")
            cmd.quit(1)
        main(args.structure_file, args.output_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        # Make sure PyMOL quits on error to prevent hanging.
        cmd.quit(1)
