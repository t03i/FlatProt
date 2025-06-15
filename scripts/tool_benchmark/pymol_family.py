# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "cyclopts"
# ]
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path
from typing import List

from pymol import cmd


def main(
    output_file: Path,
    structure_files: List[Path],
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
    object_names = cmd.get_names("objects")
    if len(object_names) > 1:
        target_object = object_names[0]
        for mobile_object in object_names[1:]:
            cmd.align(mobile_object, target_object)

    # Style the scene
    cmd.do("as cartoon")
    cmd.bg_color("white")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the image
    cmd.png(str(output_file), dpi=300, quiet=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load multiple structures, align them, and save a PNG image using PyMOL."
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to save the output PNG image.",
    )
    parser.add_argument(
        "structure_files",
        type=Path,
        nargs="+",
        help="One or more paths to input structure files (e.g., CIF, PDB).",
    )
    try:
        args = parser.parse_args()
        for f in args.structure_files:
            if not f.exists():
                print(f"Error: structure file not found: {f}", file=sys.stderr)
                cmd.quit(1)
        main(args.output_file, args.structure_files)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        cmd.quit(1)
