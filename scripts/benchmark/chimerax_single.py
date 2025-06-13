# ruff: noqa: F821
# /// script
# requires-python = ">=3.8"
# ///
import argparse
import sys
from pathlib import Path

from chimerax.core.commands import run


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a single structure visualization using ChimeraX."
    )
    parser.add_argument(
        "structure_file",
        type=Path,
        help="Path to the input structure file (e.g., .cif)",
    )
    parser.add_argument(
        "output_image",
        type=Path,
        help="Path where the output image will be saved",
    )
    return parser.parse_args()


def main(
    session, structure_file: Path, output_image: Path, supersample: int = 3
) -> None:
    """Generate a single structure visualization using ChimeraX.

    Parameters
    ----------
    session
        The ChimeraX session object.
    structure_file
        Path to the input structure file (e.g., .cif).
    output_image
        Path where the output image will be saved.
    supersample
        Supersampling factor for the output image.
    """
    if not structure_file.exists():
        print(f"Input file not found: {structure_file}", file=sys.stderr)
        run(session, "exit")

    output_image.parent.mkdir(parents=True, exist_ok=True)

    # Load and style structure
    run(session, f"open {str(structure_file)}")
    run(session, "view orient")
    run(session, "cartoon")
    run(session, "show disulfide")
    run(session, "hide solvent")
    run(session, "hide ligand")
    run(session, "windowsize 1200 1200")

    # Save image
    run(session, f'save "{str(output_image)}" supersample 3 transparent true')
    print(f"Image saved to {output_image}")
    run(session, "exit")


# This script is intended to be run with ChimeraX from the command line, e.g.:
# ChimeraX  --script chimerax_single.py <structure.cif> <output.png>

args = parse_args()
main(session, args.structure_file, args.output_image)
