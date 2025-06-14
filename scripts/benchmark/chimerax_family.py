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
        description="Generate a family structure visualization using ChimeraX."
    )
    parser.add_argument(
        "output_image",
        type=Path,
        help="Path where the output image will be saved",
    )
    parser.add_argument(
        "structure_files",
        type=Path,
        nargs="+",
        help="Paths to the input structure files (e.g., .cif)",
    )
    return parser.parse_args()


# This script is intended to be run from the command line with ChimeraX, e.g.:
# ChimeraX --nogui --script chimerax_family.py <output.png> <structure1.cif> <structure2.cif> ...


def main() -> None:
    """Main function to generate family structure visualization using ChimeraX."""
    args = parse_args()

    if not args.structure_files:
        print("No input structure files provided.", file=sys.stderr)
        run(session, "exit")

    # Load all files
    for path in args.structure_files:
        run(session, f"open {path}")

    # Determine main model (#1)
    model_ids = list(session.models)
    if not model_ids:
        print("No models were loaded.", file=sys.stderr)
        run(session, "exit")

    main = model_ids[0]

    # Align all others to main
    others = model_ids[1:]
    if others:
        other_models_spec = f"#{others[0].id[0]}-{others[-1].id[0]}"
        run(session, f"matchmaker {other_models_spec} to #{main.id[0]}")

    # Apply visual style
    run(session, "hide ribbons")
    run(session, "show cartoon")
    run(session, "view orient")
    run(session, "set bgColor white")

    # Save image
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    run(
        session,
        f"save {str(args.output_image)} supersample 3 transparentBackground false",
    )

    print(f"Overlay saved to {args.output_image}")
    run(session, "exit")


main()
