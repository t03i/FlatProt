# ruff: noqa: F821
# /// script
# requires-python = ">=3.8"
# ///
import sys
from pathlib import Path

from chimerax.core.commands import run

# This script is intended to be run with ChimeraX from the command line, e.g.:
# ChimeraX  --script chimerax_single.py <structure.cif> <output.png>

if len(sys.argv) != 3:
    print(
        "Usage: ChimeraX --script chimerax_single.py <structure.cif> <output.png>",
        file=sys.stderr,
    )
    run(session, "exit 1")

structure_file = Path(sys.argv[1])
output_image = Path(sys.argv[2])
supersample = 3

if not structure_file.exists():
    print(f"Input file not found: {structure_file}", file=sys.stderr)
    run(session, "exit 1")

# Load structure
run(session, f"open {str(structure_file)}")

# Apply visual style
run(session, "hide ribbons")
run(session, "show cartoon")
run(session, "view orient")
run(session, "set bgColor white")

# Save image
output_image.parent.mkdir(parents=True, exist_ok=True)
run(
    session,
    f"save {str(output_image)} supersample {supersample} transparentBackground false",
)

print(f"Image saved to {output_image}")
run(session, "exit")
