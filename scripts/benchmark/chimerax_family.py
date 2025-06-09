# ruff: noqa: F821
# /// script
# requires-python = ">=3.8"
# ///
import sys
from pathlib import Path

from chimerax.core.commands import run

# This script is intended to be run from the command line with ChimeraX, e.g.:
# ChimeraX --nogui --script chimerax_family.py <output.png> <structure1.cif> <structure2.cif> ...

if len(sys.argv) < 3:
    print(
        "Usage: ChimeraX --nogui --script chimerax_family.py <output.png> <structure1.cif> ...",
        file=sys.stderr,
    )
    # Use 'exit' command for ChimeraX to quit with an error code
    run(session, "exit 1")

output_image = Path(sys.argv[1])
mmcif_paths = sys.argv[2:]
supersample = 3

if not mmcif_paths:
    print("No input structure files provided.", file=sys.stderr)
    run(session, "exit 1")

# Load all files
for path in mmcif_paths:
    run(session, f"open {path}")

# Determine main model (#1)
model_ids = list(session.models)
if not model_ids:
    print("No models were loaded.", file=sys.stderr)
    run(session, "exit 1")

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
output_image.parent.mkdir(parents=True, exist_ok=True)
run(
    session,
    f"save {str(output_image)} supersample {supersample} transparentBackground false",
)

print(f"Overlay saved to {output_image}")
run(session, "exit")
