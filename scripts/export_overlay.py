# ruff: noqa: F821

# /Applications/ChimeraX-1.10-rc2025.06.07.app/Contents/MacOS/ChimeraX "scripts/export_overlay.py"
import glob
import os
from chimerax.core.commands import run

# --- Config — adjust as needed ---
input_folder = "tmp/klk_overlay/representative_structures"
output_image = "out/images/overlay.png"
supersample = 3  # increases resolution
# ----------------------------------

# Find all mmCIFs
mmcif_paths = sorted(glob.glob(os.path.join(input_folder, "*.cif")))
if not mmcif_paths:
    raise RuntimeError(f"No .cif files found in {input_folder!r}")

# Load all files
for path in mmcif_paths:
    run(session, f"open {path}")

# Determine main model (#1)
model_ids = list(session.models)
if not model_ids:
    raise RuntimeError("No models were loaded.")
main = model_ids[0]

# Align all others to main
others = model_ids[1:]
if others:
    run(session, f"matchmaker #{others[0].id[0]}-{others[-1].id[0]} to #{main.id[0]}")

# Apply visual style
for model in model_ids:
    run(session, f"transparency #{model.id[0]} 0 target ac")

run(session, "hide ribbons")
run(session, "show cartoon")
run(session, "view orient")
run(session, "set bgColor transparent")

# Save image with transparency
run(
    session, f"save {output_image} supersample {supersample} transparentBackground true"
)

print(f"Overlay saved to {output_image}")
