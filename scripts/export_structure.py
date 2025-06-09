# ruff: noqa: F821

# /Applications/ChimeraX-1.10-rc2025.06.07.app/Contents/MacOS/ChimeraX "scripts/export_structure.py"
import os
from chimerax.core.commands import run


def export_flatprot_style(session, filepath: str, output_path: str, bonds: bool = True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    run(session, f'open "{filepath}"')
    run(session, "view orient")
    run(session, "cartoon")
    if bonds:
        run(session, "show disulfide")
    else:
        run(session, "hide disulfide")
    run(session, "hide solvent")
    run(session, "hide ligand")
    run(session, "windowsize 1200 1200")
    run(session, f'save "{output_path}" supersample 3 transparent true')
    run(session, "close")


def main(session):
    export_flatprot_style(
        session, "data/1KT0/1kt0.cif", "out/images/1kt0.png", bonds=False
    )
    export_flatprot_style(session, "data/3Ftx/cobra.cif", "out/images/cobra.png")
    export_flatprot_style(session, "data/3Ftx/krait.cif", "out/images/krait.png")
    export_flatprot_style(session, "data/3Ftx/snake.cif", "out/images/snake.png")


main(session)
