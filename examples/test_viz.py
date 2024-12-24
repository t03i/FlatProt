# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from flatprot.io.structure_gemmi_adapter import GemmiStructureParser
from flatprot.projection.inertia import InertiaProjector
from flatprot.visualization.composer import structure_to_scene
from flatprot.visualization.utils import CanvasSettings
from flatprot.visualization.elements.style import StyleManager


def create_protein_visualization(
    pdb_file: Path, dssp_file: Path, output_file: Path, theme: str = "publication"
) -> None:
    """Create an SVG visualization of a protein structure.

    Args:
        pdb_file: Path to PDB structure file
        dssp_file: Path to DSSP secondary structure file
        output_file: Path where to save the SVG
        theme: Visualization theme to use (default: "publication")
    """
    # 1. Parse structure
    parser = GemmiStructureParser()
    structure = parser.parse_structure(pdb_file, dssp_file)

    # 2. Set up projection
    projector = InertiaProjector()

    # 3. Configure visualization
    canvas_settings = CanvasSettings(
        width=1200, height=1200, background_color="#FFFFFF", padding=0.1
    )

    style_manager = StyleManager.from_theme(theme)

    # 4. Create and render scene
    scene = structure_to_scene(
        structure=structure,
        projector=projector,
        canvas_settings=canvas_settings,
        style_manager=style_manager,
    )

    # 5. Save to file
    scene.render(str(output_file))


if __name__ == "__main__":
    # Example usage
    create_protein_visualization(
        pdb_file=Path("data/3Ftx/None-Nana_c1_1-Naja_naja.cif"),
        output_file=Path("data/3Ftx/None-Nana_c1_1-Naja_naja.svg"),
    )
