# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from flatprot.io.structure_gemmi_adapter import GemmiStructureParser
from flatprot.transformation.inertia import (
    InertiaTransformer,
    InertiaTransformParameters,
)
from flatprot.composer import structure_to_scene
from flatprot.drawing.canvas import Canvas
from flatprot.style.manager import StyleManager


def create_protein_visualization(pdb_file: Path, output_file: Path) -> None:
    """Create an SVG visualization of a protein structure.

    Args:
        pdb_file: Path to PDB structure file
        output_file: Path where to save the SVG
        theme: Visualization theme to use (default: "publication")
    """
    # 1. Parse structure
    parser = GemmiStructureParser()
    structure = parser.parse_structure(pdb_file)

    # 2. Set up projection
    transformer = InertiaTransformer()

    transform_parameters = InertiaTransformParameters(residues=list(structure.residues))

    style_manager = StyleManager.create_default()

    # 4. Create and render scene
    scene = structure_to_scene(
        structure=structure,
        transformer=transformer,
        style_manager=style_manager,
        transform_parameters=transform_parameters,
    )

    # 5. Save to file
    canvas = Canvas(scene, style_manager)
    canvas.render(str(output_file))


if __name__ == "__main__":
    output_path = Path("out/3Ftx")
    output_path.mkdir(parents=True, exist_ok=True)
    # Example usage
    create_protein_visualization(
        pdb_file=Path("data/3Ftx/cobra.cif"),
        output_file=Path("out/3Ftx/cobra_inertia.svg"),
    )
