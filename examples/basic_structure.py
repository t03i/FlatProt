# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from flatprot.io.structure_gemmi_adapter import GemmiStructureParser
from flatprot.transformation.inertia import (
    InertiaTransformer,
    InertiaTransformParameters,
)
from flatprot.utils.structure import structure_to_scene
from flatprot.drawing.canvas import Canvas
from flatprot.style.manager import StyleManager
from flatprot.scene import Scene, SceneGroup
from flatprot.scene.annotations import LineAnnotation, PointAnnotation, AreaAnnotation


def create_scene(pdb_file: Path) -> tuple[Scene, StyleManager]:
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

    # 3. Create scene from structure
    scene = structure_to_scene(
        structure=structure,
        transformer=transformer,
        style_manager=style_manager,
        transform_parameters=transform_parameters,
    )

    return scene, style_manager


def draw_scene(scene: Scene, style_manager: StyleManager, output_file: Path) -> None:
    canvas = Canvas(scene, style_manager)
    canvas.render(str(output_file))


def create_annotations(scene: Scene, style_manager: StyleManager) -> None:
    # Create annotations

    annotation_group = SceneGroup(id="annotation_group")
    scene.add_element(annotation_group)

    element1 = scene.get_elements_for_residue("A", 5)[0]
    index1 = scene.get_element_index_from_global_index(5, element1)
    element2 = scene.get_elements_for_residue("A", 35)[0]
    index2 = scene.get_element_index_from_global_index(35, element2)

    annotation = LineAnnotation(
        content="line annotation",
        indices=[index1, index2],
        targets=[element1, element2],
        style_manager=style_manager,
    )
    scene.add_element(annotation, annotation_group)

    elements = []

    for i in [
        50,
        70,
        42,
    ]:
        element = scene.get_elements_for_residue("A", i)[0]
        index = scene.get_element_index_from_global_index(i, element)
        elements.append(element)
        point_annotation = PointAnnotation(
            content="point annotation",
            indices=[index],
            targets=[element],
            style_manager=style_manager,
        )
        scene.add_element(point_annotation, annotation_group)

    area_annotation = AreaAnnotation(
        content="area annotation",
        indices=None,
        targets=scene.get_elements_for_residue_range("A", 1, 40),
        style_manager=style_manager,
    )
    scene.add_element(area_annotation, annotation_group)


if __name__ == "__main__":
    output_path = Path("out/3Ftx")
    output_path.mkdir(parents=True, exist_ok=True)
    # Example usage
    scene, style_manager = create_scene(
        pdb_file=Path("data/3Ftx/cobra.cif"),
    )
    create_annotations(scene, style_manager)
    draw_scene(scene, style_manager, Path("out/3Ftx/cobra_inertia.svg"))
