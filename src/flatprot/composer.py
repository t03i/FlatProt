# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np

from flatprot.core import Structure, SecondaryStructureType
from flatprot.transformation import Transformer, TransformParameters
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters
from flatprot.visualization.canvas import (
    Canvas,
    CanvasSettings,
    CanvasGroup,
    CanvasElement,
)
from flatprot.visualization.structure import (
    VisualizationElement,
    VisualizationStyle,
    StyleManager,
    HelixVisualization,
    SheetVisualization,
    CoilVisualization,
)
from flatprot.core import CoordinateManager, CoordinateType


def secondary_structure_to_visualization_element(
    secondary_structure: SecondaryStructureType,
    style: Optional[VisualizationStyle] = None,
) -> VisualizationElement:
    """Convert a secondary structure to a visualization element."""
    if secondary_structure == SecondaryStructureType.HELIX:
        return HelixVisualization(style=style)
    elif secondary_structure == SecondaryStructureType.SHEET:
        return SheetVisualization(style=style)
    elif secondary_structure == SecondaryStructureType.COIL:
        return CoilVisualization(style=style)


def coordinate_manager_from_structure(
    structure: Structure,
    transformer: Transformer,
    transform_parameters: TransformParameters,
    projector: OrthographicProjector,
    projection_parameters: OrthographicProjectionParameters,
) -> CoordinateManager:
    coordinate_manager = CoordinateManager()
    transformed_coords = transformer.transform(
        structure.coordinates, transform_parameters
    )
    canvas_coords, depth = projector.project(transformed_coords, projection_parameters)

    coordinate_manager.add(
        0, len(structure.coordinates), structure.coordinates, CoordinateType.COORDINATES
    )
    coordinate_manager.add(
        0, len(structure.coordinates), transformed_coords, CoordinateType.TRANSFORMED
    )
    coordinate_manager.add(
        0, len(structure.coordinates), canvas_coords, CoordinateType.CANVAS
    )
    coordinate_manager.add(0, len(structure.coordinates), depth, CoordinateType.DEPTH)
    return coordinate_manager


def structure_to_canvas(
    structure: Structure,
    transformer: Transformer,
    canvas_settings: Optional[CanvasSettings] = None,
    style_manager: Optional[StyleManager] = None,
    transform_parameters: Optional[TransformParameters] = None,
) -> Canvas:
    """Convert a structure to a renderable canvas.

    Args:
        structure: The structure to visualize
        transformer: The transformer to use for coordinate transformation
        canvas_settings: Optional canvas settings
        style_manager: Optional style manager
        transformation_parameters: Optional transformation parameters
    Returns:
        A Scene object ready for rendering
    """

    projector = OrthographicProjector()
    projection_parameters = OrthographicProjectionParameters(
        width=canvas_settings.width,
        height=canvas_settings.height,
        padding_x=canvas_settings.padding,
        padding_y=canvas_settings.padding,
        maintain_aspect_ratio=True,
    )

    coordinate_manager = coordinate_manager_from_structure(
        structure, transformer, transform_parameters, projector, projection_parameters
    )

    canvas = Canvas(
        coordinate_manager=coordinate_manager,
        canvas_settings=canvas_settings,
        style_manager=style_manager,
    )

    # Process each chain
    offset = 0
    for chain in structure:
        elements_with_z = []  # Reset for each chain

        for i, element in enumerate(chain.secondary_structure):
            start_idx = offset + element.start
            end_idx = offset + element.end + 1

            if i < len(chain.secondary_structure) - 1:
                end_idx += 1

            depth = coordinate_manager.get(start_idx, end_idx, CoordinateType.DEPTH)

            style = canvas.style_manager.get_style(element.type)

            viz_element = secondary_structure_to_visualization_element(
                element.type, style=style
            )
            if viz_element:
                # Use mean depth along view direction for z-ordering

                elements_with_z.append(
                    (
                        CanvasElement(
                            viz_element, start_idx, end_idx, CoordinateType.CANVAS
                        ),
                        np.mean(depth),
                    )
                )

        # Sort elements by depth (farther objects first)
        sorted_elements = [
            elem
            for elem, z in sorted(elements_with_z, key=lambda x: x[1], reverse=True)
        ]

        group = CanvasGroup(sorted_elements, id=chain.id)
        canvas.add_element(group)
        offset += chain.num_residues

    return canvas
