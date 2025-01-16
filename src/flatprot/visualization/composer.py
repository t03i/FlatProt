# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np

from flatprot.structure import Structure, SecondaryStructureType
from flatprot.transformation import Transformer, TransformParameters
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters
from flatprot.visualization.scene import Scene
from .utils import CanvasSettings
from .elements import (
    VisualizationElement,
    VisualizationStyle,
    StyleManager,
    GroupVisualization,
    HelixVisualization,
    SheetVisualization,
    CoilVisualization,
)


def secondary_structure_to_visualization_element(
    secondary_structure: SecondaryStructureType,
    coordinates: np.ndarray,
    style: Optional[VisualizationStyle] = None,
) -> VisualizationElement:
    """Convert a secondary structure to a visualization element."""
    if secondary_structure == SecondaryStructureType.HELIX:
        return HelixVisualization(coordinates, style=style)
    elif secondary_structure == SecondaryStructureType.SHEET:
        return SheetVisualization(coordinates, style=style)
    elif secondary_structure == SecondaryStructureType.COIL:
        return CoilVisualization(coordinates, style=style)


def structure_to_scene(
    structure: Structure,
    transformer: Transformer,
    canvas_settings: Optional[CanvasSettings] = None,
    style_manager: Optional[StyleManager] = None,
    transform_parameters: Optional[TransformParameters] = None,
) -> Scene:
    """Convert a structure to a renderable scene.

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
        padding_x=canvas_settings.padding_pixels[0],
        padding_y=canvas_settings.padding_pixels[1],
        maintain_aspect_ratio=True,
    )

    scene = Scene(canvas_settings=canvas_settings, style_manager=style_manager)

    # Project all coordinates at once
    transformed_coords = transformer.transform(
        structure.coordinates, transform_parameters
    )
    canvas_coords, depth = projector.project(transformed_coords, projection_parameters)

    # Process each chain
    offset = 0
    for chain in structure:
        elements_with_z = []  # Reset for each chain

        for i, element in enumerate(chain.secondary_structure):
            start_idx = offset + element.start
            end_idx = offset + element.end + 1

            if i < len(chain.secondary_structure) - 1:
                end_idx += 1

            element_coords = canvas_coords[start_idx:end_idx]
            style = scene.style_manager.get_style(element.type)

            vis_element = secondary_structure_to_visualization_element(
                element.type, element_coords, style=style
            )
            if vis_element:
                # Use mean depth along view direction for z-ordering
                elements_with_z.append((vis_element, np.mean(depth[start_idx:end_idx])))

        # Sort elements by depth (farther objects first)
        sorted_elements = [
            elem
            for elem, z in sorted(elements_with_z, key=lambda x: x[1], reverse=True)
        ]

        group = GroupVisualization(chain.id, sorted_elements)
        scene.add_element(group)
        offset += chain.num_residues

    return scene
