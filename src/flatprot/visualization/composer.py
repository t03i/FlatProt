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


def transform_to_canvas_space(
    coords: np.ndarray, canvas_settings: CanvasSettings
) -> np.ndarray:
    """Transform projected coordinates to canvas space."""
    # Get padding in pixels
    pad_x, pad_y = canvas_settings.padding_pixels

    # Calculate available space
    available_width = canvas_settings.width - 2 * pad_x
    available_height = canvas_settings.height - 2 * pad_y

    # Get bounds of projected coordinates
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    coord_width = max_coords[0] - min_coords[0]
    coord_height = max_coords[1] - min_coords[1]

    # Calculate scale factors and use the smaller one to maintain aspect ratio
    scale_x = available_width / coord_width
    scale_y = available_height / coord_height
    scale = min(scale_x, scale_y)  # Use uniform scaling to maintain aspect ratio

    # Scale coordinates (centered around origin)
    centered_coords = coords - (min_coords + max_coords) / 2
    scaled_coords = centered_coords * scale

    # Apply centering transform
    final_coords = scaled_coords

    return final_coords


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
