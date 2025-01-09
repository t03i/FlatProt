# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np

from flatprot.structure import Structure, SecondaryStructureType
from flatprot.transformation import Transformer, TransformParameters
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

    # Calculate and apply scale factors for each dimension independently
    scale_x = available_width / (max_coords[0] - min_coords[0])
    scale_y = available_height / (max_coords[1] - min_coords[1])

    # Center the coordinates and scale each dimension
    centered_coords = coords - np.mean([min_coords, max_coords], axis=0)
    scaled_coords = centered_coords * np.array([scale_x, scale_y])

    return scaled_coords


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
    scene = Scene(canvas_settings=canvas_settings, style_manager=style_manager)

    # Project all coordinates at once
    projected_coords = transformer.transform(
        structure.coordinates, transform_parameters
    )
    canvas_coords = transform_to_canvas_space(projected_coords[:, :2], canvas_settings)

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
            z_coords = projected_coords[start_idx:end_idx, 2]
            style = scene.style_manager.get_style(element.type)

            vis_element = secondary_structure_to_visualization_element(
                element.type, element_coords, style=style
            )
            if vis_element:
                elements_with_z.append((vis_element, np.mean(z_coords)))

        # Sort elements within this chain by z-coordinate
        sorted_elements = [
            elem for elem, z in sorted(elements_with_z, key=lambda x: x[1])
        ]

        # Add sorted elements as a group for this chain
        group = GroupVisualization(chain.id, sorted_elements)
        scene.add_element(group)
        offset += chain.num_residues
    return scene
