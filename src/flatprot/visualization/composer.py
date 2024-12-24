# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np

from flatprot.structure import Structure, SecondaryStructureType
from flatprot.projection import Projector
from flatprot.visualization.elements import VisualizationElement, VisualizationStyle
from .utils import CanvasSettings
from .elements import StyleManager, Group, Helix, Sheet, Coil
from flatprot.visualization.scene import Scene


def transform_to_canvas_space(
    coords: np.ndarray, canvas_settings: CanvasSettings
) -> np.ndarray:
    """Transform projected coordinates to canvas space."""
    """Transform projected coordinates to canvas space.

        Args:
            coords: Array of shape (N, 2) containing projected coordinates

        Returns:
            Array of shape (N, 2) containing canvas coordinates
        """
    # Get padding in pixels
    pad_x, pad_y = canvas_settings.padding_pixels

    # Calculate available space
    available_width = canvas_settings.width - 2 * pad_x
    available_height = canvas_settings.height - 2 * pad_y

    # Get bounds of projected coordinates
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    # Calculate scale factors for both dimensions
    scale_x = available_width / (max_coords[0] - min_coords[0])
    scale_y = available_height / (max_coords[1] - min_coords[1])

    # Use the smaller scale to maintain aspect ratio
    scale = min(scale_x, scale_y)

    # Center the scaled coordinates
    centered_coords = coords - np.mean([min_coords, max_coords], axis=0)
    scaled_coords = centered_coords * scale

    # Calculate translation to center in canvas
    translation_x = canvas_settings.width / 2
    translation_y = canvas_settings.height / 2

    # Apply translation
    canvas_coords = scaled_coords + np.array([translation_x, translation_y])

    return canvas_coords


def secondary_structure_to_visualization_element(
    secondary_structure: SecondaryStructureType,
    coordinates: np.ndarray,
    style: Optional[VisualizationStyle] = None,
) -> VisualizationElement:
    """Convert a secondary structure to a visualization element."""
    if secondary_structure == SecondaryStructureType.HELIX:
        return Helix(coordinates, style=style)
    elif secondary_structure == SecondaryStructureType.SHEET:
        return Sheet(coordinates, style=style)
    elif secondary_structure == SecondaryStructureType.COIL:
        return Coil(coordinates, style=style)


def structure_to_scene(
    structure: Structure,
    projector: Projector,
    canvas_settings: Optional[CanvasSettings] = None,
    style_manager: Optional[StyleManager] = None,
) -> Scene:
    """Convert a structure to a renderable scene.

    Args:
        structure: The structure to visualize
        projector: The projector to use for coordinate transformation
        canvas_settings: Optional canvas settings

    Returns:
        A Scene object ready for rendering
    """
    scene = Scene(canvas_settings=canvas_settings, style_manager=style_manager)

    # Get coordinates for all elements across all chains
    all_coords = np.vstack(
        [
            element.coordinates
            for chain in structure
            for element in chain.secondary_structure
        ]
    )

    # Project all coordinates at once
    projected_coords = projector.project(all_coords)
    canvas_coords = transform_to_canvas_space(projected_coords[:, :2], canvas_settings)

    # Process each chain
    start_idx = 0
    for chain in structure:
        elements_with_z = []  # Reset for each chain

        for element in chain.secondary_structure:
            end_idx = start_idx + len(element.coordinates)
            element_coords = canvas_coords[start_idx:end_idx]
            z_coords = projected_coords[start_idx:end_idx, 2]
            style = scene.style_manager.get_style(element.type)

            vis_element = secondary_structure_to_visualization_element(
                element.type, element_coords, style=style
            )
            if vis_element:
                elements_with_z.append((vis_element, np.mean(z_coords)))

            start_idx = end_idx

        # Sort elements within this chain by z-coordinate
        sorted_elements = [
            elem for elem, z in sorted(elements_with_z, key=lambda x: x[1])
        ]

        # Add sorted elements as a group for this chain
        group = Group(sorted_elements)
        scene.add_element(group)

    return scene
