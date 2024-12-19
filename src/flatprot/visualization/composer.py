# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np

from flatprot.structure import Structure, SecondaryStructureType
from flatprot.projection import Projector
from flatprot.visualization.elements import VisualizationElement
from .utils import CanvasSettings
from flatprot.visualization.scene import Scene
from flatprot.visualization.elements.group import Group


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
    translation_x = self._canvas_settings.width / 2
    translation_y = self._canvas_settings.height / 2

    # Apply translation
    canvas_coords = scaled_coords + np.array([translation_x, translation_y])

    return canvas_coords


def secondary_structure_to_visualization_element(
    secondary_structure: SecondaryStructureType,
    coordinates: np.ndarray,
) -> VisualizationElement:
    """Convert a secondary structure to a visualization element."""
    pass


def structure_to_scene(
    structure: Structure,
    projector: Projector,
    canvas_settings: Optional[CanvasSettings] = None,
) -> Scene:
    """Convert a structure to a renderable scene.

    Args:
        structure: The structure to visualize
        projector: The projector to use for coordinate transformation
        canvas_settings: Optional canvas settings

    Returns:
        A Scene object ready for rendering
    """
    scene = Scene(canvas_settings=canvas_settings)

    # Process each chain
    for chain in structure:
        # Create a group for this chain
        chain_group = Group([])

        # Get coordinates for all elements in the chain
        coords = np.vstack(
            [element.coordinates for element in chain.secondary_structure]
        )

        # Project coordinates
        projected_coords = projector.project(coords)
        canvas_coords = transform_to_canvas_space(projected_coords, canvas_settings)

        # Create visualization elements
        start_idx = 0
        for element in chain.secondary_structure:
            end_idx = start_idx + len(element.coordinates)
            element_coords = canvas_coords[start_idx:end_idx]

            vis_element = secondary_structure_to_visualization_element(
                element_type=element.type, coordinates=element_coords
            )
            if vis_element:
                chain_group.add_element(vis_element)

            start_idx = end_idx

        scene.add_element(chain_group)

    return scene
