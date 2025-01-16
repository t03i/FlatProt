# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Projector, ProjectionParameters


@dataclass
class OrthographicProjectionParameters(ProjectionParameters):
    """Parameters for orthographic canvas projection."""

    width: int = 1200
    height: int = 1200
    padding_x: int = 50
    padding_y: int = 50
    maintain_aspect_ratio: bool = True


class OrthographicProjector(Projector):
    """Orthographic projection from 3D directly to canvas space."""

    def project(
        self,
        coordinates: np.ndarray,
        parameters: Optional[OrthographicProjectionParameters] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        params = parameters or OrthographicProjectionParameters()

        # 1. Center 3D coordinates if requested
        if params.center:
            coordinates = coordinates - np.mean(coordinates, axis=0)

        # 2. Calculate view matrix for orthographic projection
        view = params.view_direction / np.linalg.norm(params.view_direction)
        up = params.up_vector / np.linalg.norm(params.up_vector)

        # Ensure orthogonality
        right = np.cross(up, view)
        right = right / np.linalg.norm(right)
        up = np.cross(view, right)

        # Create projection matrix with maximum variance axes first
        proj_matrix = np.vstack([right, up])

        # 3. Project to 2D and calculate depth
        coords_2d = (proj_matrix @ coordinates.T).T * params.scale
        depth = coordinates @ view  # Depth along view direction

        # 4. Calculate available canvas space
        available_width = params.width - 2 * params.padding_x
        available_height = params.height - 2 * params.padding_y

        # 5. Get bounds of projected coordinates
        min_coords = np.min(coords_2d, axis=0)
        max_coords = np.max(coords_2d, axis=0)
        coord_width = max_coords[0] - min_coords[0]
        coord_height = max_coords[1] - min_coords[1]

        # 6. Calculate scale factors
        scale_x = available_width / coord_width if coord_width > 0 else 1.0
        scale_y = available_height / coord_height if coord_height > 0 else 1.0

        if params.maintain_aspect_ratio:
            scale_factor = min(scale_x, scale_y)
        else:
            scale_factor = np.array([scale_x, scale_y])

        # 7. Center and scale to canvas space
        centered_coords = coords_2d - (min_coords + max_coords) / 2
        canvas_coords = centered_coords * scale_factor

        return canvas_coords, depth
