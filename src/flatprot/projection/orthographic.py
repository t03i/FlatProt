# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from typing_extensions import Annotated
import numpy as np

from .base import Projector, ProjectionParameters

from pydantic import Field


class OrthographicProjectionParameters(ProjectionParameters):
    """Parameters for orthographic canvas projection."""

    width: int = 1200
    height: int = 1200
    padding_x: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.05
    padding_y: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.05
    maintain_aspect_ratio: bool = True


class OrthographicProjector(Projector):
    """Orthographic projection from 3D directly to canvas space."""

    def project(
        self,
        coordinates: np.ndarray,
        parameters: Optional[OrthographicProjectionParameters] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        params = parameters or OrthographicProjectionParameters()

        # 1. Project to view space
        view = params.view_direction / np.linalg.norm(params.view_direction)
        up = params.up_vector / np.linalg.norm(params.up_vector)

        # Ensure orthogonality
        right = np.cross(up, view)
        if np.allclose(right, 0):
            right = (
                np.array([1.0, 0.0, 0.0])
                if np.allclose(up[0], 0)
                else np.array([0.0, 1.0, 0.0])
            )
        right = right / np.linalg.norm(right)
        up = np.cross(view, right)

        # Create projection matrix
        proj_matrix = np.vstack([right, up])

        # 2. Project to 2D
        coords_2d = (proj_matrix @ coordinates.T).T
        depth = -(coordinates @ view)

        # 3. Center if requested
        if params.center:
            coords_2d = coords_2d - np.mean(coords_2d, axis=0)

        # 4. Calculate scaling to fit within padded canvas
        available_width = params.width * (1 - 2 * params.padding_x)
        available_height = params.height * (1 - 2 * params.padding_y)

        coord_width = np.max(coords_2d[:, 0]) - np.min(coords_2d[:, 0])
        coord_height = np.max(coords_2d[:, 1]) - np.min(coords_2d[:, 1])

        EPSILON = 1e-10
        scale_x = available_width / max(coord_width, EPSILON)
        scale_y = available_height / max(coord_height, EPSILON)

        # Apply uniform scaling if maintaining aspect ratio
        if params.maintain_aspect_ratio:
            scale = min(scale_x, scale_y)
            coords_2d = coords_2d * scale
        else:
            coords_2d = coords_2d * np.array([scale_x, scale_y])

        if not params.center:
            # Calculate the offset to position coordinates within padded area
            min_coords = np.min(coords_2d, axis=0)
            offset = np.array(
                [params.padding_x * params.width, params.padding_y * params.height]
            )
            coords_2d = coords_2d - min_coords + offset

        return coords_2d, depth
