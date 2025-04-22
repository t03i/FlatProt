# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Literal

import numpy as np

from .base import BaseProjection, BaseProjectionParameters

from pydantic import Field
from numpydantic import NDArray, Shape


class OrthographicProjectionParameters(BaseProjectionParameters):
    """Parameters for orthographic canvas projection."""

    width: int = 1200
    height: int = 1200
    # Padding as a fraction of width/height
    padding_x: Annotated[float, Field(strict=True, ge=0, lt=0.5)] = 0.05
    padding_y: Annotated[float, Field(strict=True, ge=0, lt=0.5)] = 0.05
    maintain_aspect_ratio: bool = True
    canvas_alignment: Literal["center", "top_left"] = "top_left"
    center_original_coordinates: bool = True
    view_direction: NDArray[Shape["3"], float] = np.array([0.0, 0.0, 1.0])
    up_vector: NDArray[Shape["3"], float] = np.array([0.0, 1.0, 0.0])


class OrthographicProjection(BaseProjection[OrthographicProjectionParameters]):
    """Performs orthographic projection from 3D world space to 2D canvas space + depth."""

    def project(
        self,
        coordinates: np.ndarray,
        parameters: OrthographicProjectionParameters,
    ) -> np.ndarray:  # Corrected return type to single array
        """Projects 3D coordinates onto a 2D canvas with depth information.

        Args:
            coordinates: NumPy array of shape (N, 3) representing 3D points.
            parameters: OrthographicProjectionParameters object configuring the projection.

        Returns:
            NumPy array of shape (N, 3) where columns are [X_canvas, Y_canvas, Depth].

        Raises:
            ValueError: If input coordinates do not have shape (N, 3).
        """
        if (
            not isinstance(coordinates, np.ndarray)
            or coordinates.ndim != 2
            or coordinates.shape[1] != 3
        ):
            raise ValueError(
                f"Input coordinates must be a NumPy array with shape (N, 3), got {coordinates}"
            )

        if coordinates.shape[0] == 0:
            return np.empty((0, 3), dtype=float)  # Handle empty input

        # 1. Calculate orthonormal view basis vectors (right, up, view)
        # Ensure view and up vectors are normalized and non-zero
        view_norm = np.linalg.norm(parameters.view_direction)
        up_norm = np.linalg.norm(parameters.up_vector)
        if view_norm < 1e-9 or up_norm < 1e-9:
            raise ValueError("View direction and up vector must be non-zero vectors.")

        view = parameters.view_direction / view_norm
        up = parameters.up_vector / up_norm

        # Gram-Schmidt orthogonalization (or similar) to ensure basis is orthonormal
        right = np.cross(up, view)
        right_norm = np.linalg.norm(right)
        # If up and view are near parallel
        if right_norm < 1e-9:
            # Attempt to create a right vector robustly
            # If view is axis-aligned, pick a perpendicular axis
            if np.allclose(np.abs(view), [0, 0, 1]):
                right = np.array([1.0, 0.0, 0.0])
            elif np.allclose(np.abs(view), [0, 1, 0]):
                right = np.array([1.0, 0.0, 0.0])
            elif np.allclose(np.abs(view), [1, 0, 0]):
                right = np.array([0.0, 1.0, 0.0])
            else:  # Otherwise cross with Z axis (usually safe)
                right = np.cross(np.array([0.0, 0.0, 1.0]), view)
                right_norm_check = np.linalg.norm(right)
                if right_norm_check < 1e-9:  # if view was Z axis
                    right = np.array([1.0, 0.0, 0.0])
                else:
                    right = right / right_norm_check
        else:
            right = right / right_norm

        # Recompute 'up' to be orthogonal to 'view' and 'right'
        up = np.cross(view, right)
        # No need to normalize 'up' again as 'view' and 'right' are orthogonal unit vectors

        # 2. Project 3D coordinates to 2D view plane + get depth
        # Projection matrix selects the components along the 'right' and 'up' axes
        proj_matrix = np.vstack([right, up])  # Shape (2, 3)
        coords_2d = (proj_matrix @ coordinates.T).T  # Result shape (N, 2)

        # Calculate depth along the negative view direction.
        # Larger values indicate points closer to the viewpoint along the -view axis.
        depth = -(coordinates @ view)  # Result shape (N,)

        # 3. Center coordinates in 2D view plane *before* scaling if requested
        coords_2d_centered = coords_2d
        if parameters.center_original_coordinates:
            coords_2d_mean = np.mean(coords_2d, axis=0)
            coords_2d_centered = coords_2d - coords_2d_mean

        # 4. Calculate scaling factors to fit within padded canvas dimensions
        available_width = parameters.width * (1 - 2 * parameters.padding_x)
        available_height = parameters.height * (1 - 2 * parameters.padding_y)

        # Calculate the range (max - min) of the (potentially centered) 2D coordinates
        coord_min = np.min(coords_2d_centered, axis=0)
        coord_max = np.max(coords_2d_centered, axis=0)
        coord_range = coord_max - coord_min

        # Avoid division by zero if all points project to the same location (range is zero)
        EPSILON = 1e-10
        scale_x = available_width / max(coord_range[0], EPSILON)
        scale_y = available_height / max(coord_range[1], EPSILON)

        # Apply scaling (uniform or non-uniform)
        if parameters.maintain_aspect_ratio:
            scale = min(scale_x, scale_y)
            coords_2d_scaled = coords_2d_centered * scale
        else:
            coords_2d_scaled = coords_2d_centered * np.array([scale_x, scale_y])

        # 5. Translate scaled coordinates to the final canvas position
        if parameters.canvas_alignment == "center":
            # Translate the centered+scaled coordinates so their mean is at the canvas center
            canvas_center = np.array([parameters.width / 2, parameters.height / 2])
            final_coords_2d = coords_2d_scaled + canvas_center
        else:
            # Translate the scaled coordinates so their min point aligns with the padded origin
            # We need the minimum of the *scaled* coordinates relative to the *centered* coords min/max
            scaled_min = np.min(coords_2d_scaled, axis=0)
            padded_origin = np.array(
                [
                    parameters.padding_x * parameters.width,
                    parameters.padding_y * parameters.height,
                ]
            )
            # Translate: move the minimum of the scaled shape to the padded origin
            final_coords_2d = coords_2d_scaled - scaled_min + padded_origin

        # 6. Combine 2D canvas coordinates and depth, ensuring correct shape and type
        # Reshape depth to (N, 1) for horizontal stacking
        projected_data = np.hstack((final_coords_2d, depth.reshape(-1, 1)))

        return projected_data.astype(float)  # Ensure float output
