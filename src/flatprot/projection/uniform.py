# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
import numpy as np
from pydantic import Field

from .base import BaseProjection, BaseProjectionParameters


class UniformProjectionParameters(BaseProjectionParameters):
    """Parameters for uniform scale projection."""

    scale_factor: Annotated[float, Field(strict=True, gt=0)] = 30


class UniformProjection(BaseProjection[UniformProjectionParameters]):
    """Performs orthographic-style projection but applies a fixed, uniform scale factor."""

    def project(
        self,
        coordinates: np.ndarray,
        parameters: UniformProjectionParameters,
    ) -> np.ndarray:
        """Projects 3D coordinates onto a 2D plane with uniform scaling and depth.

        Args:
            coordinates: NumPy array of shape (N, 3) representing 3D points.
            parameters: UniformProjectionParameters object configuring the projection.

        Returns:
            NumPy array of shape (N, 3) where columns are
            [X_scaled, Y_scaled, Depth].

        Raises:
            ValueError: If input coordinates do not have shape (N, 3) or if
                        view/up vectors are invalid.
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
        # (Logic copied directly from OrthographicProjection)
        view_norm = np.linalg.norm(parameters.view_direction)
        up_norm = np.linalg.norm(parameters.up_vector)
        if view_norm < 1e-9 or up_norm < 1e-9:
            raise ValueError("View direction and up vector must be non-zero vectors.")

        view = parameters.view_direction / view_norm
        up = parameters.up_vector / up_norm

        right = np.cross(up, view)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-9:
            # (Robust right vector calculation copied from orthographic.py)
            if np.allclose(np.abs(view), [0, 0, 1]):
                right = np.array([1.0, 0.0, 0.0])
            elif np.allclose(np.abs(view), [0, 1, 0]):
                right = np.array([1.0, 0.0, 0.0])
            elif np.allclose(np.abs(view), [1, 0, 0]):
                right = np.array([0.0, 1.0, 0.0])
            else:
                right = np.cross(np.array([0.0, 0.0, 1.0]), view)
                right_norm_check = np.linalg.norm(right)
                if right_norm_check < 1e-9:
                    right = np.array([1.0, 0.0, 0.0])
                else:
                    right = right / right_norm_check
        else:
            right = right / right_norm

        up = np.cross(view, right)

        # 2. Project 3D coordinates to 2D view plane + get depth
        proj_matrix = np.vstack([right, up])
        coords_2d = (proj_matrix @ coordinates.T).T
        depth = -(coordinates @ view)

        # 3. Center coordinates in 2D view plane *before* scaling if requested
        coords_2d_processed = coords_2d
        if parameters.center:
            if coords_2d.shape[0] > 0:  # Avoid mean of empty array
                coords_2d_mean = np.mean(coords_2d, axis=0)
                coords_2d_processed = coords_2d - coords_2d_mean
            # else: coords_2d_processed remains empty

        # 4. Apply the fixed, uniform scale factor
        coords_2d_scaled = coords_2d_processed * parameters.scale_factor

        # 5. Combine scaled 2D coordinates and depth
        projected_data = np.hstack((coords_2d_scaled, depth.reshape(-1, 1)))

        return projected_data.astype(float)  # Ensure float output
