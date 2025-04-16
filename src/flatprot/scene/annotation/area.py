# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple

import numpy as np
from pydantic import Field

from flatprot.core import (
    ResidueRangeSet,
    ResidueCoordinate,
    Structure,
    CoordinateCalculationError,
    logger,
)

from .base_annotation import BaseAnnotationElement, BaseAnnotationStyle
from ..base_element import SceneGroupType


def _convex_hull(points):
    """Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm."""
    # Ensure points is a NumPy array for sorting and calculations
    points = np.asarray(points)
    if points.shape[0] <= 2:
        return points
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    points = points[sorted_indices]

    lower = []
    for p in points:
        # Use cross product to check orientation
        # while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-1]) <= 0:
        # Use explicit 2D cross product calculation to avoid deprecation warning
        while (
            len(lower) >= 2
            and (
                (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-1][1])
                - (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-1][0])
            )
            <= 0
        ):
            lower.pop()
        lower.append(p)

    upper = []
    # Iterate in reverse for the upper hull
    for p in points[::-1]:
        # while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-1]) <= 0:
        # Use explicit 2D cross product calculation
        while (
            len(upper) >= 2
            and (
                (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-1][1])
                - (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-1][0])
            )
            <= 0
        ):
            upper.pop()
        upper.append(p)

    # Concatenate the lower and upper hulls
    # The last point of lower and the first point of upper are the same
    # The first point of lower and the last point of upper are the same
    # Exclude the endpoints to avoid duplication: lower[:-1] and upper[:-1]
    return np.array(lower[:-1] + upper[:-1])


def _apply_padding(hull_points_2d, padding):
    centroid = np.mean(hull_points_2d, axis=0)

    # Offset each vertex outward from the centroid by the padding distance
    offset_vectors = hull_points_2d - centroid
    # Normalize vectors, handle potential zero-length vectors
    norms = np.linalg.norm(offset_vectors, axis=1, keepdims=True)
    # Avoid division by zero if a point coincides with the centroid (or all points are the same)
    safe_norms = np.where(norms == 0, 1, norms)
    normalized_vectors = offset_vectors / safe_norms

    padded_points_2d = hull_points_2d + normalized_vectors * padding
    if len(padded_points_2d) >= 3:
        padded_points_2d = _convex_hull(padded_points_2d)

    return padded_points_2d


# --- Area Annotation Specific Style ---
class AreaAnnotationStyle(BaseAnnotationStyle):
    """Style properties specific to AreaAnnotation elements."""

    fill_color: Optional[str] = Field(
        default=None,
        description="Optional fill color (hex string). If None, uses 'color' with reduced opacity.",
    )
    fill_opacity: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Opacity for the fill color."
    )
    stroke_width: float = Field(
        default=1.0, ge=0, description="Width of the area outline stroke."
    )
    linestyle: Tuple[float, ...] = Field(
        default=(),
        description="Dash pattern for the outline (e.g., (5, 5) for dashed). Empty tuple means solid.",
    )
    padding: float = Field(
        default=20.0,
        ge=0,
        description="Padding pixels added outside the convex hull.",
    )
    interpolation_points: int = Field(
        default=3,
        ge=3,
        description="Number of points to generate along the hull outline before smoothing.",
    )
    smoothing_window: int = Field(
        default=1,
        ge=1,
        description="Window size for rolling average smoothing (odd number recommended).",
    )
    # Inherits color (used for stroke if fill_color is None), offset, label etc.


# --- Area Annotation Scene Element ---
class AreaAnnotation(BaseAnnotationElement[AreaAnnotationStyle]):
    """Represents an annotation highlighting an area encompassing specific residues or ranges."""

    def __init__(
        self,
        id: str,
        style: Optional[AreaAnnotationStyle] = None,
        label: Optional[str] = None,
        residue_range_set: Optional[ResidueRangeSet] = None,
        target_coordinates: Optional[List[ResidueCoordinate]] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes an AreaAnnotation.

        Exactly one of `residue_range_set` or `target_coordinates` must be provided
        to define the residues encompassed by the area.

        Args:
            id: A unique identifier for this annotation element.
            style: The specific style instance for this area annotation.
            label: Optional text label for the annotation.
            residue_range_set: The set of residue ranges this annotation targets.
            target_coordinates: A list of specific residue coordinates this annotation targets.
            parent: The parent SceneGroup in the scene graph, if any.

        Raises:
            ValueError: If neither or both targeting arguments are provided.
        """
        # Metadata argument removed, using label directly
        super().__init__(
            id=id,
            residue_range_set=residue_range_set,
            target_coordinates=target_coordinates,
            style=style,
            label=label,
            parent=parent,
        )
        self._cached_outline_coords: Optional[np.ndarray] = None

    @property
    def default_style(self) -> AreaAnnotationStyle:
        """Provides the default style for AreaAnnotation elements."""
        return AreaAnnotationStyle()

    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Calculate the padded convex hull outline coordinates for the area annotation.

        Fetches coordinates for all residues defined in the residue_range_set
        that exist in the structure. Calculates the convex hull if at least 3
        points are found.

        Args:
            structure: The core Structure object containing coordinate data.

        Returns:
            A NumPy array of 2D + Depth coordinates (shape [N, 3]) representing
            the padded convex hull outline of the area (X, Y, AvgDepth), or None if
            the calculation fails (e.g., fewer than 3 points found).

        Raises:
            CoordinateCalculationError: If fewer than 3 valid coordinates are found
                                        for the specified residue range set.
        """
        logger.debug(f"Calculating area coordinates for '{self.id}'")
        if self._cached_outline_coords is not None:
            return self._cached_outline_coords

        if self.residue_range_set is None:
            # This case should ideally be prevented by the __init__ validation
            # but good to handle defensively.
            logger.error(f"AreaAnnotation '{self.id}' has no residue_range_set.")
            return None

        # 1. Collect all available target 3D coordinates from the structure
        target_coords_3d_list: List[np.ndarray] = []
        # Iterate through all individual ResidueCoordinate objects defined by the ranges
        for res_coord in self.residue_range_set:
            point = structure.get_coordinate_at_residue(res_coord)
            if point is not None:
                target_coords_3d_list.append(point)
            else:
                logger.debug(
                    f"Residue {res_coord} not found in structure for AreaAnnotation '{self.id}', silently skipping."
                )

        if len(target_coords_3d_list) < 3:
            # Raise specific error if not enough points were found
            raise CoordinateCalculationError(
                f"Need at least 3 valid points to calculate area for annotation '{self.id}', found {len(target_coords_3d_list)} within its range set."
            )

        # Convert list to numpy array for calculations
        target_coords_3d = np.array(target_coords_3d_list)

        target_coords_2d = target_coords_3d[:, :2]  # Use only XY for shape calculation
        avg_depth = float(np.mean(target_coords_3d[:, 2]))  # Calculate average depth

        # 2. Compute the convex hull using Andrew's monotone chain algorithm
        hull_points_2d = _convex_hull(target_coords_2d)

        # 3. Apply padding by offsetting the vertices of the convex hull
        padding = self.style.padding
        if padding > 0 and len(hull_points_2d) > 0:  # Add check for non-empty hull
            padded_points_2d = _apply_padding(hull_points_2d, padding)
        else:
            padded_points_2d = hull_points_2d

        # 4. Combine XY with the calculated average depth
        if len(padded_points_2d) == 0:
            # This could happen if input points were collinear/identical and hull failed
            raise CoordinateCalculationError(
                f"Could not compute valid outline for AreaAnnotation '{self.id}' after padding. This is a calculation issue, please check the residue range set."
            )

        num_outline_points = len(padded_points_2d)
        depth_column = np.full((num_outline_points, 1), avg_depth)
        outline_coords_3d = np.hstack((padded_points_2d, depth_column))

        self._cached_outline_coords = outline_coords_3d
        return self._cached_outline_coords
