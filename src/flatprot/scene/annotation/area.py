# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple

import numpy as np
from pydantic import Field

from flatprot.core.coordinates import ResidueRangeSet, ResidueCoordinate
from flatprot.core.structure import Structure


from .base_annotation import BaseAnnotationElement, BaseAnnotationStyle
from ..base_element import SceneGroupType
from ..errors import CoordinateCalculationError


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
        default=200000.0,
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

        Args:
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array of 2D + Depth coordinates (shape [N, 3]) representing
            the padded convex hull outline of the area (X, Y, AvgDepth), or None if
            insufficient points are found.
        """
        print(f"Calculating padded convex hull outline for area annotation '{self.id}'")
        if self._cached_outline_coords is not None:
            return self._cached_outline_coords

        # 1. Collect all target 2D+Depth coordinates
        target_coords_3d = self._get_target_coordinates(structure)
        if len(target_coords_3d) < 3:
            raise CoordinateCalculationError(
                f"Need at least 3 valid points to calculate area for annotation '{self.id}', found {len(target_coords_3d)}."
            )

        target_coords_2d = target_coords_3d[:, :2]  # Use only XY for shape calculation
        avg_depth = float(np.mean(target_coords_3d[:, 2]))  # Calculate average depth

        # 2. Compute the convex hull using Andrew's monotone chain algorithm
        def convex_hull(points):
            """Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm."""
            points = np.array(sorted(points, key=lambda p: (p[0], p[1])))
            if len(points) <= 3:
                return points
            lower = []
            for p in points:
                while (
                    len(lower) >= 2
                    and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0
                ):
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(points):
                while (
                    len(upper) >= 2
                    and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0
                ):
                    upper.pop()
                upper.append(p)
            return np.array(lower[:-1] + upper[:-1])

        hull_points_2d = convex_hull(target_coords_2d)

        # 3. Apply padding by offsetting the vertices of the convex hull
        padding = self.style.padding
        if padding > 0:
            # Calculate the centroid of the convex hull
            centroid = np.mean(hull_points_2d, axis=0)

            # Offset each vertex outward from the centroid by the padding distance
            offset_vectors = hull_points_2d - centroid
            offset_vectors = offset_vectors / np.linalg.norm(
                offset_vectors, axis=1, keepdims=True
            )
            padded_points_2d = hull_points_2d + offset_vectors * padding

            # Recompute the convex hull of the padded points
            padded_points_2d = convex_hull(padded_points_2d)
        else:
            padded_points_2d = hull_points_2d

        # 4. Combine XY with the calculated average depth
        num_outline_points = len(padded_points_2d)
        depth_column = np.full((num_outline_points, 1), avg_depth)
        outline_coords_3d = np.hstack((padded_points_2d, depth_column))

        self._cached_outline_coords = outline_coords_3d
        return self._cached_outline_coords
