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
        default=5.0, ge=0, description="Padding pixels added outside the convex hull."
    )
    interpolation_points: int = Field(
        default=50,
        ge=3,
        description="Number of points to generate along the hull outline before smoothing.",
    )
    smoothing_window: int = Field(
        default=5,
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
        style: AreaAnnotationStyle,  # Style is mandatory
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
        """Calculate the smoothed outline coordinates for the area annotation.

        Determines the target residues, fetches their pre-projected 2D+Depth
        coordinates, calculates a smoothed convex hull around the 2D points,
        and returns the outline coordinates with an averaged depth.

        Args:
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array of 2D + Depth coordinates (shape [N, 3]) representing
            the smoothed outline of the area (X, Y, AvgDepth), or None if
            insufficient points are found.
        """
        if self._cached_outline_coords is not None:
            return self._cached_outline_coords

        target_coords_3d = []
        struct_id = getattr(structure, "id", "N/A")

        # 1. Collect all target 2D+Depth coordinates
        target_residues: List[ResidueCoordinate] = []
        if self.targets_specific_coordinates:
            target_residues = self._target_coordinates
        else:  # Use residue_range_set
            if self.residue_range_set:
                # Iterate through all individual coordinates defined by the set
                target_residues = list(
                    self.residue_range_set
                )  # __iter__ yields ResidueCoordinate

        if not target_residues:
            raise CoordinateCalculationError(
                f"No target residues found for AreaAnnotation '{self.id}'."
            )

        # Fetch coordinates for all target residues
        try:
            for res_coord in target_residues:
                chain = structure.get_chain(res_coord.chain_id)
                if res_coord.residue_index in chain:
                    coord_index = chain.coordinate_index(res_coord.residue_index)
                    if 0 <= coord_index < len(structure.coordinates):
                        target_coords_3d.append(structure.coordinates[coord_index])
                    else:
                        raise CoordinateCalculationError(
                            f"Coordinate index {coord_index} out of bounds for {res_coord} in structure '{struct_id}' (AreaAnnotation '{self.id}'). Skipping point."
                        )
                # else: # Silently skip residues not found in chain map
        except (KeyError, IndexError, AttributeError) as e:
            raise CoordinateCalculationError(
                f"Error retrieving coordinates for AreaAnnotation '{self.id}' in structure '{struct_id}': {e}"
            ) from e
            # Continue if possible, maybe some points were gathered

        if len(target_coords_3d) < 3:
            raise CoordinateCalculationError(
                f"Need at least 3 valid points to calculate area for annotation '{self.id}', found {len(target_coords_3d)}."
            )

        target_coords_3d = np.array(target_coords_3d)
        target_coords_2d = target_coords_3d[:, :2]  # Use only XY for shape calculation
        avg_depth = float(np.mean(target_coords_3d[:, 2]))  # Calculate average depth

        # 2. Find the convex hull of the 2D points
        # Using simple angle-sort hull algorithm (might replace with scipy.spatial.ConvexHull if available/needed)
        centroid = np.mean(target_coords_2d, axis=0)
        angles = np.arctan2(
            target_coords_2d[:, 1] - centroid[1], target_coords_2d[:, 0] - centroid[0]
        )
        # Need to handle potential duplicate points which could break hull logic
        unique_points, unique_indices = np.unique(
            target_coords_2d, axis=0, return_index=True
        )
        unique_angles = angles[unique_indices]
        hull_indices = unique_indices[np.argsort(unique_angles)]
        hull_points_2d = target_coords_2d[hull_indices]

        # Ensure the hull is closed
        if not np.allclose(hull_points_2d[0], hull_points_2d[-1]):
            hull_points_2d = np.vstack([hull_points_2d, hull_points_2d[0]])

        # 3. Add padding by expanding points outward from centroid
        padding = self.style.padding
        if padding > 0:
            hull_vectors = hull_points_2d - centroid
            norms = np.linalg.norm(hull_vectors, axis=1)
            # Avoid division by zero for points at the centroid (though unlikely for hull)
            norms[norms < 1e-9] = 1.0
            normalized_vectors = hull_vectors / norms[:, np.newaxis]
            hull_points_2d = hull_points_2d + normalized_vectors * padding

        # 4. Generate more points through linear interpolation along the hull edges
        num_segments = len(hull_points_2d) - 1
        if num_segments == 0:  # Should not happen if len >= 3 and hull closed
            raise CoordinateCalculationError(
                f"Hull calculation resulted in zero segments for AreaAnnotation '{self.id}'."
            )

        points_per_segment = max(1, self.style.interpolation_points // num_segments)
        interpolated_points_list = []
        for i in range(num_segments):
            start_pt = hull_points_2d[i]
            end_pt = hull_points_2d[i + 1]
            # Exclude endpoint (t=1) except for the very last segment to avoid duplication
            num_interp = (
                points_per_segment if i < num_segments - 1 else points_per_segment + 1
            )
            t = np.linspace(0, 1, num_interp, endpoint=(i == num_segments - 1))[
                :, np.newaxis
            ]
            segment = (1 - t) * start_pt + t * end_pt
            interpolated_points_list.append(segment)

        interp_points_2d = np.vstack(interpolated_points_list)

        # 5. Apply rolling average to smooth the curve (if enough points and window > 1)
        window_size = self.style.smoothing_window
        if len(interp_points_2d) > window_size and window_size > 1:
            # Ensure odd window size for centered average
            if window_size % 2 == 0:
                window_size += 1
            half_window = window_size // 2
            kernel = np.ones(window_size) / window_size

            # Pad the points array for periodic boundary conditions (closed loop)
            padded_points = np.vstack(
                [
                    interp_points_2d[-half_window:],
                    interp_points_2d,
                    interp_points_2d[:half_window],
                ]
            )

            # Apply convolution separately for x and y coordinates
            smoothed_x = np.convolve(padded_points[:, 0], kernel, mode="valid")
            smoothed_y = np.convolve(padded_points[:, 1], kernel, mode="valid")
            smoothed_outline_2d = np.column_stack([smoothed_x, smoothed_y])
        else:
            # Not enough points or no smoothing requested
            smoothed_outline_2d = interp_points_2d

        # 6. Combine smoothed XY with the calculated average depth
        num_outline_points = len(smoothed_outline_2d)
        depth_column = np.full((num_outline_points, 1), avg_depth)
        outline_coords_3d = np.hstack((smoothed_outline_2d, depth_column))

        self._cached_outline_coords = outline_coords_3d
        return self._cached_outline_coords
