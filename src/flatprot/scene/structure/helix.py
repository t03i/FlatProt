# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core import (
    ResidueRangeSet,
    ResidueCoordinate,
    Structure,
    CoordinateCalculationError,
)
from ..base_element import SceneGroupType

# Assuming base_structure lives directly in scene/structure now
from .base_structure import (
    BaseStructureSceneElement,
    BaseStructureStyle,
)


# --- Helper Function --- (Could be moved to utils)
def calculate_zigzag_points(
    start_point_3d: np.ndarray,
    end_point_3d: np.ndarray,
    thickness: float,
    wavelength: float,
    amplitude: float,
) -> Optional[np.ndarray]:
    """Calculate points for a sharp zigzag helix ribbon based on 3D start/end points.

    Uses only the XY components for shape calculation but preserves Z (depth).

    Args:
        start_point_3d: The start coordinate [X, Y, Depth].
        end_point_3d: The end coordinate [X, Y, Depth].
        thickness: The thickness of the ribbon.
        wavelength: The length of one full zigzag cycle.
        amplitude: The height of the zigzag peaks/valleys from the center.

    Returns:
        NumPy array of ribbon outline points [X, Y, Depth], or None if length is zero.
    """
    start_xy = start_point_3d[:2]
    end_xy = end_point_3d[:2]
    start_depth = start_point_3d[2]
    end_depth = end_point_3d[2]

    direction_xy = end_xy - start_xy
    length = np.linalg.norm(direction_xy)

    if length < 1e-6:  # Handle zero-length case
        return None  # Cannot draw zigzag

    # Ensure direction_xy is float before division or avoid in-place
    direction_xy = direction_xy / length
    perpendicular_xy = np.array([-direction_xy[1], direction_xy[0]])

    num_cycles = max(1, int(length / wavelength))
    # Number of points needed for peaks/valleys plus start/end
    num_wave_points = num_cycles * 2 + 1
    t = np.linspace(0, length, num_wave_points)

    # Alternate amplitudes for zigzag peaks/valleys
    wave_amp = np.zeros(num_wave_points)
    wave_amp[1:-1] = amplitude * np.array(
        [1 if i % 2 != 0 else -1 for i in range(num_wave_points - 2)]
    )

    # Create base points along the path (XY)
    base_points_xy = t[:, None] * direction_xy + start_xy
    # Apply zigzag pattern perpendicular to the path (XY)
    wave_points_xy = base_points_xy + (wave_amp[:, None] * perpendicular_xy)

    # Interpolate depth along the path
    depths = np.linspace(start_depth, end_depth, num_wave_points)

    # Create ribbon effect by offsetting top and bottom
    half_thickness = thickness / 2
    top_points_xy = wave_points_xy + (perpendicular_xy * half_thickness)
    bottom_points_xy = wave_points_xy - (perpendicular_xy * half_thickness)

    # Combine XY with interpolated depths
    top_points_3d = np.hstack((top_points_xy, depths[:, None]))
    bottom_points_3d = np.hstack((bottom_points_xy, depths[:, None]))

    # Combine points to form complete ribbon outline (top -> bottom reversed)
    # Shape will be (num_wave_points * 2, 3)
    return np.concatenate((top_points_3d, bottom_points_3d[::-1]))


# --- Helix Specific Style ---
class HelixStyle(BaseStructureStyle):
    """Style properties specific to Helix elements.

    Defines properties for rendering helices as zigzag ribbons.
    """

    # Override inherited defaults
    color: Color = Field(
        default=Color("#ff0000"), description="Default color for helix (red)."
    )
    stroke_width: float = Field(
        default=1, description="Reference width for calculating helix dimensions."
    )
    simplified_width: float = Field(
        default=2,
        description="Width to use for simplified helix rendering (line only).",
    )
    # Helix-specific attributes
    ribbon_thickness: float = Field(
        default=8,
        description="Factor to multiply linewidth by for the ribbon thickness.",
    )
    wavelength: float = Field(
        default=10.0,
        description="Factor to multiply linewidth by for the zigzag wavelength.",
    )
    amplitude: float = Field(
        default=3.0,
        description="Factor to multiply linewidth by for the zigzag amplitude.",
    )
    min_helix_length: int = Field(
        default=4,
        ge=2,
        description="Minimum number of residues required to draw a zigzag shape instead of a line.",
    )


# --- Helix Scene Element ---
class HelixSceneElement(BaseStructureSceneElement[HelixStyle]):
    """Represents an Alpha Helix segment, visualized as a zigzag ribbon."""

    def __init__(
        self,
        residue_range_set: ResidueRangeSet,
        style: Optional[HelixStyle] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes the HelixSceneElement."""
        super().__init__(residue_range_set, style, parent)
        # Cache for the calculated zigzag coordinates and original length
        self._cached_display_coords: Optional[np.ndarray] = None
        self._original_coords_len: Optional[int] = None

    @property
    def default_style(self) -> HelixStyle:
        """Provides the default style for Helix elements."""
        return HelixStyle()

    def _get_original_coords_slice(self, structure: Structure) -> Optional[np.ndarray]:
        """Helper to extract the original coordinate slice for this helix."""
        coords_list = []
        if not self.residue_range_set.ranges:
            # Cannot get coordinates if no ranges are defined.
            raise CoordinateCalculationError(
                f"Cannot get coordinates for Helix '{self.id}': no residue ranges defined."
            )
        helix_range = self.residue_range_set.ranges[0]

        try:
            chain = structure[helix_range.chain_id]
            for res_idx in range(helix_range.start, helix_range.end + 1):
                if res_idx in chain:
                    coord_idx = chain.coordinate_index(res_idx)
                    if 0 <= coord_idx < len(structure.coordinates):
                        coords_list.append(structure.coordinates[coord_idx])
                    else:
                        # Coordinate index out of bounds.
                        raise CoordinateCalculationError(
                            f"Coordinate index {coord_idx} out of bounds for residue {helix_range.chain_id}:{res_idx} in structure."
                        )
                else:
                    # Residue not found in chain coordinate map.
                    raise CoordinateCalculationError(
                        f"Residue {helix_range.chain_id}:{res_idx} not found in chain coordinate map."
                    )
        except (KeyError, IndexError, AttributeError) as e:
            # Error fetching coordinates
            raise CoordinateCalculationError(
                f"Error fetching coordinates for helix '{self.id}': {e}"
            ) from e

        return np.array(coords_list) if coords_list else None

    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Retrieve the 2D + Depth coordinates for the helix zigzag ribbon.

        Calculates the ribbon shape based on start/end points of the pre-projected
        coordinate slice and style parameters. Handles minimum length.

        Args:
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array of the ribbon outline coordinates [X, Y, Depth],
            or a simple line [start, end] if below min_helix_length.
        """
        if self._cached_display_coords is not None:
            return self._cached_display_coords

        original_coords = self._get_original_coords_slice(structure)
        if original_coords is None or len(original_coords) == 0:
            self._cached_display_coords = None
            self._original_coords_len = 0
            return None

        self._original_coords_len = len(original_coords)

        if self._original_coords_len < 2:
            # If only one residue, return just that point
            self._cached_display_coords = np.array([original_coords[0]])
            return self._cached_display_coords

        # If too short, return a simple line (start and end points)
        if self._original_coords_len < self.style.min_helix_length:
            self._cached_display_coords = np.array(
                [original_coords[0], original_coords[-1]]
            )
            return self._cached_display_coords

        # Calculate zigzag points
        start_point_3d = original_coords[0]
        end_point_3d = original_coords[-1]

        zigzag_coords = calculate_zigzag_points(
            start_point_3d,
            end_point_3d,
            self.style.ribbon_thickness,
            self.style.wavelength,
            self.style.amplitude,
        )

        if (
            zigzag_coords is None and self._original_coords_len >= 2
        ):  # Log only if zigzag expected but failed
            raise CoordinateCalculationError(
                f"Could not generate zigzag points for helix '{self.id}' (length={self._original_coords_len}), likely zero length between endpoints."
            )

        self._cached_display_coords = zigzag_coords
        return self._cached_display_coords

    def get_coordinate_at_residue(
        self, residue: ResidueCoordinate, structure: Structure
    ) -> Optional[np.ndarray]:
        """Retrieves the specific 2D coordinate + Depth corresponding to a residue
        along the central axis of the helix representation.

        For short helices, interpolates linearly. For zigzag helices, finds the
        midpoint between the top and bottom ribbon points at the corresponding position.

        Args:
            residue: The residue coordinate (chain and index) to find the point for.
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array [X, Y, Depth] corresponding to the residue's position.
        """
        # 1. Ensure display coordinates are calculated
        display_coords = self.get_coordinates(structure)
        if (
            display_coords is None
            or self._original_coords_len is None
            or self._original_coords_len == 0
        ):
            return None

        # 2. Check if residue is within the element's range
        if residue not in self.residue_range_set:
            return None
        element_range = self.residue_range_set.ranges[0]  # Assuming single range
        if residue.chain_id != element_range.chain_id:
            return None

        # 3. Calculate the 0-based index within the original sequence length
        try:
            original_sequence_index = residue.residue_index - element_range.start
            if not (0 <= original_sequence_index < self._original_coords_len):
                return None
        except Exception:
            return None

        # Handle single point case
        if self._original_coords_len == 1:
            return display_coords[0]  # Should be shape (1, 3)

        # 4. Handle the case where a simple line was drawn
        if len(display_coords) == 2:
            # Linear interpolation along the line
            frac = original_sequence_index / (self._original_coords_len - 1)
            interpolated_coord = (
                display_coords[0] * (1 - frac) + display_coords[1] * frac
            )
            return interpolated_coord

        # 5. Handle the zigzag ribbon case
        # display_coords contains top points then bottom points reversed
        num_wave_points = len(display_coords) // 2  # Number of points along one edge

        if num_wave_points < 1:
            raise CoordinateCalculationError(
                f"Invalid number of wave points ({num_wave_points}) for helix {self.id}"
            )

        # Map original sequence index to fractional position along the wave points (0 to num_wave_points-1)
        mapped_wave_frac = (original_sequence_index * (num_wave_points - 1)) / (
            self._original_coords_len - 1
        )

        # Find the indices in the display_coords array
        idx_low = int(np.floor(mapped_wave_frac))
        idx_high = min(idx_low + 1, num_wave_points - 1)
        idx_low = min(idx_low, num_wave_points - 1)  # Clamp low index too
        frac = mapped_wave_frac - idx_low

        # Get corresponding points on top and bottom edges
        top_low = display_coords[idx_low]
        top_high = display_coords[idx_high]
        # Bottom indices are reversed: num_total - 1 - index
        bottom_low = display_coords[len(display_coords) - 1 - idx_low]
        bottom_high = display_coords[len(display_coords) - 1 - idx_high]

        # Interpolate along top and bottom edges
        interp_top = top_low * (1 - frac) + top_high * frac
        interp_bottom = bottom_low * (1 - frac) + bottom_high * frac

        # Return the midpoint between the interpolated top and bottom points
        return (interp_top + interp_bottom) / 2.0

    def get_start_connection_point(self, structure: Structure) -> Optional[np.ndarray]:
        """Calculate the 2D coordinate for the start connection point.

        Args:
            structure: The core Structure object containing projected coordinates.

        Returns:
            A NumPy array [X, Y] or None if calculation fails.
        """
        # Get the full 3D coordinates used for rendering
        display_coords = self.get_coordinates(structure)
        if display_coords is None or len(display_coords) == 0:
            return None

        coords_2d = display_coords[:, :2]  # Work with XY

        # If rendered as a simple line (2 points)
        if len(coords_2d) == 2:
            return coords_2d[0]

        # If rendered as zigzag (even number of points >= 4)
        if len(coords_2d) >= 4:
            # Midpoint of the starting edge
            # First point (top edge start) = coords_2d[0]
            # Corresponding bottom point (bottom edge start) = coords_2d[-1]
            return (coords_2d[0] + coords_2d[-1]) / 2.0

        # Fallback for unexpected cases (e.g., single point helix coord result)
        return coords_2d[0]  # Return the first point

    def get_end_connection_point(self, structure: Structure) -> Optional[np.ndarray]:
        """Calculate the 2D coordinate for the end connection point.

        Args:
            structure: The core Structure object containing projected coordinates.

        Returns:
            A NumPy array [X, Y] or None if calculation fails.
        """
        # Get the full 3D coordinates used for rendering
        display_coords = self.get_coordinates(structure)
        if display_coords is None or len(display_coords) == 0:
            return None

        coords_2d = display_coords[:, :2]  # Work with XY

        # If rendered as a simple line (2 points)
        if len(coords_2d) == 2:
            return coords_2d[1]

        # If rendered as zigzag (even number of points >= 4)
        if len(coords_2d) >= 4:
            # Midpoint of the ending edge
            # Last point of top edge = coords_2d[num_edge_points - 1]
            # Corresponding last point of bottom edge = coords_2d[num_edge_points]
            num_edge_points = len(coords_2d) // 2
            last_top_point = coords_2d[num_edge_points - 1]
            last_bottom_point = coords_2d[num_edge_points]
            return (last_top_point + last_bottom_point) / 2.0

        # Fallback for unexpected cases (e.g., single point helix coord result)
        return coords_2d[-1]  # Return the last point
