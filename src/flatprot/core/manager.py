# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
import numpy as np
from typing import (
    Optional,
    Dict,
    List,
    Tuple,
    Callable,
    overload,
    Union,
    Literal,
)

from .coordinates import ResidueCoordinate, ResidueRange, ResidueRangeSet
from .error import CoordinateError


class CoordinateType(Enum):
    """Enumeration for different types of coordinate representations."""

    COORDINATES = "coordinates"  # Original coordinates from input
    TRANSFORMED = "transformed"  # Coordinates after geometric transformations
    CANVAS = "canvas"  # Coordinates projected onto the 2D canvas
    DEPTH = "depth"  # Depth values (e.g., z-coordinates) for occlusion


class CoordinateManager:
    """Manages different types of coordinates for protein residues.

    This class stores and retrieves coordinate data (e.g., original 3D,
    transformed, 2D canvas projections) associated with specific residues or
    ranges of residues, identified by chain ID and residue index.

    Attributes:
        _coordinates: A nested dictionary storing coordinates. The structure is:
            {CoordinateType: {ResidueCoordinate: np.ndarray}}
    """

    def __init__(self) -> None:
        """Initialize an empty coordinate manager.

        Sets up the internal storage dictionary for each coordinate type.
        """
        self._coordinates: Dict[CoordinateType, Dict[ResidueCoordinate, np.ndarray]] = {
            coord_type: {} for coord_type in CoordinateType
        }

    def add_residue_coordinate(
        self,
        residue_coordinate: ResidueCoordinate,
        coords: np.ndarray,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> None:
        """Add coordinates for a single residue.

        Args:
            residue_coordinate: The specific residue (chain and index) to add
                                coordinates for.
            coords: Numpy array representing the coordinates for this residue.
            coord_type: The type of coordinates being added.
        """
        # Ensure coords is a numpy array
        coord_array = np.asarray(coords)
        target_map = self._coordinates[coord_type]
        target_map[residue_coordinate] = coord_array

    def add_range(
        self,
        residue_range: ResidueRange,
        coords: np.ndarray,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> None:
        """Add coordinates for a continuous range of residues.

        The input `coords` array must have the same length as the `residue_range`.

        Args:
            residue_range: The continuous range of residues.
            coords: A numpy array of coordinates. `coords[i]` corresponds to the
                    i-th residue in the `residue_range` iteration order.
            coord_type: The type of coordinates being added.

        Raises:
            CoordinateError: If the length of the `coords` array does not match
                             the number of residues in `residue_range`.
        """
        coord_array = np.asarray(coords)
        if coord_array.shape[0] != len(residue_range):
            raise CoordinateError(
                f"Coordinate array length {coord_array.shape[0]} does not match "
                f"range length {len(residue_range)} for range {residue_range}"
            )

        target_map = self._coordinates[coord_type]
        for i, residue in enumerate(residue_range):
            target_map[residue] = coord_array[i]

    def add_range_set(
        self,
        range_set: ResidueRangeSet,
        coords: np.ndarray,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> None:
        """Add coordinates for multiple residue ranges defined by a ResidueRangeSet.

        The input `coords` array must have a length equal to the total number
        of residues in the `range_set`. Coordinates are assigned sequentially
        based on the iteration order of the `range_set`.

        Args:
            range_set: The set of residue ranges.
            coords: A numpy array of coordinates. The length must match `len(range_set)`.
            coord_type: The type of coordinates being added.

        Raises:
            CoordinateError: If the length of the `coords` array does not match
                             the total number of residues in `range_set`.
        """
        coord_array = np.asarray(coords)
        if coord_array.shape[0] != len(range_set):
            raise CoordinateError(
                f"Coordinate array length {coord_array.shape[0]} does not match "
                f"total range length {len(range_set)} for set {range_set}"
            )

        target_map = self._coordinates[coord_type]
        offset = 0
        for range_ in range_set.ranges:
            length = len(range_)
            # Add coordinates for the current range
            for i, residue in enumerate(range_):
                target_map[residue] = coord_array[offset + i]
            offset += length

    def get_residue_coordinate(
        self,
        residue_coordinate: ResidueCoordinate,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> np.ndarray:
        """Get coordinates for a specific residue.

        Args:
            residue_coordinate: The specific residue coordinate to retrieve.
            coord_type: Type of coordinates to retrieve.

        Returns:
            Numpy array containing the requested coordinate.

        Raises:
            CoordinateError: If no coordinates exist for the specified residue coordinate
                             and coordinate type.
        """
        try:
            target_map = self._coordinates[coord_type]
            return target_map[residue_coordinate]
        except KeyError:
            raise CoordinateError(
                f"No {coord_type.value} coordinates found for {residue_coordinate}"
            ) from None

    def get_range(
        self,
        residue_range: ResidueRange,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> np.ndarray:
        """Get coordinates for a continuous range of residues.

        Args:
            residue_range: The continuous range of residues to retrieve coordinates for.
            coord_type: Type of coordinates to retrieve.

        Returns:
            A numpy array containing the coordinates for the specified range,
            ordered by residue index. The shape will be (N, D), where N is the
            number of residues in the range and D is the coordinate dimension.

        Raises:
            CoordinateError: If coordinates are missing for any residue in the range,
                             or if the coordinate type map is empty and dimensions cannot be determined.
        """
        coords = []
        target_map = self._coordinates[coord_type]

        # Check if target map is empty before proceeding
        if not target_map and len(residue_range) > 0:
            raise CoordinateError(f"No coordinates of type {coord_type.value} exist.")

        for residue in residue_range:
            try:
                coord = target_map[residue]
                coords.append(coord)
            except KeyError:
                raise CoordinateError(
                    f"No {coord_type.value} coordinates found for {residue} within range {residue_range}"
                ) from None

        if not coords:
            # Range was empty (e.g., length 0, which shouldn't happen for valid ResidueRange)
            # Or target_map was empty and range had length 0
            # Determine coordinate dimension from the first available coord if possible
            if not target_map:
                # Cannot determine shape if no coords of this type exist at all
                return np.array([])  # Return empty 1D array

            # If target_map is not empty, get shape from an arbitrary element
            any_coord_key = next(iter(target_map))
            coord_shape_tail = target_map[any_coord_key].shape[
                1:
            ]  # Shape excluding the residue dimension
            dtype = target_map[any_coord_key].dtype
            # Return shape (0, *coord_shape_tail)
            return np.empty((0, *coord_shape_tail), dtype=dtype)

        # Stack along a new first axis -> (N, *coord_shape_tail)
        return np.stack(coords)

    def get_range_set(
        self,
        range_set: ResidueRangeSet,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> np.ndarray:
        """Get coordinates for multiple residue ranges specified by a ResidueRangeSet.

        Args:
            range_set: The set of residue ranges to retrieve coordinates for.
            coord_type: Type of coordinates to retrieve.

        Returns:
            A numpy array containing the concatenated coordinates for all ranges
            in the set, ordered first by range (as sorted in the ResidueRangeSet)
            and then by residue index within each range. Shape is (N_total, D).
            Returns an empty array with appropriate dimensions if the set is empty
            or contains only empty ranges.

        Raises:
            CoordinateError: If coordinates are missing for any residue in any range,
                             or if the coordinate type map is empty and dimensions cannot be determined.
        """
        coords_list: List[np.ndarray] = []
        target_map = self._coordinates[coord_type]

        # Handle empty coordinate type or empty range set
        if not target_map:
            if len(range_set) == 0:  # If set is empty, return empty array
                return np.array([])
            else:  # Set is not empty, but no coords of this type exist
                raise CoordinateError(
                    f"No coordinates of type {coord_type.value} exist."
                )

        # Determine coordinate dimension and dtype from the first available coord
        # Needed for returning empty array with correct shape later if necessary
        try:
            first_key = next(iter(target_map))
            coord_shape_tail = target_map[first_key].shape[1:]
            dtype = target_map[first_key].dtype
            empty_array_shape = (0, *coord_shape_tail)
        except StopIteration:  # target_map is empty
            if len(range_set) == 0:
                return np.array([])
            else:
                raise CoordinateError(
                    f"No coordinates of type {coord_type.value} exist."
                )

        if not range_set.ranges:  # Handle empty range set explicitly
            return np.empty(empty_array_shape, dtype=dtype)

        for range_ in range_set.ranges:
            # Reuse get_range logic efficiently within this loop
            range_coords: List[np.ndarray] = []
            if len(range_) == 0:  # Skip empty ranges within the set
                continue
            for residue in range_:
                try:
                    coord = target_map[residue]
                    range_coords.append(coord)
                except KeyError:
                    raise CoordinateError(
                        f"No {coord_type.value} coordinates found for {residue} within range set {range_set}"
                    ) from None
            if (
                range_coords
            ):  # Only stack and append if the range actually yielded coordinates
                coords_list.append(np.stack(range_coords))

        if not coords_list:
            # This happens if range_set contained only empty ranges
            return np.empty(empty_array_shape, dtype=dtype)

        # Concatenate results from all non-empty ranges
        return np.concatenate(coords_list)

    def get_chain(
        self,
        chain_id: str,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> np.ndarray:
        """Get all coordinates for a specific chain, sorted by residue index.

        Args:
            chain_id: The ID of the chain to retrieve coordinates for.
            coord_type: Type of coordinates to retrieve.

        Returns:
            A numpy array containing all coordinates for the specified chain,
            sorted by residue index. Shape is (N_chain, D). Returns an empty
            array with appropriate dimensions if the chain has no coordinates
            of the specified type.

        Raises:
            CoordinateError: If the coordinate type map is empty and dimensions cannot be determined.

        """
        target_map = self._coordinates[coord_type]

        if not target_map:
            # Cannot determine shape if no coords of this type exist at all
            return np.array([])  # Return empty 1D array

        # Determine coordinate dimension and dtype from the first available coord
        try:
            first_key = next(iter(target_map))
            coord_shape_tail = target_map[first_key].shape[1:]
            dtype = target_map[first_key].dtype
            empty_array_shape = (0, *coord_shape_tail)
        except StopIteration:  # target_map is empty
            return np.array([])

        # Filter coordinates by chain_id and collect (residue_index, coordinate_array) pairs
        chain_coords_dict: Dict[int, np.ndarray] = {}
        for residue_coord, coord_array in target_map.items():
            if residue_coord.chain_id == chain_id:
                chain_coords_dict[residue_coord.residue_index] = coord_array

        if not chain_coords_dict:
            # Chain exists but has no coordinates of this type, or chain not found
            return np.empty(empty_array_shape, dtype=dtype)

        # Sort by residue index and extract coordinate arrays
        sorted_indices = sorted(chain_coords_dict.keys())
        sorted_coords = [chain_coords_dict[index] for index in sorted_indices]

        # Stack coordinates into a single numpy array
        return np.stack(sorted_coords)

    def has_type(self, coord_type: CoordinateType) -> bool:
        """Check if any coordinates of a specific type exist.

        Args:
            coord_type: Type of coordinates to check for.

        Returns:
            True if coordinates of the specified type exist, False otherwise.
        """
        return coord_type in self._coordinates and bool(self._coordinates[coord_type])

    @overload
    def get_all(
        self, coord_type: CoordinateType, return_keys: Literal[True]
    ) -> Tuple[np.ndarray, List[ResidueCoordinate]]:
        ...

    @overload
    def get_all(
        self, coord_type: CoordinateType, return_keys: Literal[False] = False
    ) -> np.ndarray:
        ...

    def get_all(
        self, coord_type: CoordinateType, return_keys: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[ResidueCoordinate]]]:
        """Get all coordinates of a specific type, sorted by chain ID and residue index.

        This function concatenates all coordinate segments of the specified type
        in a predictable order.

        Args:
            coord_type: Type of coordinates to retrieve.
            return_keys: If True, also return the list of ResidueCoordinate keys
                         corresponding to the rows in the returned array.

        Returns:
            If return_keys is False (default):
                Numpy array containing all coordinates of the specified type, sorted
                first by chain ID and then by residue index. Shape is (N_total, D).
                Returns an empty array with appropriate dimensions if no coordinates
                of this type exist.
            If return_keys is True:
                A tuple containing:
                - The numpy array of coordinates (as above).
                - A list of ResidueCoordinate objects corresponding to each row
                  of the coordinate array.

        """
        target_map = self._coordinates[coord_type]
        if not target_map:
            empty_coords = np.array([])
            empty_keys: List[ResidueCoordinate] = []
            return (empty_coords, empty_keys) if return_keys else empty_coords

        # Determine coordinate dimension and dtype from the first available coord
        # Use try-except for robustness although target_map is not empty here
        try:
            first_key = next(iter(target_map))
            coord_shape_tail = target_map[first_key].shape[1:]
            dtype = target_map[first_key].dtype
            empty_array_shape = (0, *coord_shape_tail)
        except StopIteration:
            # Should not be reachable if target_map is not empty, but as safeguard
            empty_coords = np.array([])
            empty_keys: List[ResidueCoordinate] = []
            return (empty_coords, empty_keys) if return_keys else empty_coords

        # Sort keys (ResidueCoordinate) first by chain_id, then by residue_index
        sorted_keys = sorted(
            target_map.keys(), key=lambda rc: (rc.chain_id, rc.residue_index)
        )

        if not sorted_keys:
            empty_coords = np.empty(empty_array_shape, dtype=dtype)
            empty_keys: List[ResidueCoordinate] = []
            return (empty_coords, empty_keys) if return_keys else empty_coords

        # Extract coordinate arrays in the sorted order
        sorted_coords_list = [target_map[key] for key in sorted_keys]

        # Stack all coordinate arrays
        stacked_coords = np.stack(sorted_coords_list)

        if return_keys:
            return stacked_coords, sorted_keys
        else:
            return stacked_coords

    @staticmethod
    def get_bounds(coords: np.ndarray) -> Optional[np.ndarray]:
        """Calculate the 2D bounding box [min_x, min_y, max_x, max_y].

        Assumes input coordinates are at least 2D (e.g., canvas coordinates).
        Returns None if the input array is empty or not suitable for bounds calculation.

        Args:
            coords: Coordinate array, typically of shape (N, 2) or (N, 3).

        Returns:
            Numpy array with bounding box coordinates [min_x, min_y, max_x, max_y],
            or None if bounds cannot be determined (e.g., empty array, wrong dimensions).
        """
        if (
            not isinstance(coords, np.ndarray)
            or coords.ndim < 2
            or coords.shape[0] == 0
            or coords.shape[1] < 2
        ):
            return None

        try:
            min_vals = np.min(coords[:, :2], axis=0)
            max_vals = np.max(coords[:, :2], axis=0)
        except IndexError:  # Should not happen with ndim check, but as safeguard
            return None

        return np.array([min_vals[0], min_vals[1], max_vals[0], max_vals[1]])

    def apply_vectorized_transform(
        self,
        source_coord_type: CoordinateType,
        target_coord_type: CoordinateType,
        transform_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Apply a vectorized transformation function to all coordinates of a source type
        and store the results under a target type.

        This method retrieves all coordinates for the `source_coord_type` as a single
        NumPy array, passes this array to the `transform_func`, and stores the
        resulting transformed coordinates under the `target_coord_type`.

        The `transform_func` must be capable of processing a batch of coordinates,
        typically an array of shape (N, D), where N is the total number of residues
        and D is the coordinate dimension. It should return an array of shape (N, D'),
        where D' is the dimension of the transformed coordinates.

        Args:
            source_coord_type: The coordinate type to read from.
            target_coord_type: The coordinate type to write the results to.
                               Any existing data for this type will be cleared
                               and replaced with the transformation results.
            transform_func: A callable that takes a NumPy array of shape (N, D)
                            and returns a transformed NumPy array of shape (N, D').

        Raises:
            CoordinateError: If no coordinates exist for the `source_coord_type`.
            ValueError: If the `transform_func` returns an array with a different
                        number of rows (N) than the input array.
        """
        source_map = self._coordinates[source_coord_type]
        if not source_map:
            raise CoordinateError(
                f"Cannot apply transform: No coordinates found for source type {source_coord_type.value}"
            )

        # Get all coordinates and corresponding sorted keys directly from get_all
        all_source_coords, sorted_keys = self.get_all(
            source_coord_type, return_keys=True
        )

        # Check if coordinates were actually returned
        if not sorted_keys or all_source_coords.size == 0:
            # No source coordinates, clear target and return
            self._coordinates[target_coord_type].clear()
            return

        # Apply the vectorized transformation function
        all_transformed_coords = transform_func(all_source_coords)

        # Ensure result is a numpy array
        all_transformed_coords = np.asarray(all_transformed_coords)

        # Validate the shape of the result
        if all_transformed_coords.shape[0] != len(sorted_keys):
            raise ValueError(
                f"Transformation function returned an array with {all_transformed_coords.shape[0]} rows, "
                f"but expected {len(sorted_keys)} rows matching the input."
            )

        # Populate the target map
        target_map = self._coordinates[target_coord_type]
        target_map.clear()

        for key, transformed_coord in zip(sorted_keys, all_transformed_coords):
            target_map[key] = transformed_coord
