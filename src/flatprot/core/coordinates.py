# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Iterator, Optional, Union
import re

from .types import ResidueType, SecondaryStructureType

_RESIDUE_COORD_PATTERN = re.compile(r"^([^:]+):(\d+)$")


@dataclass(frozen=True)
class ResidueCoordinate:
    chain_id: str
    residue_index: int
    residue_type: Optional[ResidueType] = None
    coordinate_index: Optional[int] = None

    @staticmethod
    def from_string(string: str) -> "ResidueCoordinate":
        """Parse 'CHAIN:INDEX' format (e.g., 'A:123').

        Args:
            string: The string representation to parse.

        Returns:
            A ResidueCoordinate instance.

        Raises:
            ValueError: If the string format is invalid.
        """
        match = _RESIDUE_COORD_PATTERN.match(string)
        if not match:
            raise ValueError(
                f"Invalid ResidueCoordinate format: '{string}'. Expected 'CHAIN:INDEX'."
            )
        chain_name, residue_index_str = match.groups()
        return ResidueCoordinate(chain_name, int(residue_index_str))

    def __str__(self) -> str:
        if self.residue_type:
            return f"{self.chain_id}:{self.residue_index} ({self.residue_type.name})"
        else:
            return f"{self.chain_id}:{self.residue_index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueCoordinate):
            raise TypeError(f"Cannot compare ResidueCoordinate with {type(other)}")
        return (
            self.chain_id == other.chain_id
            and self.residue_index == other.residue_index
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ResidueCoordinate):
            raise TypeError(f"Cannot compare ResidueCoordinate with {type(other)}")

        if self.chain_id != other.chain_id:
            return self.chain_id < other.chain_id
        return self.residue_index < other.residue_index


_RESIDUE_RANGE_PATTERN = re.compile(r"^([^:]+):(\d+)-(\d+)$")


@dataclass(frozen=True)
class ResidueRange:
    """Represents a continuous range of residues within a single chain."""

    chain_id: str
    start: int
    end: int
    coordinates_start_index: Optional[int] = None
    secondary_structure_type: Optional[SecondaryStructureType] = None

    def to_set(self) -> "ResidueRangeSet":
        return ResidueRangeSet([self])

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Invalid range: {self.start} > {self.end}")

    def __iter__(self) -> Iterator[ResidueCoordinate]:
        """Iterate over all ResidueCoordinates in this range."""
        for i in range(self.start, self.end + 1):
            yield ResidueCoordinate(self.chain_id, i)

    def __len__(self) -> int:
        """Number of residues in this range."""
        return self.end - self.start + 1

    @staticmethod
    def from_string(string: str) -> "ResidueRange":
        """Parse 'CHAIN:START-END' format (e.g., 'A:14-20').

        Args:
            string: The string representation to parse.

        Returns:
            A ResidueRange instance.

        Raises:
            ValueError: If the string format is invalid or start > end.
        """
        match = _RESIDUE_RANGE_PATTERN.match(string)
        if not match:
            raise ValueError(
                f"Invalid ResidueRange format: '{string}'. Expected 'CHAIN:START-END'."
            )
        chain, start_str, end_str = match.groups()
        start, end = int(start_str), int(end_str)
        # The __post_init__ check for start > end will still run
        return ResidueRange(chain, start, end)

    def __str__(self) -> str:
        if self.secondary_structure_type:
            return f"{self.chain_id}:{self.start}-{self.end} ({self.secondary_structure_type.name})"
        else:
            return f"{self.chain_id}:{self.start}-{self.end}"

    def __lt__(self, other: "ResidueRange") -> bool:
        """Compare ranges for sorting.

        Sorts first by chain_id, then by start position.

        Args:
            other: Another ResidueRange to compare with

        Returns:
            bool: True if this range should come before the other
        """
        if not isinstance(other, ResidueRange):
            raise TypeError(f"Cannot compare ResidueRange with {type(other)}")

        if self.chain_id != other.chain_id:
            return self.chain_id < other.chain_id
        return self.start < other.start

    def __eq__(self, other: object) -> bool:
        """Check if two ranges are equal.

        Args:
            other: Another object to compare with

        Returns:
            bool: True if the ranges are equal
        """
        if not isinstance(other, ResidueRange):
            raise TypeError(f"Cannot compare ResidueRange with {type(other)}")

        return (
            self.chain_id == other.chain_id
            and self.start == other.start
            and self.end == other.end
        )

    def __contains__(self, other: object) -> bool:
        """Check if this range contains another range or residue coordinate.

        Args:
            other: Another object to check containment with

        Returns:
            bool: True if this range contains the other object
        """
        if isinstance(other, ResidueRange):
            return (
                self.chain_id == other.chain_id
                and self.start <= other.start
                and self.end >= other.end
            )
        elif isinstance(other, ResidueCoordinate):
            return (
                self.chain_id == other.chain_id
                and self.start <= other.residue_index <= self.end
            )

    def is_adjacent_to(self, other: Union["ResidueRange", ResidueCoordinate]) -> bool:
        """Check if this range is adjacent to another range or residue coordinate.

        Two ranges are considered adjacent if they are in the same chain and
        one range's end is exactly one residue before the other range's start.
        For example, A:10-15 is adjacent to A:16-20, but not to A:15-20 (overlap)
        or A:17-20 (gap).

        A range and a residue coordinate are considered adjacent if they are in the
        same chain and the coordinate is exactly one residue before the range's start
        or exactly one residue after the range's end.

        Args:
            other: The other range or residue coordinate to check adjacency with.

        Returns:
            bool: True if the range is adjacent to the other object, False otherwise.

        Raises:
            TypeError: If other is not a ResidueRange or ResidueCoordinate.
        """
        if not isinstance(other, (ResidueRange, ResidueCoordinate)):
            raise TypeError(f"Cannot check adjacency with {type(other)}")

        # Must be in the same chain to be adjacent
        if self.chain_id != other.chain_id:
            return False

        if isinstance(other, ResidueRange):
            # Check if self's end is exactly one residue before other's start
            if self.end + 1 == other.start:
                return True

            # Check if other's end is exactly one residue before self's start
            if other.end + 1 == self.start:
                return True
        else:  # ResidueCoordinate
            # Check if the coordinate is exactly one residue before the range's start
            if other.residue_index + 1 == self.start:
                return True

            # Check if the coordinate is exactly one residue after the range's end
            if self.end + 1 == other.residue_index:
                return True

        return False


class ResidueRangeSet:
    """Represents multiple ranges of residues, potentially across different chains."""

    def __init__(self, ranges: List[ResidueRange]):
        # Sort ranges for consistent representation and efficient searching
        self.ranges = sorted(ranges, key=lambda r: (r.chain_id, r.start))

    @staticmethod
    def from_string(string: str) -> "ResidueRangeSet":
        """Parse 'A:14-20, B:10-15' format.

        Ranges are separated by commas. Whitespace around ranges and commas
        is ignored.

        Args:
            string: The string representation to parse.

        Returns:
            A ResidueRangeSet instance.

        Raises:
            ValueError: If the string is empty, contains empty parts after
                      splitting by comma, or if any individual range
                      part is invalid according to ResidueRange.from_string.
        """
        if not string:
            raise ValueError("Input string for ResidueRangeSet cannot be empty.")

        parts = string.split(",")
        ranges = []
        for part in parts:
            stripped_part = part.strip()
            if not stripped_part:
                # Handle cases like "A:1-5,,B:6-10" or leading/trailing commas
                raise ValueError(
                    f"Empty range part found in string: '{string}' after splitting by comma."
                )
            try:
                # Leverage ResidueRange's parsing and validation
                ranges.append(ResidueRange.from_string(stripped_part))
            except ValueError as e:
                # Re-raise with context about which part failed
                raise ValueError(
                    f"Error parsing range part '{stripped_part}' in '{string}': {e}"
                ) from e

        if not ranges:
            # This case might be redundant if empty string/parts are caught above,
            # but kept for robustness, e.g., if string only contains commas/whitespace.
            raise ValueError(f"No valid residue ranges found in string: '{string}'.")

        return ResidueRangeSet(ranges)

    def __iter__(self) -> Iterator[ResidueCoordinate]:
        """Iterate over all ResidueCoordinates in all ranges."""
        for range_ in self.ranges:
            yield from range_

    def __len__(self) -> int:
        """Total number of residues across all ranges."""
        return sum(len(range_) for range_ in self.ranges)

    def __str__(self) -> str:
        return ",".join(str(r) for r in self.ranges)

    def __repr__(self) -> str:
        return f"ResidueRangeSet({self.__str__()})"

    def __contains__(self, other: object) -> bool:
        """Check if this range set contains another range or residue coordinate.

        Args:
            other: Another object to check containment with

        Returns:
            bool: True if this range set contains the other object
        """
        if isinstance(other, ResidueRange):
            return any(other in range_ for range_ in self.ranges)
        elif isinstance(other, ResidueCoordinate):
            return any(other in range_ for range_ in self.ranges)
        return False

    def __eq__(self, other: object) -> bool:
        """Check if two range sets are equal.

        Args:
            other: Another object to compare with

        Returns:
            bool: True if the range sets are equal
        """
        if not isinstance(other, ResidueRangeSet):
            raise TypeError(f"Cannot compare ResidueRangeSet with {type(other)}")

        return sorted(self.ranges) == sorted(other.ranges)

    def is_adjacent_to(
        self, other: Union["ResidueRangeSet", ResidueRange, ResidueCoordinate]
    ) -> bool:
        """Check if this range set is adjacent to another range set, range, or coordinate.

        Two range sets are considered adjacent if any range in one set is
        adjacent to any range in the other set. Ranges are adjacent if they
        are in the same chain and one range's end is exactly one residue
        before the other range's start, or vice versa.

        Args:
            other: The other object to check adjacency with. Can be a ResidueRangeSet,
                  ResidueRange, or ResidueCoordinate.

        Returns:
            bool: True if the range sets are adjacent, False otherwise.

        Raises:
            TypeError: If other is not a ResidueRangeSet, ResidueRange, or ResidueCoordinate.
        """
        if isinstance(other, ResidueRangeSet):
            # Check each pair of ranges from both sets
            for range1 in self.ranges:
                for range2 in other.ranges:
                    if range1.is_adjacent_to(range2):
                        return True
        elif isinstance(other, (ResidueRange, ResidueCoordinate)):
            # Convert the single range to a set and check adjacency
            for range1 in self.ranges:
                if range1.is_adjacent_to(other):
                    return True
        else:
            raise TypeError(f"Cannot check adjacency with {type(other)}")

        return False
