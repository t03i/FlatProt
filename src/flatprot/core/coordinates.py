# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Iterator, Optional
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
    secondary_structure: Optional[SecondaryStructureType] = None

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
        if self.secondary_structure:
            return f"{self.chain_id}:{self.start}-{self.end} ({self.secondary_structure.name})"
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
