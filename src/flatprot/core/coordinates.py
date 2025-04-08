# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Iterator, Optional

from .types import ResidueType, SecondaryStructureType


@dataclass(frozen=True)
class ResidueCoordinate:
    chain_id: str
    residue_index: int
    residue_type: Optional[ResidueType] = None
    coordinate_index: Optional[int] = None

    @staticmethod
    def from_string(string: str) -> "ResidueCoordinate":
        chain_name, residue_index = string.split(":")
        return ResidueCoordinate(chain_name, int(residue_index))

    def __str__(self) -> str:
        if self.residue_type:
            return f"{self.chain_id}:{self.residue_index} ({self.residue_type})"
        else:
            return f"{self.chain_id}:{self.residue_index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueCoordinate):
            return False
        return (
            self.chain_id == other.chain_id
            and self.residue_index == other.residue_index
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ResidueCoordinate):
            return False

        if self.chain_id != other.chain_id:
            return self.chain_id < other.chain_id
        return self.residue_index < other.residue_index


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
        """Parse 'A:14-20' format."""
        chain, range_part = string.split(":")
        start, end = map(int, range_part.split("-"))
        return ResidueRange(chain, start, end)

    def __str__(self) -> str:
        if self.secondary_structure:
            return (
                f"{self.chain_id}:{self.start}-{self.end} ({self.secondary_structure})"
            )
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
            return False
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
        self.ranges = sorted(ranges, key=lambda r: (r.chain_id, r.start))

    @staticmethod
    def from_string(string: str) -> "ResidueRangeSet":
        """Parse 'A:14-20,B:10-15' format."""
        parts = string.split(",")
        ranges = [ResidueRange.from_string(part.strip()) for part in parts]
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
