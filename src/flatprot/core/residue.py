# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from dataclasses import dataclass
from typing import List, Iterator


@dataclass(frozen=True)
class ResidueCoordinate:
    chain_name: str
    residue_index: int

    @staticmethod
    def from_string(string: str) -> "ResidueCoordinate":
        chain_name, residue_index = string.split(":")
        return ResidueCoordinate(chain_name, int(residue_index))

    def __str__(self) -> str:
        return f"{self.chain_name}:{self.residue_index}"


@dataclass(frozen=True)
class ResidueRange:
    """Represents a continuous range of residues within a single chain."""

    chain_name: str
    start: int
    end: int

    def __iter__(self) -> Iterator[ResidueCoordinate]:
        """Iterate over all ResidueCoordinates in this range."""
        for i in range(self.start, self.end + 1):
            yield ResidueCoordinate(self.chain_name, i)

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
        return f"{self.chain_name}:{self.start}-{self.end}"


class ResidueRangeSet:
    """Represents multiple ranges of residues, potentially across different chains."""

    def __init__(self, ranges: List[ResidueRange]):
        self.ranges = sorted(ranges, key=lambda r: (r.chain_name, r.start))

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


class Residue(str, Enum):
    # Standard amino acids
    ALA = "A"  # Alanine
    ARG = "R"  # Arginine
    ASN = "N"  # Asparagine
    ASP = "D"  # Aspartate
    CYS = "C"  # Cysteine
    GLN = "Q"  # Glutamine
    GLU = "E"  # Glutamate
    GLY = "G"  # Glycine
    HIS = "H"  # Histidine
    ILE = "I"  # Isoleucine
    LEU = "L"  # Leucine
    LYS = "K"  # Lysine
    MET = "M"  # Methionine
    PHE = "F"  # Phenylalanine
    PRO = "P"  # Proline
    SER = "S"  # Serine
    THR = "T"  # Threonine
    TRP = "W"  # Tryptophan
    TYR = "Y"  # Tyrosine
    VAL = "V"  # Valine

    # Special amino acids
    SEC = "U"  # Selenocysteine
    PYL = "O"  # Pyrrolysine

    # Ambiguous amino acids
    XAA = "X"  # Any/unknown
    ASX = "B"  # Asparagine or Aspartate
    GLX = "Z"  # Glutamine or Glutamate
    XLE = "J"  # Leucine or Isoleucine
