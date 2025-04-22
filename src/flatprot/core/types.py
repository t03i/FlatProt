# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class SecondaryStructureType(str, Enum):
    HELIX = "H"
    SHEET = "S"
    COIL = "O"


class ResidueType(str, Enum):
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
