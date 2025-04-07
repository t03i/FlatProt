# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from .error import FlatProtError, CoordinateError
from .types import (
    ResidueType,
)

from .coordinates import (
    ResidueCoordinate,
    ResidueRange,
    ResidueRangeSet,
)
from .secondary import (
    SecondaryStructure,
    SecondaryStructureType,
    Helix,
    Sheet,
    Coil,
)
from .components import (
    Chain,
    Structure,
)
from .manager import (
    CoordinateManager,
    CoordinateType,
)

__all__ = [
    # Error types
    "FlatProtError",
    "CoordinateError",
    # Residue types and handling
    "ResidueType",
    "ResidueCoordinate",
    "ResidueRange",
    "ResidueRangeSet",
    # Secondary structure
    "SecondaryStructure",
    "SecondaryStructureType",
    "Helix",
    "Sheet",
    "Coil",
    # Structure components
    "Chain",
    "Structure",
    # Coordinate management
    "CoordinateManager",
    "CoordinateType",
]
