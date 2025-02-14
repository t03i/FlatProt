# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from flatprot.core import SecondaryStructureType
from flatprot.style import StyleManager, StyleType
from .structure import (
    StructureSceneElement,
    HelixElement,
    SheetElement,
    CoilElement,
)

import numpy as np


def secondary_structure_to_scene_element(
    secondary_structure: SecondaryStructureType,
    coordinates: np.ndarray,
    metadata: dict,
    style_manager: StyleManager,
) -> StructureSceneElement:
    """Convert a secondary structure to a visualization element."""
    if secondary_structure == SecondaryStructureType.HELIX:
        return HelixElement(coordinates, metadata, style_manager, StyleType.HELIX)
    elif secondary_structure == SecondaryStructureType.SHEET:
        return SheetElement(coordinates, metadata, style_manager, StyleType.SHEET)
    elif secondary_structure == SecondaryStructureType.COIL:
        return CoilElement(coordinates, metadata, style_manager, StyleType.COIL)
    raise ValueError(f"Unknown secondary structure type: {secondary_structure}")
