# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from .base import StructureSceneElement
from .coil import CoilElement
from .helix import HelixElement
from .sheet import SheetElement

from flatprot.core import SecondaryStructure, SecondaryStructureType
from flatprot.style import StyleManager, StyleType

import numpy as np


__all__ = [
    "StructureSceneElement",
    "CoilElement",
    "HelixElement",
    "SheetElement",
    "secondary_structure_to_scene_element",
]


def secondary_structure_to_scene_element(
    element: SecondaryStructure,
    coordinates: np.ndarray,
    style_manager: StyleManager,
    metadata: Optional[dict] = None,
) -> StructureSceneElement:
    """Convert a secondary structure to a visualization element."""
    if element.secondary_structure_type == SecondaryStructureType.HELIX:
        return HelixElement(
            coordinates,
            style_manager,
            StyleType.HELIX,
            metadata,
        )
    elif element.secondary_structure_type == SecondaryStructureType.SHEET:
        return SheetElement(
            coordinates,
            style_manager,
            StyleType.SHEET,
            metadata,
        )
    elif element.secondary_structure_type == SecondaryStructureType.COIL:
        return CoilElement(
            coordinates,
            style_manager,
            StyleType.COIL,
            metadata,
        )
    raise ValueError(
        f"Unknown secondary structure type: {element.secondary_structure_type}"
    )
