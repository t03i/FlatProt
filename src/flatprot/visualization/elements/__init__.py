# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from flatprot.core import SecondaryStructure, Helix, Sheet, Coil

from .base import (
    VisualizationStyle,
    SmoothingMixin,
    VisualizationElement,
)

from .coil import CoilVisualization, CoilStyle
from .helix import HelixVisualization, HelixStyle
from .sheet import SheetVisualization, SheetStyle

from .style import StyleManager


def secondary_structure_to_visualization_element(
    secondary_structure: SecondaryStructure,
) -> type[VisualizationElement]:
    """Convert a secondary structure to a visualization element."""

    match type(secondary_structure):
        case Helix:
            return HelixVisualization
        case Sheet:
            return SheetVisualization
        case Coil:
            return CoilVisualization
        case _:
            raise ValueError(
                f"Unknown secondary structure type: {type(secondary_structure)}"
            )


__all__ = [
    "VisualizationStyle",
    "SmoothingMixin",
    "VisualizationElement",
    "CoilVisualization",
    "CoilStyle",
    "HelixVisualization",
    "HelixStyle",
    "SheetVisualization",
    "SheetStyle",
    "StyleManager",
]
