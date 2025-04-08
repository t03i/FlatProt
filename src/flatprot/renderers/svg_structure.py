# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from drawsvg import Path

from flatprot.scene import (
    HelixSceneElement,
    SheetSceneElement,
    CoilSceneElement,
)
from flatprot.core import Structure
from flatprot.core.logger import logger


def _draw_coil(element: CoilSceneElement, structure: Structure) -> Optional[Path]:
    """Draws a Coil element as a path."""
    coords = element.get_coordinates(structure)
    if coords is None or len(coords) < 2:
        logger.warning(f"Skipping Coil {element.id}: Insufficient coordinates.")
        return None

    style = element.style
    path = Path(
        stroke=style.stroke_color.as_hex(),
        stroke_width=style.stroke_width,
        fill="none",  # Coils are typically lines
        stroke_opacity=style.opacity,
        class_="element coil",
        id=element.id,
        linecap="round",
    )
    path.M(coords[0, 0], coords[0, 1])  # Move to start
    for point in coords[1:]:
        path.L(point[0], point[1])  # Line to next point
    return path


def _draw_helix(element: HelixSceneElement, structure: Structure) -> Optional[Path]:
    """Draws a Helix element as a filled path."""
    coords = element.get_coordinates(structure)
    if coords is None or len(coords) < 3:
        logger.warning(
            f"Skipping Helix {element.id}: Insufficient coordinates for polygon."
        )
        # TODO: Draw as line if too short based on style?
        return None

    style = element.style
    path = Path(
        stroke=style.stroke_color.as_hex(),
        stroke_width=style.stroke_width,
        fill=style.color.as_hex(),
        fill_opacity=style.opacity,  # Use main opacity for fill
        stroke_opacity=style.opacity,
        class_="element helix",
        id=element.id,
    )
    path.M(coords[0, 0], coords[0, 1])
    for point in coords[1:]:
        path.L(point[0], point[1])
    path.Z()  # Close the path
    return path


def _draw_sheet(element: SheetSceneElement, structure: Structure) -> Optional[Path]:
    """Draws a Sheet element as a filled path (arrow)."""
    coords = element.get_coordinates(structure)
    if coords is None or len(coords) < 3:
        logger.warning(
            f"Skipping Sheet {element.id}: Insufficient coordinates for arrow."
        )
        # TODO: Draw as line if too short based on style?
        return None

    style = element.style
    path = Path(
        stroke=style.stroke_color.as_hex(),
        stroke_width=style.stroke_width,
        fill=style.color.as_hex(),
        fill_opacity=style.opacity,
        stroke_opacity=style.opacity,
        class_="element sheet",
        id=element.id,
    )
    # Assumes get_coordinates returns the points of the arrow shape
    path.M(coords[0, 0], coords[0, 1])
    for point in coords[1:]:
        path.L(point[0], point[1])
    path.Z()  # Close the path
    return path
