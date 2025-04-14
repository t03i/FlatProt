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
        fill=None,
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
    """Draws a Helix element as a filled path, or a line if too short."""
    coords = element.get_coordinates(structure)
    style = element.style

    if coords is None or len(coords) < 2:
        logger.debug(f"Skipping Helix {element.id}: Less than 2 coordinates.")
        return None

    # Common arguments
    kwargs = {
        "stroke_width": style.stroke_width,
        "stroke_opacity": style.opacity,
        "class_": "element helix",
        "id": element.id,
    }
    d_string = ""

    if len(coords) == 2:
        logger.debug(
            f"Drawing Helix {element.id} as line: Only 2 coordinates available."
        )
        # Line style
        kwargs["fill"] = None  # Explicitly set fill to none for lines
        kwargs["stroke"] = style.color.as_hex()  # Use main color for line stroke
        kwargs["linecap"] = "round"
        # Build path data
        d_string = f"M {coords[0, 0]},{coords[0, 1]} L {coords[1, 0]},{coords[1, 1]}"
    else:  # len(coords) >= 3
        # Filled polygon style
        kwargs["fill"] = style.color.as_hex()
        kwargs["fill_opacity"] = style.opacity
        kwargs["stroke"] = style.stroke_color.as_hex()  # Use stroke_color for outline
        # Build path data
        d_parts = [f"M {coords[0, 0]},{coords[0, 1]}"]
        for point in coords[1:]:
            d_parts.append(f"L {point[0]},{point[1]}")
        d_parts.append("Z")
        d_string = " ".join(d_parts)

    # Create Path object with all arguments
    path = Path(d=d_string, **kwargs)
    return path


def _draw_sheet(element: SheetSceneElement, structure: Structure) -> Optional[Path]:
    """Draws a Sheet element as a filled path (arrow), or a line if too short."""
    coords = element.get_coordinates(structure)
    style = element.style

    if coords is None or len(coords) < 2:
        logger.debug(f"Skipping Sheet {element.id}: Less than 2 coordinates.")
        return None

    # Common arguments
    kwargs = {
        "stroke_width": style.stroke_width,
        "stroke_opacity": style.opacity,
        "class_": "element sheet",
        "id": element.id,
    }
    d_string = ""

    if len(coords) == 2:
        logger.debug(
            f"Drawing Sheet {element.id} as line: Only 2 coordinates available."
        )
        # Line style
        kwargs["fill"] = None
        kwargs["stroke"] = style.color.as_hex()  # Use main color for line stroke
        kwargs["linecap"] = "round"
        # Build path data
        d_string = f"M {coords[0, 0]},{coords[0, 1]} L {coords[1, 0]},{coords[1, 1]}"
    else:  # len(coords) >= 3
        # Filled arrow style
        kwargs["fill"] = style.color.as_hex()
        kwargs["fill_opacity"] = style.opacity
        kwargs["stroke"] = style.stroke_color.as_hex()  # Use stroke_color for outline
        # Build path data
        d_parts = [f"M {coords[0, 0]},{coords[0, 1]}"]
        for point in coords[1:]:
            d_parts.append(f"L {point[0]},{point[1]}")
        d_parts.append("Z")
        d_string = " ".join(d_parts)

    # Create Path object with all arguments
    path = Path(d=d_string, **kwargs)
    return path
