# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any
import numpy as np  # Added for type hints
from drawsvg import Path, Line, Circle

from flatprot.scene import (
    HelixSceneElement,
    SheetSceneElement,
    CoilSceneElement,
)

from flatprot.scene.structure.coil import CoilStyle
from flatprot.core.logger import logger


def _draw_coil(
    element: CoilSceneElement,
    coords_2d: np.ndarray,
) -> Optional[Path | Circle]:
    """Draws a Coil element as a path or a small circle for single points."""
    style = element.style

    if coords_2d is None or len(coords_2d) < 1:
        logger.warning(f"Skipping Coil {element.id}: No coordinates.")
        return None
    elif len(coords_2d) == 1:
        logger.debug(
            f"Skipping 1 residue Coil {element.id}. Rendered through connections"
        )
        return None
    # Build the path string directly from coordinates
    d_parts = [f"M {coords_2d[0, 0]},{coords_2d[0, 1]}"]
    for point in coords_2d[1:]:
        d_parts.append(f"L {point[0]},{point[1]}")

    d_string = " ".join(d_parts)

    # Create Path object
    path = Path(
        d=d_string,
        stroke=style.color.as_hex(),
        stroke_width=style.stroke_width,
        fill="none",
        stroke_linecap="round",
        stroke_linejoin="round",
        opacity=style.opacity,
        class_="element coil",
        id=element.id,
    )
    return path


def _draw_helix(element: HelixSceneElement, coords_2d: np.ndarray) -> Optional[Path]:
    """Draws a Helix element as a filled path, or a line if too short."""
    style = element.style

    if coords_2d is None or len(coords_2d) < 2:
        logger.debug(f"Skipping Helix {element.id}: Less than 2 coordinates.")
        return None

    # Common arguments
    kwargs = {
        "stroke_width": style.stroke_width,
        "stroke_opacity": style.opacity,
        "stroke_linecap": "round",
        "stroke_linejoin": "round",
        "class_": "element helix",
        "id": element.id,
    }
    d_string = ""

    if len(coords_2d) == 2:
        logger.debug(
            f"Drawing Helix {element.id} as line: Only 2 coordinates available."
        )
        # Line style
        kwargs["fill"] = "none"
        kwargs["stroke"] = style.color.as_hex()
        kwargs["stroke_width"] = style.simplified_width
        kwargs["opacity"] = style.opacity
        d_string = f"M {coords_2d[0, 0]},{coords_2d[0, 1]} L {coords_2d[1, 0]},{coords_2d[1, 1]}"
    else:  # len(coords_2d) >= 3
        # Filled polygon style
        kwargs["fill"] = style.color.as_hex()
        kwargs["opacity"] = style.opacity
        kwargs["stroke"] = style.stroke_color.as_hex()
        d_parts = [f"M {coords_2d[0, 0]},{coords_2d[0, 1]}"]
        for point in coords_2d[1:]:
            d_parts.append(f"L {point[0]},{point[1]}")
        d_parts.append("Z")
        d_string = " ".join(d_parts)

    path = Path(d=d_string, **kwargs)
    return path


def _draw_sheet(element: SheetSceneElement, coords_2d: np.ndarray) -> Optional[Path]:
    """Draws a Sheet element as a filled path (arrow), or a line if too short."""
    style = element.style

    if coords_2d is None or len(coords_2d) < 2:
        logger.debug(f"Skipping Sheet {element.id}: Less than 2 coordinates.")
        return None

    kwargs = {
        "stroke_width": style.stroke_width,
        "stroke_opacity": style.opacity,
        "stroke_linecap": "round",
        "stroke_linejoin": "round",
        "class_": "element sheet",
        "id": element.id,
    }
    d_string = ""

    if len(coords_2d) == 2:
        logger.debug(
            f"Drawing Sheet {element.id} as line: Only 2 coordinates available."
        )
        kwargs["fill"] = "none"
        kwargs["stroke"] = style.color.as_hex()
        kwargs["stroke_width"] = style.simplified_width
        kwargs["opacity"] = style.opacity
        d_string = f"M {coords_2d[0, 0]},{coords_2d[0, 1]} L {coords_2d[1, 0]},{coords_2d[1, 1]}"
    else:  # len(coords_2d) >= 3
        kwargs["fill"] = style.color.as_hex()
        kwargs["opacity"] = style.opacity

        kwargs["stroke"] = style.stroke_color.as_hex()
        d_parts = [f"M {coords_2d[0, 0]},{coords_2d[0, 1]}"]
        for point in coords_2d[1:]:
            d_parts.append(f"L {point[0]},{point[1]}")
        d_parts.append("Z")
        d_string = " ".join(d_parts)

    path = Path(d=d_string, **kwargs)
    return path


def _draw_connection(
    start_point: np.ndarray,
    end_point: np.ndarray,
    id: Optional[str] = None,
    style: Optional[Dict[str, Any]] = None,
) -> Line:
    """Creates a drawsvg.Line object for connecting structure elements.

    Args:
        start_point: The (x, y) coordinates of the line start.
        end_point: The (x, y) coordinates of the line end.
        style: Optional dictionary of drawsvg style attributes.
               Defaults derived from CoilStyle will be used if None.

    Returns:
        A drawsvg.Line object.
    """
    # If no specific style is provided, derive defaults from CoilStyle
    if style is None:
        # Use the class attribute directly
        default_coil_style = CoilStyle()
        style = {
            "stroke_width": default_coil_style.stroke_width,
            "opacity": default_coil_style.opacity,
            "stroke_linecap": "round",
            "stroke_linejoin": "round",
            "stroke": default_coil_style.color.as_hex(),
        }

    # Create a Line object, not a Path
    connection_line = Line(
        sx=start_point[0],
        sy=start_point[1],
        ex=end_point[0],
        ey=end_point[1],
        id=id,
        **style,
        class_="connection",
    )
    return connection_line
