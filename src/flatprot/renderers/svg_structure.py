# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import numpy as np
from drawsvg import Path, Line, Circle

from flatprot.scene import (
    HelixSceneElement,
    SheetSceneElement,
    CoilSceneElement,
)

from flatprot.scene.connection import ConnectionStyle
from flatprot.core import logger


# Define SVG stroke-dasharray values for different line styles
LINE_STYLE_MAP = {
    "solid": None,
    "dashed": "5,5",
    "dotted": "1,3",
}


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


def _draw_connection_line(
    start_point: np.ndarray,
    end_point: np.ndarray,
    style: ConnectionStyle,
    id: Optional[str] = None,
) -> Line:
    """Creates a drawsvg.Line object based on a ConnectionStyle.

    Args:
        start_point: The (x, y) coordinates of the line start.
        end_point: The (x, y) coordinates of the line end.
        style: The ConnectionStyle object defining visual attributes.
        id: Optional SVG ID for the line.

    Returns:
        A drawsvg.Line object.
    """
    # Check if points are too close to avoid zero-length lines
    if np.allclose(start_point, end_point, atol=1e-3):
        logger.debug(
            f"Skipping connection line {id}: start and end points are identical."
        )
        # Return an empty line or handle as needed, maybe return None? For now, creating zero length.
        # Returning None might be better if the caller handles it.
        return Line(
            sx=start_point[0],
            sy=start_point[1],
            ex=start_point[0],
            ey=start_point[1],
            stroke="none",
            stroke_width=0,
            opacity=0,
            stroke_linecap="round",
            stroke_linejoin="round",
            id=id,
            class_="connection",
        )

    # Get the dasharray based on the line_style
    dasharray = LINE_STYLE_MAP.get(style.line_style)

    connection_line = Line(
        sx=start_point[0],
        sy=start_point[1],
        ex=end_point[0],
        ey=end_point[1],
        stroke=style.color.as_hex(),
        stroke_width=style.stroke_width,
        stroke_opacity=style.opacity,
        stroke_linecap="round",
        class_="connection",
    )

    # Apply dasharray only if it's not None (i.e., not 'solid')
    if dasharray:
        connection_line.args["stroke-dasharray"] = dasharray

    return connection_line
