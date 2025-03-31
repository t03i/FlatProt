# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from drawsvg import DrawingElement, Path, Line
import numpy as np

from ..scene.structure import (
    CoilElement,
    HelixElement,
    SheetElement,
    StructureSceneElement,
)


def draw_coil(element: CoilElement) -> DrawingElement:
    """Draw a coil element"""

    # Smooth the coordinates to reduce noise
    coords = element.display_coordinates

    path = Path(
        stroke=element.style.stroke_color,
        stroke_width=element.style.line_width * element.style.stroke_width_factor,
        fill=element.style.fill_color,
        stroke_opacity=element.style.opacity,
        opacity=element.style.opacity,
        class_="element coil",
        linecap="round",
    )

    # Start path at first point
    path.M(coords[0][0], coords[0][1])

    # Draw lines to each subsequent point
    for point in coords[1:]:
        path.L(point[0], point[1])

    return path


def draw_short_element(element: StructureSceneElement) -> DrawingElement:
    element_class = "helix" if isinstance(element, HelixElement) else "sheet"
    return Line(
        *element.coordinates[0],
        *element.coordinates[-1],
        stroke=element.style.fill_color,
        stroke_opacity=element.style.opacity,
        opacity=element.style.opacity,
        stroke_width=element.style.line_width * element.style.stroke_width_factor,
        class_=f"element {element_class} short-element",
        linecap="round",
    )


def draw_helix(element: HelixElement) -> DrawingElement:
    """Draw a helix element"""

    coords = element.display_coordinates

    if len(element.coordinates) <= element.style.min_helix_length:
        return draw_short_element(element)

    path = Path(
        stroke=element.style.stroke_color,
        stroke_width=element.style.line_width * element.style.stroke_width_factor,
        fill=element.style.fill_color,
        stroke_opacity=element.style.opacity,
        opacity=element.style.opacity,
        class_="element helix",
    )

    # Start path at first point
    path.M(*coords[0])

    # Draw lines to each subsequent point
    for point in coords[1:]:
        path.L(*point)

    # Close the path
    path.Z()

    return path


def draw_sheet(element: SheetElement) -> DrawingElement:
    """Draw a sheet element"""

    coords = element.display_coordinates

    if (
        np.isclose(
            element.coordinates[0, :], element.coordinates[-1, :], atol=1e-6
        ).all()
        or len(element.coordinates) <= element.style.min_sheet_length
    ):
        return draw_short_element(element)

    # Create path
    path = Path(
        stroke=element.style.stroke_color,
        stroke_width=element.style.line_width * element.style.stroke_width_factor,
        fill=element.style.fill_color,
        stroke_opacity=element.style.opacity,
        opacity=element.style.opacity,
        class_="element sheet",
    )

    # Draw simple triangular arrow
    path.M(coords[0][0], coords[0][1])  # Start at left point
    path.L(coords[1][0], coords[1][1])  # To arrow tip
    path.L(coords[2][0], coords[2][1])  # To right point
    path.Z()  # Close the path

    return path
