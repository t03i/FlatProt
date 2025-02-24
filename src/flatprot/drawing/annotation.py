# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from drawsvg import DrawingElement, Circle, Line, Path

from flatprot.scene import PointAnnotation, LineAnnotation, AreaAnnotation


def draw_point_annotation(annotation: PointAnnotation) -> DrawingElement:
    """Draw a point annotation"""
    return Circle(
        *annotation.display_coordinates()[0],
        r=2,
        fill="white",
    )


def draw_line_annotation(annotation: LineAnnotation) -> DrawingElement:
    """Draw a line annotation"""
    return [
        Circle(
            *annotation.display_coordinates()[0],
            r=2,
            fill="white",
        ),
        Line(
            *annotation.display_coordinates()[0],
            *annotation.display_coordinates()[1],
            stroke="purple",
            stroke_width=3,
        ),
        Circle(
            *annotation.display_coordinates()[1],
            r=2,
            fill="white",
        ),
    ]


def draw_area_annotation(annotation: AreaAnnotation) -> DrawingElement:
    """Draw an area annotation"""
    path = Path(stroke="purple", stroke_width=3, fill="black", fill_opacity=0.2)
    coords = annotation.display_coordinates()
    path.M(*coords[0])

    # Draw lines to each subsequent point
    for point in coords[1:]:
        path.L(*point)

    # Close the path
    path.Z()

    return path
