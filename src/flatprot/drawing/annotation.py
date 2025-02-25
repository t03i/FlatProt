# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from drawsvg import DrawingElement, Circle, Line, Path

from flatprot.scene import PointAnnotation, LineAnnotation, AreaAnnotation


def draw_point_annotation(annotation: PointAnnotation) -> DrawingElement:
    """Draw a point annotation"""
    return Circle(
        *annotation.display_coordinates()[0],
        r=annotation.style.connector_radius,
        fill=annotation.style.fill_color,
        fill_opacity=annotation.style.fill_opacity,
        class_="annotation point",
    )


def draw_line_annotation(annotation: LineAnnotation) -> DrawingElement:
    """Draw a line annotation"""
    return [
        Line(
            *annotation.display_coordinates()[0],
            *annotation.display_coordinates()[1],
            stroke=annotation.style.stroke_color,
            stroke_width=annotation.style.stroke_width,
            stroke_opacity=annotation.style.stroke_opacity,
            class_="annotation line",
        ),
        Circle(
            *annotation.display_coordinates()[0],
            r=annotation.style.connector_radius,
            fill=annotation.style.connector_color,
            fill_opacity=annotation.style.connector_opacity,
            class_="annotation connector",
        ),
        Circle(
            *annotation.display_coordinates()[1],
            r=annotation.style.connector_radius,
            fill=annotation.style.connector_color,
            fill_opacity=annotation.style.connector_opacity,
            class_="annotation connector",
        ),
    ]


def draw_area_annotation(annotation: AreaAnnotation) -> DrawingElement:
    """Draw an area annotation"""
    path = Path(
        stroke=annotation.style.stroke_color,
        stroke_width=annotation.style.stroke_width,
        stroke_opacity=annotation.style.stroke_opacity,
        fill=annotation.style.fill_color,
        fill_opacity=annotation.style.fill_opacity,
        class_="annotation area",
    )
    coords = annotation.display_coordinates()
    path.M(*coords[0])

    # Draw lines to each subsequent point
    for point in coords[1:]:
        path.L(*point)

    # Close the path
    path.Z()

    return path
