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
    return Line(
        *annotation.display_coordinates()[0],
        *annotation.display_coordinates()[1],
    )


def draw_area_annotation(annotation: AreaAnnotation) -> DrawingElement:
    """Draw an area annotation"""
    return Path(
        *annotation.display_coordinates(),
        stroke="black",
        stroke_width=1,
        fill="none",
    )
