# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from drawsvg import DrawingElement, Circle

from flatprot.scene import PointAnnotation


def draw_point_annotation(annotation: PointAnnotation) -> DrawingElement:
    """Draw a point annotation"""
    return Circle(
        *annotation.display_coordinates()[0],
        r=20,
        fill="black",
    )
