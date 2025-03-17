# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from drawsvg import DrawingElement, Circle, Line, Path, Text
import numpy as np
from typing import Any, Tuple

from flatprot.scene import PointAnnotation, LineAnnotation, AreaAnnotation


def create_label_element(x: float, y: float, label: str, style: Any, id: str) -> Text:
    """Create a text element for annotation labels.

    Args:
        x: X coordinate for label placement
        y: Y coordinate for label placement
        label: The text to display
        style: Style object containing text styling properties

    Returns:
        Text element with proper styling
    """
    return Text(
        text=label,
        text_anchor="start",
        x=x,
        y=y,
        font_size=style.label_font_size,
        font_family=style.label_font_family,
        fill=style.label_color,
        dominant_baseline="middle",
        class_="annotation label",
        id=id,
    )


def calc_label_pos_area(coords: np.ndarray, offset: float = 10) -> Tuple[float, float]:
    """Calculate optimal label position outside the shape.

    Args:
        coords: Array of coordinates defining the shape
        offset: Distance to offset label from the shape boundary

    Returns:
        Tuple of (x, y) coordinates for label placement
    """
    # Calculate centroid
    centroid_y = coords[:, 1].mean()

    # Find the rightmost point of the shape
    max_x = coords[:, 0].max()

    # Position label to the right of the shape
    label_x = max_x + offset
    label_y = centroid_y

    return label_x, label_y


def calc_label_pos_line(coords: np.ndarray, offset: float = 10) -> Tuple[float, float]:
    """Calculate optimal label position for line annotations.

    Args:
        coords: Array of two points defining the line endpoints
        offset: Distance to offset label from the line

    Returns:
        Tuple of (x, y) coordinates for label placement
    """
    # Calculate midpoint
    mid_x = (coords[0][0] + coords[1][0]) / 2
    mid_y = (coords[0][1] + coords[1][1]) / 2

    # Calculate vector perpendicular to the line
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    length = np.sqrt(dx * dx + dy * dy)

    if length > 0:
        # Normalize and rotate 90 degrees for perpendicular vector
        perpendicular = np.array([dy / length, -dx / length])
        # Position label along perpendicular vector
        label_x = mid_x + perpendicular[0] * offset
        label_y = mid_y + perpendicular[1] * offset
    else:
        # Fallback if points are identical
        label_x = mid_x + offset
        label_y = mid_y

    return label_x, label_y


def draw_point_annotation(annotation: PointAnnotation) -> DrawingElement:
    """Draw a point annotation with label"""
    coords = annotation.display_coordinates()[0]
    # Position label to the right of the point
    label_x = coords[0] + annotation.style.stroke_width + annotation.style.label_offset
    return [
        Circle(
            *coords,
            r=annotation.style.stroke_width,
            fill=annotation.style.fill_color,
            fill_opacity=annotation.style.fill_opacity,
            class_="annotation point",
            id=f"annotation-point-{annotation.id}",
        ),
        create_label_element(
            label_x,
            coords[1],  # Keep same y-coordinate as point
            annotation.label,
            annotation.style,
            id=f"annotation-label-{annotation.id}",
        ),
    ]


def draw_line_annotation(annotation: LineAnnotation) -> DrawingElement:
    """Draw a line annotation with label"""
    coords = annotation.display_coordinates()

    # Calculate label position
    label_x, label_y = calc_label_pos_line(coords, offset=annotation.style.label_offset)

    return [
        Line(
            *coords[0],
            *coords[1],
            stroke=annotation.style.stroke_color,
            stroke_width=annotation.style.stroke_width,
            stroke_opacity=annotation.style.stroke_opacity,
            class_="annotation line",
            id=f"annotation-line-{annotation.id}",
        ),
        Circle(
            *coords[0],
            r=annotation.style.connector_radius,
            fill=annotation.style.connector_color,
            fill_opacity=annotation.style.connector_opacity,
            class_="annotation connector",
            id=f"annotation-connector-start-{annotation.id}",
        ),
        Circle(
            *coords[1],
            r=annotation.style.connector_radius,
            fill=annotation.style.connector_color,
            fill_opacity=annotation.style.connector_opacity,
            class_="annotation connector",
            id=f"annotation-connector-end-{annotation.id}",
        ),
        create_label_element(
            label_x,
            label_y,
            annotation.label,
            annotation.style,
            id=f"annotation-label-{annotation.id}",
        ),
    ]


def draw_area_annotation(annotation: AreaAnnotation) -> DrawingElement:
    """Draw an area annotation with label"""
    coords = annotation.display_coordinates()

    # Calculate label position outside the area
    label_x, label_y = calc_label_pos_area(coords, offset=annotation.style.label_offset)

    # Create the area outline
    outline = Path(
        stroke=annotation.style.stroke_color,
        stroke_width=annotation.style.stroke_width,
        stroke_opacity=annotation.style.stroke_opacity,
        fill=annotation.style.fill_color,
        fill_opacity=annotation.style.fill_opacity,
        class_="annotation area",
        id=f"annotation-area-{annotation.id}",
    )
    outline.M(*coords[0])
    for p in coords[1:]:
        outline.L(p[0], p[1])
    outline.Z()

    return [
        outline,
        create_label_element(
            label_x,
            label_y,
            annotation.label,
            annotation.style,
            id=f"annotation-label-{annotation.id}",
        ),
    ]
