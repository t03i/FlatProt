# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import numpy as np
from drawsvg import Circle, Text, Line, Path, Group
from flatprot.scene import (
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
    BaseAnnotationElement,
)
from flatprot.core.logger import logger


def _create_annotation_label(
    annotation: BaseAnnotationElement,
    x: float,
    y: float,
    text_anchor: str = "start",  # Default anchor
) -> Text:
    """Creates a styled drawsvg.Text element for an annotation label."""
    style = annotation.style
    # Ensure annotation.label exists before using it
    label_text = annotation.label if annotation.label else ""
    return Text(
        text=label_text,  # Use label directly from annotation
        font_size=style.label_font_size,
        font_family=style.label_font_family,
        font_weight=style.label_font_weight,
        fill=style.label_color.as_hex(),
        x=x,
        y=y,
        text_anchor=text_anchor,
        dominant_baseline="middle",
        class_="annotation label",
        id=f"{annotation.id}-label",
    )


def _draw_point_annotation(
    annotation: PointAnnotation, anchor_coords: np.ndarray
) -> Optional[Group]:
    """Draws a PointAnnotation (marker + label) wrapped in a group."""
    if anchor_coords is None or len(anchor_coords) == 0:
        logger.warning(
            f"Skipping PointAnnotation {annotation.id}: No anchor coordinates."
        )
        return None

    style = annotation.style
    # Use the first anchor point
    center_x, center_y = (
        anchor_coords[0, 0] + style.offset[0],
        anchor_coords[0, 1] + style.offset[1],
    )

    group = Group(id=annotation.id, class_="annotation point-annotation")
    marker = Circle(
        cx=center_x,
        cy=center_y,
        r=style.marker_radius,
        fill=style.color.as_hex(),
        opacity=style.opacity,
        class_="annotation point-marker",
        id=f"{annotation.id}-marker",
    )
    group.append(marker)

    # Use the helper function if label exists
    if annotation.label:
        label_x = center_x + style.marker_radius + style.label_offset[0]
        label_y = center_y + style.label_offset[1]
        label_element = _create_annotation_label(
            annotation, label_x, label_y, text_anchor="start"
        )
        group.append(label_element)

    return group


def _draw_line_annotation(
    annotation: LineAnnotation, anchor_coords: np.ndarray
) -> Optional[Group]:
    """Draws a LineAnnotation (line + connectors + label) wrapped in a group."""
    if anchor_coords is None or len(anchor_coords) < 2:
        logger.warning(
            f"Skipping LineAnnotation {annotation.id}: Requires at least 2 anchor coordinates."
        )
        return None

    style = annotation.style
    start_x, start_y = (
        anchor_coords[0, 0] + style.offset[0],
        anchor_coords[0, 1] + style.offset[1],
    )
    end_x, end_y = (
        anchor_coords[1, 0] + style.offset[0],
        anchor_coords[1, 1] + style.offset[1],
    )
    connector_radius = style.connector_radius

    group = Group(id=annotation.id, class_="annotation line-annotation")
    line = Line(
        sx=start_x,
        sy=start_y,
        ex=end_x,
        ey=end_y,
        stroke_linecap="round",
        stroke_linejoin="round",
        stroke_dasharray=",".join(str(x) for x in style.line_style)
        if style.line_style and len(style.line_style) > 0
        else None,
        stroke=style.line_color.as_hex(),
        stroke_width=style.stroke_width,
        opacity=style.opacity,
        class_="annotation line",
        id=f"{annotation.id}-line",
    )
    group.append(line)

    # Connectors (optional, could be styled)
    marker_start = Circle(  # Renamed variable to avoid conflict
        cx=start_x,
        cy=start_y,
        r=connector_radius,
        fill=style.color.as_hex(),
        class_="annotation connector",
        id=f"{annotation.id}-connector-start",
    )
    group.append(marker_start)
    marker_end = Circle(  # Renamed variable to avoid conflict
        cx=end_x,
        cy=end_y,
        r=connector_radius,
        fill=style.color.as_hex(),
        class_="annotation connector",
        id=f"{annotation.id}-connector-end",
    )
    group.append(marker_end)

    # Use the helper function if label exists
    if annotation.label:
        # Position label near midpoint, offset perpendicularly
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        dx, dy = end_x - start_x, end_y - start_y
        length = np.sqrt(dx * dx + dy * dy)
        offset_dist = 10  # Example offset distance
        label_x, label_y = mid_x, mid_y
        if length > 1e-6:
            # Normalized perpendicular vector: (-dy/length, dx/length)
            label_x += (-dy / length) * offset_dist + style.label_offset[0]
            label_y += (dx / length) * offset_dist + style.label_offset[1]
        else:  # Points coincide
            label_x += offset_dist + style.label_offset[0]
            label_y += style.label_offset[1]

        label_element = _create_annotation_label(
            annotation, label_x, label_y, text_anchor="middle"
        )
        group.append(label_element)

    return group


def _draw_area_annotation(
    annotation: AreaAnnotation, rendered_coords: np.ndarray
) -> Optional[Group]:
    """Draws an AreaAnnotation (outline + label) wrapped in a group.

    Receives the full set of rendered coordinates for the area.
    Draws an outline path connecting these points and places the label
    offset from the rightmost point, vertically centered.
    """
    if rendered_coords is None or len(rendered_coords) < 1:
        logger.warning(
            f"Skipping AreaAnnotation {annotation.id}: No rendered coordinates provided."
        )
        return None

    style = annotation.style
    group = Group(id=annotation.id, class_="annotation area-annotation")

    # 1. Draw the outline path (only if enough points)
    if len(rendered_coords) >= 3:
        outline = Path(
            stroke=style.color.as_hex(),
            stroke_width=1,  # Example, make configurable in style?
            stroke_opacity=style.opacity,  # Slightly transparent stroke
            fill=style.fill_color.as_hex(),
            fill_opacity=style.fill_opacity,  # More transparent fill
            class_="annotation area-outline",
            id=f"{annotation.id}-outline",
        )
        # Apply main offset to all points for the outline
        offset_coords = rendered_coords[:, :2] + style.offset
        outline.M(offset_coords[0, 0], offset_coords[0, 1])
        for p in offset_coords[1:]:
            outline.L(p[0], p[1])
        outline.Z()  # Close the path
        group.append(outline)
    elif len(rendered_coords) == 2:  # Draw line if only 2 points
        offset_coords = rendered_coords[:, :2] + style.offset
        line = Line(
            x1=offset_coords[0, 0],
            y1=offset_coords[0, 1],
            x2=offset_coords[1, 0],
            y2=offset_coords[1, 1],
            stroke=style.color.as_hex(),
            stroke_width=1,
            opacity=style.opacity * 0.8,
            class_="annotation area-line",
            id=f"{annotation.id}-line",
        )
        group.append(line)
    # If only 1 point, maybe draw a marker? For now, just use it for label anchor.

    # 2. Position and draw the label
    # Use the helper function if label exists
    if annotation.label:
        # Calculate centroid of the *offset* XY coordinates for label Y position
        offset_coords_xy = rendered_coords[:, :2] + style.offset
        centroid_y = np.mean(offset_coords_xy[:, 1])

        # Find the rightmost point of the *offset* shape for label X position
        max_x = np.max(offset_coords_xy[:, 0])

        # Position label to the right, vertically centered via centroid_y
        # Apply label-specific offset relative to this calculated position
        label_offset_distance = 10  # Example distance, could be styled
        label_x = max_x + label_offset_distance + style.label_offset[0]
        label_y = centroid_y + style.label_offset[1]

        label_element = _create_annotation_label(
            annotation, label_x, label_y, text_anchor="start"
        )
        group.append(label_element)

    return group if group.children else None
