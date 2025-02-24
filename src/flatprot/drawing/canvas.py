# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Iterable

from drawsvg import Drawing, Rectangle, Group

from flatprot.style import StyleManager, StyleType
from flatprot.scene import (
    Scene,
    SceneGroup,
    SceneElement,
    HelixElement,
    SheetElement,
    CoilElement,
)
from flatprot.scene.annotations import (
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
)
from .structure import draw_helix, draw_sheet, draw_coil
from .annotation import (
    draw_point_annotation,
    draw_line_annotation,
    draw_area_annotation,
)

DRAW_MAP = {
    HelixElement: draw_helix,
    SheetElement: draw_sheet,
    CoilElement: draw_coil,
    PointAnnotation: draw_point_annotation,
    LineAnnotation: draw_line_annotation,
    AreaAnnotation: draw_area_annotation,
}


def draw_element(element: SceneElement) -> Drawing:
    """Draw an element to an SVG."""
    return DRAW_MAP[type(element)](element)


def _draw_scene_common(
    scene: Scene | SceneGroup, group_id: str, transforms: dict | None = None
) -> Drawing:
    """Common drawing logic for both scenes and scene groups."""
    group = Group(id=group_id, **(transforms or {}))
    for element in scene:
        if isinstance(element, SceneGroup):
            drawing = _draw_scene_common(
                element, element.id, transforms=element.transforms
            )
        elif isinstance(element, SceneElement):
            drawing = draw_element(element)

        if drawing is not None:
            if isinstance(drawing, Iterable):
                group.extend(drawing)
            else:
                group.append(drawing)

    return group


def draw_scene_group(scene: SceneGroup, transforms: dict | None = None) -> Drawing:
    """Draw a scene group to an SVG."""
    return _draw_scene_common(scene, scene.id, transforms)


def draw_scene(scene: Scene, transforms: dict | None = None) -> Drawing:
    """Draw a scene to an SVG."""
    return _draw_scene_common(scene, "root", transforms)


class Canvas:
    """Manages the overall visualization canvas and rendering."""

    def __init__(
        self,
        scene: Scene,
        style_manager: Optional[StyleManager] = None,
    ):
        self.scene = scene
        st = style_manager or StyleManager.create_default()
        self.canvas_settings = st.get_style(StyleType.CANVAS)

    def render(self, output_path: Optional[str] = None) -> Drawing:
        """Render the scene to SVG."""
        drawing = Drawing(
            self.canvas_settings.width,
            self.canvas_settings.height,
            origin="center",
        )

        # Add background if specified
        if self.canvas_settings.background_color:
            drawing.append(
                Rectangle(
                    -self.canvas_settings.width / 2,
                    -self.canvas_settings.height / 2,
                    self.canvas_settings.width,
                    self.canvas_settings.height,
                    fill=self.canvas_settings.background_color,
                    opacity=self.canvas_settings.background_opacity,
                    class_="background",
                )
            )

        # Render and add the root group
        root_scene = draw_scene(self.scene)

        drawing.append(root_scene)

        if output_path:
            drawing.save_svg(output_path)

        return drawing
