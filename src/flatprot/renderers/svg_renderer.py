# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Renders a FlatProt Scene object to an SVG image using the drawsvg library."""

from typing import List, Optional, Tuple, Dict
from drawsvg import Drawing, Group, Rectangle

from flatprot.scene import (
    Scene,
    BaseSceneElement,
    SceneGroup,
    CoilSceneElement,
    HelixSceneElement,
    SheetSceneElement,
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
    BaseAnnotationElement,
    BaseStructureSceneElement,
)
from flatprot.core.logger import logger

from .svg_structure import _draw_coil, _draw_helix, _draw_sheet
from .svg_annotations import (
    _draw_point_annotation,
    _draw_line_annotation,
    _draw_area_annotation,
)

# --- Helper Functions --- #


# Removed _get_rendered_coords_for_annotation helper function


# --- Renderer Class --- #


class SVGRenderer:
    """Renders a Scene object to an SVG Drawing.

    Attributes:
        scene: The Scene object to render.
        width: The width of the SVG canvas.
        height: The height of the SVG canvas.
        background_color: Optional background color for the SVG.
        background_opacity: Opacity for the background color.
    """

    DEFAULT_WIDTH = 600
    DEFAULT_HEIGHT = 400
    DEFAULT_BG_COLOR = "#FFFFFF"
    DEFAULT_BG_OPACITY = 1.0

    # Map element types to their drawing functions
    DRAW_MAP = {
        CoilSceneElement: _draw_coil,
        HelixSceneElement: _draw_helix,
        SheetSceneElement: _draw_sheet,
        # Annotations handled separately due to anchor calculation
    }
    ANNOTATION_DRAW_MAP = {
        PointAnnotation: _draw_point_annotation,
        LineAnnotation: _draw_line_annotation,
        AreaAnnotation: _draw_area_annotation,
    }

    def __init__(
        self,
        scene: Scene,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        background_color: Optional[str] = DEFAULT_BG_COLOR,
        background_opacity: float = DEFAULT_BG_OPACITY,
    ):
        """Initializes the SVGRenderer.

        Args:
            scene: The Scene object containing the elements to render.
            width: Desired width of the SVG canvas.
            height: Desired height of the SVG canvas.
            background_color: Background color (CSS string, e.g., '#FFFFFF' or 'white'). None for transparent.
            background_opacity: Background opacity (0.0 to 1.0).
        """
        if not isinstance(scene, Scene):
            raise TypeError("Renderer requires a valid Scene object.")

        self.scene = scene
        self.width = width
        self.height = height
        self.background_color = background_color
        self.background_opacity = background_opacity
        self._element_map: Dict[str, BaseSceneElement] = {
            e.id: e for e in self.scene.get_all_elements()
        }

    def _collect_and_sort_renderables(
        self,
    ) -> Tuple[
        List[Tuple[float, BaseStructureSceneElement]],
        List[Tuple[float, BaseAnnotationElement]],
    ]:
        """Traverses scene, collects renderable leaf nodes, sorts them by Z-depth."""
        structure_elements: List[Tuple[float, BaseStructureSceneElement]] = []
        annotation_elements: List[Tuple[float, BaseAnnotationElement]] = []
        structure = self.scene.structure  # Get structure once

        for element, _hierarchy_depth in self.scene.traverse():  # Use scene traverse
            # Skip invisible or group elements
            if not element.style.visibility or isinstance(element, SceneGroup):
                continue

            render_depth = element.get_depth(structure)

            if render_depth is None:
                logger.debug(
                    f"Element {element.id} ({type(element).__name__}) has no rendering depth, skipping."
                )
                continue

            # Categorize and collect
            if isinstance(element, BaseStructureSceneElement):
                structure_elements.append((render_depth, element))
            elif isinstance(element, BaseAnnotationElement):
                # Depth is inf, but store it for consistency (sorting won't change)
                annotation_elements.append((render_depth, element))
            # else: Ignore other potential non-group, non-renderable types

        # Sort structure elements by depth (ascending)
        structure_elements.sort(key=lambda item: item[0])
        # Annotations are naturally last due to depth=inf, but explicit sort doesn't hurt
        annotation_elements.sort(key=lambda item: item[0])

        return structure_elements, annotation_elements

    def _build_svg_hierarchy(
        self,
        element: BaseSceneElement,
        parent_svg_group: Group,
        svg_group_map: Dict[str, Group],
    ) -> None:
        """Recursively builds SVG groups mirroring SceneGroups."""
        if not isinstance(element, SceneGroup):
            return  # Only process groups

        # Create corresponding SVG group
        # TODO: Map SceneGroup transforms to SVG transforms (e.g., 'translate(x,y)')
        # This requires defining how transforms are stored in SceneGroup. Assuming a dict for now.
        svg_transform_str = " ".join(
            [f"{k}({v})" for k, v in element.transforms.items()]
        )
        current_svg_group = Group(id=element.id, transform=svg_transform_str)
        svg_group_map[element.id] = current_svg_group
        parent_svg_group.append(current_svg_group)

        # Recursively process children
        for child in element.children:
            self._build_svg_hierarchy(child, current_svg_group, svg_group_map)

    def render(self) -> Drawing:
        """Renders the scene to a drawsvg.Drawing object."""
        drawing = Drawing(self.width, self.height, origin="center")
        svg_group_map: Dict[str, Group] = {}

        # 1. Add Background
        if self.background_color:
            drawing.append(
                Rectangle(
                    -self.width / 2,
                    -self.height / 2,
                    self.width,
                    self.height,
                    fill=self.background_color,
                    opacity=self.background_opacity,
                    class_="background",
                )
            )

        # 2. Build SVG Group Hierarchy
        root_group = Group(id="flatprot-root")
        drawing.append(root_group)
        svg_group_map["flatprot-root"] = root_group  # Register root

        for top_level_node in self.scene.top_level_nodes:
            self._build_svg_hierarchy(top_level_node, root_group, svg_group_map)

        # 3. Collect and Sort Renderable Elements
        sorted_structures, sorted_annotations = self._collect_and_sort_renderables()

        # 4. Draw Structure Elements into correct SVG groups
        structure = self.scene.structure
        for depth, element in sorted_structures:
            element_type = type(element)
            draw_func = self.DRAW_MAP.get(element_type)
            if not draw_func:
                logger.warning(
                    f"No drawing func for structure type: {element_type.__name__}"
                )
                continue

            svg_shape = draw_func(element, structure)
            if svg_shape:
                # Find the parent SVG group
                parent_group_id = (
                    element.parent.id if element.parent else "flatprot-root"
                )
                target_svg_group = svg_group_map.get(parent_group_id)
                if target_svg_group:
                    if isinstance(svg_shape, list):
                        for item in svg_shape:
                            target_svg_group.append(item)
                    else:
                        target_svg_group.append(svg_shape)
                else:
                    logger.error(
                        f"Could not find target SVG group '{parent_group_id}' for element {element.id}"
                    )

        # 5. Draw Annotation Elements (typically added to root for visibility)
        for depth, element in sorted_annotations:
            element_type = type(element)
            draw_func = self.ANNOTATION_DRAW_MAP.get(element_type)
            if not draw_func:
                logger.warning(
                    f"No drawing func for annotation type: {element_type.__name__}"
                )
                continue

            # Get coordinates using the new Scene method
            rendered_coords = self.scene.get_rendered_coordinates_for_annotation(
                element
            )
            if rendered_coords is None:
                # Scene method already logs warnings/errors
                logger.debug(
                    f"Skipping annotation {element.id} due to missing rendered coordinates."
                )
                continue

            # Call the specific annotation drawing function
            svg_shapes = draw_func(element, rendered_coords)
            if svg_shapes:
                # Add annotations directly to the root group to ensure they are on top
                if isinstance(svg_shapes, list):
                    for item in svg_shapes:
                        root_group.append(item)
                else:
                    root_group.append(svg_shapes)

        return drawing

    def save_svg(self, filename: str) -> None:
        """Renders the scene and saves it to an SVG file.

        Args:
            filename: The path to the output SVG file.
        """
        drawing = self.render()
        drawing.save_svg(filename)
        logger.info(f"SVG saved to {filename}")

    def get_svg_string(self) -> str:
        """Renders the scene and returns the SVG content as a string.

        Returns:
            The SVG content as a string.
        """
        drawing = self.render()
        return drawing.as_svg()
