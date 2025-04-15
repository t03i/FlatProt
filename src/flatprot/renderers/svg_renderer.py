# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Renders a FlatProt Scene object to an SVG image using the drawsvg library."""

from typing import List, Optional, Tuple, Dict, Any
from drawsvg import Drawing, Group, Rectangle
import numpy as np

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

from .svg_structure import (
    _draw_coil,
    _draw_helix,
    _draw_sheet,
    _calculate_coil_connection_points,
    _calculate_helix_connection_points,
    _calculate_sheet_connection_points,
    _draw_connection,
)
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

        logger.debug("--- Starting _collect_and_sort_renderables ---")  # DEBUG
        for element, hierarchy_depth in self.scene.traverse():  # Use scene traverse
            logger.debug(
                f"Traversing: {element.id} (Type: {type(element).__name__}, Depth: {hierarchy_depth})"
            )

            # Skip invisible or group elements
            if not element.style.visibility:
                logger.debug(f"Skipping {element.id}: Invisible")  # DEBUG
                continue
            if isinstance(element, SceneGroup):
                logger.debug(f"Skipping {element.id}: Is SceneGroup")  # DEBUG
                continue

            render_depth = element.get_depth(structure)
            logger.debug(
                f"Calculated render_depth for {element.id}: {render_depth}"
            )  # DEBUG

            if render_depth is None:
                logger.debug(
                    f"Element {element.id} ({type(element).__name__}) has no rendering depth, skipping collection."
                )  # DEBUG: Changed message slightly
                continue

            # Categorize and collect
            if isinstance(element, BaseStructureSceneElement):
                logger.debug(f"Collecting Structure Element: {element.id}")  # DEBUG
                structure_elements.append((render_depth, element))
            elif isinstance(element, BaseAnnotationElement):
                # Depth is inf, but store it for consistency (sorting won't change)
                logger.debug(f"Collecting Annotation Element: {element.id}")  # DEBUG
                annotation_elements.append((render_depth, element))
            # else: Ignore other potential non-group, non-renderable types
        logger.debug("--- Finished _collect_and_sort_renderables ---")  # DEBUG

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

    def _prepare_render_data(
        self,
    ) -> Tuple[
        List[BaseStructureSceneElement],  # Ordered structure elements
        Dict[str, np.ndarray],  # Element ID -> coords_2d
        Dict[
            str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        ],  # Element ID -> (start_conn, end_conn)
    ]:
        """Pre-calculates 2D coordinates and connection points for structure elements."""
        # ordered_elements: List[BaseStructureSceneElement] = [] # Keep type hint
        element_coords_cache: Dict[str, np.ndarray] = {}
        connection_points_cache: Dict[
            str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        ] = {}
        structure = self.scene.structure

        # --- Get Sequentially Ordered Structure Elements --- #
        # Uses the new method in Scene to get elements sorted by chain and residue index
        try:
            ordered_elements = self.scene.get_sequential_structure_elements()
            # Filter for visibility *after* getting the ordered list
            ordered_elements = [el for el in ordered_elements if el.style.visibility]
        except Exception as e:
            logger.error(
                f"Error getting sequential structure elements: {e}", exc_info=True
            )
            return [], {}, {}  # Return empty if fetching/sorting failed

        # --- Calculate Coordinates and Connection Points --- #
        for element in ordered_elements:
            element_id = element.id
            try:
                # CRITICAL ASSUMPTION: get_coordinates returns *final projected* 2D/3D coords.
                # If it returns raw 3D, projection needs to happen here.
                coords = element.get_coordinates(structure)

                if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
                    logger.warning(
                        f"Element {element_id} provided invalid coordinates shape: {coords.shape if coords is not None else 'None'}. Skipping."
                    )
                    # Add placeholders to avoid key errors later if neighbors expect connections
                    element_coords_cache[element_id] = np.empty((0, 2))
                    connection_points_cache[element_id] = (None, None)
                    continue

                coords_2d = coords[:, :2]  # Ensure we only use X, Y
                element_coords_cache[element_id] = coords_2d

                # Calculate connection points using imported functions
                start_conn, end_conn = None, None
                if isinstance(element, CoilSceneElement):
                    start_conn, end_conn = _calculate_coil_connection_points(coords_2d)
                elif isinstance(element, HelixSceneElement):
                    start_conn, end_conn = _calculate_helix_connection_points(coords_2d)
                elif isinstance(element, SheetSceneElement):
                    start_conn, end_conn = _calculate_sheet_connection_points(coords_2d)
                else:
                    # Fallback for unknown structure types? Or assume only these 3 exist.
                    logger.warning(
                        f"Unknown structure element type {type(element).__name__} encountered during connection point calculation."
                    )
                    start_conn, end_conn = _calculate_coil_connection_points(
                        coords_2d
                    )  # Default to coil logic

                connection_points_cache[element_id] = (start_conn, end_conn)

            except Exception as e:
                logger.error(
                    f"Error preparing render data for element {element_id}: {e}",
                    exc_info=True,
                )
                # Add placeholders if preparation fails for an element
                element_coords_cache[element_id] = np.empty((0, 2))
                connection_points_cache[element_id] = (None, None)
                continue

        return ordered_elements, element_coords_cache, connection_points_cache

    def render(self) -> Drawing:
        """Renders the scene to a drawsvg.Drawing object."""
        drawing = Drawing(self.width, self.height, origin="center")
        svg_group_map: Dict[str, Group] = {}
        connection_line_group = Group(id="flatprot-connections", class_="connections")

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
        root_group.append(connection_line_group)

        for top_level_node in self.scene.top_level_nodes:
            self._build_svg_hierarchy(top_level_node, root_group, svg_group_map)

        # 3. Prepare Render Data for Structure Elements
        try:
            (
                ordered_structure_elements,
                element_coords_cache,
                connection_points_cache,
            ) = self._prepare_render_data()
        except Exception as e:
            logger.error(f"Failed to prepare render data: {e}", exc_info=True)
            ordered_structure_elements = []
            connection_points_cache = {}

        # 4. Draw Structure Elements into correct SVG groups using prepared data
        num_elements = len(ordered_structure_elements)
        for i, element in enumerate(ordered_structure_elements):
            element_id = element.id
            element_type = type(element)

            # Get cached coordinates for the current element
            coords_2d = element_coords_cache.get(element_id)
            if coords_2d is None or coords_2d.size == 0:
                logger.warning(
                    f"Skipping draw for {element_id}: No valid cached 2D coordinates found."
                )
                continue

            # Select and call the appropriate drawing function
            svg_shape: Optional[Any] = None
            try:
                if isinstance(element, CoilSceneElement):
                    svg_shape = _draw_coil(element, coords_2d)
                elif isinstance(element, HelixSceneElement):
                    svg_shape = _draw_helix(element, coords_2d)
                elif isinstance(element, SheetSceneElement):
                    svg_shape = _draw_sheet(element, coords_2d)
                else:
                    logger.warning(
                        f"No specific draw function mapped for structure type: {element_type.__name__}. Skipping {element_id}."
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Error calling draw function for {element_id}: {e}", exc_info=True
                )
                continue

            # Append the generated shape to the correct SVG group
            if svg_shape:
                parent_group_id = (
                    element.parent.id if element.parent else "flatprot-root"
                )
                target_svg_group = svg_group_map.get(parent_group_id)
                if target_svg_group:
                    target_svg_group.append(svg_shape)
                else:
                    logger.error(
                        f"Could not find target SVG group '{parent_group_id}' for element {element_id}"
                    )

        # 5. Draw Connection Lines Between Adjacent Structure Elements
        for i in range(num_elements - 1):
            element_i = ordered_structure_elements[i]
            element_i_plus_1 = ordered_structure_elements[i + 1]

            if not element_i.is_adjacent_to(element_i_plus_1):
                continue

            # Get connection points from cache
            conn_i_data = connection_points_cache.get(element_i.id)
            conn_i_plus_1_data = connection_points_cache.get(element_i_plus_1.id)

            if conn_i_data and conn_i_plus_1_data:
                end_conn_i = conn_i_data[1]  # End point of element i
                start_conn_i_plus_1 = conn_i_plus_1_data[
                    0
                ]  # Start point of element i+1

                if end_conn_i is not None and start_conn_i_plus_1 is not None:
                    # Check if points are too close to avoid zero-length lines
                    if not np.allclose(end_conn_i, start_conn_i_plus_1, atol=1e-3):
                        connection_line = _draw_connection(
                            start_point=end_conn_i,
                            end_point=start_conn_i_plus_1,
                            style=None,
                            id=f"connection-{element_i.id}-{element_i_plus_1.id}",
                        )
                        connection_line_group.append(connection_line)

        # 6. Draw Annotation Elements (Retrieve separately, logic remains similar)
        _, sorted_annotations = self._collect_and_sort_renderables()

        for (
            depth,
            element,
        ) in sorted_annotations:
            element_type = type(element)
            draw_func = self.ANNOTATION_DRAW_MAP.get(element_type)
            if not draw_func:
                logger.warning(
                    f"No drawing func for annotation type: {element_type.__name__}"
                )
                continue

            # Removed the broad try...except block here
            # Errors during annotation processing should propagate
            rendered_coords = self.scene.get_rendered_coordinates_for_annotation(
                element
            )
            if rendered_coords is None or rendered_coords.size == 0:
                logger.debug(
                    f"Skipping annotation {element.id} due to missing/empty rendered coordinates."
                )
                continue

            svg_shapes = draw_func(element, rendered_coords)
            if svg_shapes:
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
