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
    SceneError,
    LineAnnotation,
    AreaAnnotation,
    BaseAnnotationElement,
    BaseStructureSceneElement,
    TargetResidueNotFoundError,
    Connection,
)
from flatprot.core import logger, CoordinateCalculationError, Structure

from .svg_structure import (
    _draw_coil,
    _draw_helix,
    _draw_sheet,
    _draw_connection_line,
)
from .svg_annotations import (
    _draw_point_annotation,
    _draw_line_annotation,
    _draw_area_annotation,
)

# --- Helper Functions --- #


# Removed _get_rendered_coords_for_annotation helper function


# --- Renderer Class --- #


def _draw_connection_element(
    element: Connection,
    structure: Structure,
) -> Optional[Any]:
    """Retrieves connection points and calls the line drawing function."""
    try:
        start_element = element.start_element
        end_element = element.end_element

        # Get end point of the first element and start point of the second
        start_point_2d = start_element.get_end_connection_point(structure)
        end_point_2d = end_element.get_start_connection_point(structure)

        if start_point_2d is not None and end_point_2d is not None:
            # Call the drawing function from svg_structure
            return _draw_connection_line(
                start_point=start_point_2d,
                end_point=end_point_2d,
                style=element.style,
                id=element.id,
            )
        else:
            logger.warning(
                f"Could not get connection points for Connection {element.id}. "
                f"Start: {start_point_2d is not None}, End: {end_point_2d is not None}"
            )
            return None
    except Exception as e:
        logger.error(
            f"Error drawing Connection element {element.id}: {e}", exc_info=True
        )
        return None


class SVGRenderer:
    """Renders a Scene object to an SVG Drawing.

    Attributes:
        scene: The Scene object to render.
        width: The width of the SVG canvas.
        height: The height of the SVG canvas.
        background_color: Optional background color for the SVG.
        background_opacity: Opacity for the background color.
        padding: Padding around the content within the viewBox.
    """

    DEFAULT_WIDTH = 600
    DEFAULT_HEIGHT = 400
    DEFAULT_BG_COLOR = "#FFFFFF"
    DEFAULT_BG_OPACITY = 1.0
    DEFAULT_PADDING = 10  # Default padding in SVG units

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
        padding: int = DEFAULT_PADDING,
    ):
        """Initializes the SVGRenderer.

        Args:
            scene: The Scene object containing the elements to render.
            width: Desired width of the SVG canvas.
            height: Desired height of the SVG canvas.
            background_color: Background color (CSS string, e.g., '#FFFFFF' or 'white'). None for transparent.
            background_opacity: Background opacity (0.0 to 1.0).
            padding: Padding around the content within the viewBox.
        """
        if not isinstance(scene, Scene):
            raise TypeError("Renderer requires a valid Scene object.")

        self.scene = scene
        self.width = width
        self.height = height
        self.background_color = background_color
        self.background_opacity = background_opacity
        self.padding = padding  # Store padding
        self._element_map: Dict[str, BaseSceneElement] = {
            e.id: e for e in self.scene.get_all_elements()
        }

    def _collect_and_sort_renderables(
        self,
    ) -> Tuple[
        List[Tuple[float, BaseStructureSceneElement]],
        List[Tuple[float, BaseAnnotationElement]],
        List[Tuple[float, Connection]],
    ]:
        """Traverses scene, collects renderable leaf nodes, sorts them by Z-depth."""
        structure_elements: List[Tuple[float, BaseStructureSceneElement]] = []
        annotation_elements: List[Tuple[float, BaseAnnotationElement]] = []
        connection_elements: List[Tuple[float, Connection]] = []
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
            elif isinstance(element, Connection):
                logger.debug(f"Collecting Connection Element: {element.id}")  # DEBUG
                connection_elements.append((render_depth, element))
            # else: Ignore other potential non-group, non-renderable types
        logger.debug("--- Finished _collect_and_sort_renderables ---")  # DEBUG

        # Sort structure elements by depth (ascending)
        structure_elements.sort(key=lambda item: item[0])
        # Annotations are naturally last due to depth=inf, but explicit sort doesn't hurt
        annotation_elements.sort(key=lambda item: item[0])
        # Sort connections by depth (ascending) - based on average of connected elements
        connection_elements.sort(key=lambda item: item[0])

        return structure_elements, annotation_elements, connection_elements

    def _build_svg_hierarchy(
        self,
        element: BaseSceneElement,
        parent_svg_group: Group,
        svg_group_map: Dict[str, Group],
    ) -> None:
        """Recursively builds SVG groups mirroring SceneGroups."""
        if not isinstance(element, SceneGroup):
            return  # Only process groups

        svg_transform_str = str(element.transforms)
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
    ]:
        """Pre-calculates 2D coordinates and connection points for structure elements."""
        # ordered_elements: List[BaseStructureSceneElement] = [] # Keep type hint
        element_coords_cache: Dict[str, np.ndarray] = {}
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
            return [], {}  # Return empty if fetching/sorting failed

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
                    continue

                coords_2d = coords[:, :2]  # Ensure we only use X, Y
                element_coords_cache[element_id] = coords_2d

            except Exception as e:
                logger.error(
                    f"Error preparing render data for element {element_id}: {e}",
                    exc_info=True,
                )
                # Add placeholders if preparation fails for an element
                element_coords_cache[element_id] = np.empty((0, 2))
                continue

        return ordered_elements, element_coords_cache

    def render(self) -> Drawing:
        """Renders the scene to a drawsvg.Drawing object."""
        drawing = Drawing(self.width, self.height)
        svg_group_map: Dict[str, Group] = {}

        # 1. Add Background
        if self.background_color:
            drawing.append(
                Rectangle(
                    0,
                    0,
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

        # 3. Prepare Render Data for Structure Elements
        try:
            (
                _,
                element_coords_cache,
            ) = self._prepare_render_data()
        except Exception as e:
            logger.error(f"Failed to prepare render data: {e}", exc_info=True)
            return drawing

        # 3.5 Collect and Sort All Renderable Elements
        try:
            (
                sorted_structure_elements,
                sorted_annotations,
                sorted_connections,
            ) = self._collect_and_sort_renderables()
        except Exception as e:
            logger.error(f"Failed to collect/sort renderables: {e}", exc_info=True)
            return drawing

        # 4. Draw Structure Elements (sorted by depth)
        for depth, element in sorted_structure_elements:
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

        # 5. Draw Connection Elements (sorted by depth)
        for _, element in sorted_connections:
            connection_line_svg = _draw_connection_element(
                element, self.scene.structure
            )
            if connection_line_svg:
                # Determine the parent group for the connection
                parent_group_id = (
                    element.parent.id if element.parent else "flatprot-root"
                )
                target_svg_group = svg_group_map.get(parent_group_id)
                if target_svg_group:
                    target_svg_group.append(connection_line_svg)
                else:
                    # Log error if the parent group is not found in the map
                    logger.error(
                        f"Could not find target SVG group '{parent_group_id}' for connection element {element.id}"
                    )

        # 6. Draw Annotation Elements (sorted by depth - effectively always last)
        for _, element in sorted_annotations:
            element_type = type(element)
            draw_func = self.ANNOTATION_DRAW_MAP.get(element_type)
            if not draw_func:
                logger.warning(
                    f"No drawing func for annotation type: {element_type.__name__}"
                )
                continue

            try:
                # Calculate coordinates using the annotation's own method + resolver
                rendered_coords = element.get_coordinates(self.scene.resolver)
            except (
                CoordinateCalculationError,
                SceneError,
                ValueError,
                TargetResidueNotFoundError,
            ) as e:
                # Catch expected errors during coordinate resolution/calculation
                logger.error(
                    f"Could not get coordinates for annotation '{element.id}': {e}"
                )
                logger.exception(e)
                continue  # Skip rendering this annotation
            except Exception as e:
                # Catch unexpected errors
                logger.error(
                    f"Unexpected error getting coordinates for annotation '{element.id}': {e}",
                    exc_info=True,
                )
                logger.exception(e)
                continue  # Skip rendering this annotation

            # Basic validation (redundant with Scene checks, but safe)
            if rendered_coords is None or rendered_coords.size == 0:
                logger.debug(
                    f"Skipping annotation {element.id} due to missing/empty resolved coordinates."
                )
                continue

            # Call the drawing function with the resolved coordinates
            try:
                svg_shapes = draw_func(element, rendered_coords)
                if svg_shapes:
                    # Determine the parent group for the annotation
                    parent_group_id = (
                        element.parent.id if element.parent else "flatprot-root"
                    )
                    target_svg_group = svg_group_map.get(parent_group_id)
                    if target_svg_group:
                        target_svg_group.append(svg_shapes)
                    else:
                        # Log error if the parent group is not found
                        logger.error(
                            f"Could not find target SVG group '{parent_group_id}' for annotation element {element.id}"
                        )
            except Exception as e:
                logger.error(
                    f"Error drawing annotation '{element.id}' (type {element_type.__name__}): {e}",
                    exc_info=True,
                )

            drawing.view_box = (0, 0, self.width, self.height)

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
