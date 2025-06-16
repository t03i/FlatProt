# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Type, Tuple, List, Union
from pathlib import Path

from flatprot.core import (
    Structure,
    ResidueRange,
    ResidueRangeSet,
    ResidueCoordinate,
    SecondaryStructureType,
    logger,
    CoordinateCalculationError,
)
from flatprot.scene import Scene, SceneGroup
from flatprot.scene import SceneCreationError
from flatprot.scene.connection import Connection, ConnectionStyle

# Import specific annotation types
from flatprot.scene import (
    BaseStructureSceneElement,
    BaseStructureStyle,
    HelixSceneElement,
    SheetSceneElement,
    CoilSceneElement,
    HelixStyle,
    SheetStyle,
    CoilStyle,
    BaseAnnotationElement,  # Import base for type hinting
    PositionAnnotation,
    PositionAnnotationStyle,
    PositionType,
)

# Import the refactored AnnotationParser and its errors
from flatprot.io import (
    AnnotationParser,
    AnnotationError,
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
)

# Mapping from SS type enum to the corresponding SceneElement class and default style class
STRUCTURE_ELEMENT_MAP: Dict[
    SecondaryStructureType,
    Tuple[Type[BaseStructureSceneElement], Type[BaseStructureStyle]],
] = {
    SecondaryStructureType.HELIX: (HelixSceneElement, HelixStyle),
    SecondaryStructureType.SHEET: (SheetSceneElement, SheetStyle),
    SecondaryStructureType.COIL: (CoilSceneElement, CoilStyle),
    # Add other types like TURN if needed
}


def create_scene_from_structure(
    structure: Structure,
    default_styles: Optional[
        Dict[str, Union[BaseStructureStyle, ConnectionStyle]]
    ] = None,
) -> Scene:
    """Creates a Scene object populated with elements derived from a Structure.

    Iterates through chains and secondary structure elements within the provided
    Structure object, creating corresponding SceneGroup and structure-specific
    SceneElements (Helix, Sheet, Coil). Elements within each chain group are
    sorted by their mean depth based on the pre-calculated coordinates in the
    Structure object.

    Args:
        structure: The core Structure object containing chain, secondary structure,
                   and pre-projected coordinate data (X, Y, Depth).
        default_styles: An optional dictionary mapping lowercase element type names
                        ('helix', 'sheet', 'coil', 'connection') to specific style
                        instances to be used as defaults. If not provided or a type
                        is missing, the element's own default style will be used.

    Returns:
        A Scene object representing the structure.

    Raises:
        SceneCreationError: If Structure has no coordinates or other critical errors occur.
        CoordinateCalculationError: If depth calculation fails for an element.
    """
    if structure.coordinates is None or len(structure.coordinates) == 0:
        raise SceneCreationError(f"Structure '{structure.id}' has no coordinates.")
    scene = Scene(structure=structure)
    styles = default_styles or {}

    for chain_id, chain in structure:
        chain_group = SceneGroup(id=f"{structure.id}_{chain_id}")
        scene.add_element(chain_group)
        logger.debug(f"\t>Adding chain group {chain_group.id} to scene")

        elements_with_depth = []
        chain_elements_in_order: List[BaseStructureSceneElement] = []
        for ss_element in chain.secondary_structure:
            ss_type = ss_element.secondary_structure_type
            element_info = STRUCTURE_ELEMENT_MAP.get(ss_type)

            if not element_info:
                logger.warning(f"Unsupported secondary structure type: {ss_type.value}")
                continue

            ElementClass, DefaultStyleClass = element_info
            # Ensure ss_element start/end are valid ints
            start_idx = int(ss_element.start)
            end_idx = int(ss_element.end)
            ss_range_set = ResidueRangeSet(
                [ResidueRange(chain_id=chain_id, start=start_idx, end=end_idx)]
            )

            # Determine the style: Use provided default or element's default
            element_type_key = ss_type.name.lower()
            style_instance = styles.get(element_type_key, None)
            try:
                viz_element = ElementClass(
                    residue_range_set=ss_range_set,
                    style=style_instance,  # Pass the specific instance or None
                )

                # Calculate depth based on *pre-projected* coords in structure
                depth = viz_element.get_depth(structure)
                if depth is None:
                    # Raise or warn if depth calculation fails
                    raise CoordinateCalculationError(
                        f"Could not calculate depth for element {viz_element.id}"
                    )

                elements_with_depth.append((viz_element, depth))
                chain_elements_in_order.append(
                    viz_element
                )  # <-- Add element to ordered list

            except CoordinateCalculationError as e:
                # Log or handle coordinate/depth errors
                # For now, re-raise to indicate a problem
                raise SceneCreationError(
                    f"Error processing element {ss_type.value} {ss_range_set} in chain {chain_id}: {e}"
                ) from e
            except Exception as e:
                # Catch unexpected errors during element creation
                raise SceneCreationError(
                    f"Unexpected error creating element {ss_type.value} {ss_range_set} in chain {chain_id}: {e}"
                ) from e

        # Sort elements by depth (farthest first)
        elements_with_depth.sort(key=lambda x: x[1], reverse=True)

        # Add Connections between adjacent elements in the original structural order
        for i in range(len(chain_elements_in_order) - 1):
            element_i = chain_elements_in_order[i]
            element_i_plus_1 = chain_elements_in_order[i + 1]
            if element_i.is_adjacent_to(element_i_plus_1):
                # Get the default connection style if provided
                conn_style = styles.get("connection", None)
                # Ensure it's a ConnectionStyle or None before passing
                if conn_style is not None and not isinstance(
                    conn_style, ConnectionStyle
                ):
                    logger.warning(
                        f"Invalid type provided for 'connection' style. Expected ConnectionStyle, got {type(conn_style)}. Using default."
                    )
                    conn_style = None

                conn = Connection(
                    start_element=element_i,
                    end_element=element_i_plus_1,
                    style=conn_style,  # Pass the default style
                )
                logger.debug(
                    f"\t>Adding connection {conn.id} to chain group {chain_group.id}"
                )
                scene.add_element(conn, parent_id=chain_group.id)

        # Add sorted elements to the chain group
        for element, _ in elements_with_depth:
            logger.debug(
                f"\t>Adding element {element.id} to chain group {chain_group.id}"
            )
            scene.add_element(element, parent_id=chain_group.id)

    return scene


def add_annotations_to_scene(annotations_path: Path, scene: Scene) -> None:
    """Parses annotations from a file and adds them to the scene.

    Args:
        annotations_path: Path to the TOML annotations file.
        scene: The Scene object to add annotations to.

    Raises:
        AnnotationFileNotFoundError: If the annotation file is not found.
        MalformedAnnotationError: If the annotation file has invalid content or format.
        AnnotationError: For other annotation parsing related errors.
        SceneCreationError: If adding an element to the scene fails (e.g., duplicate ID).
    """
    try:
        # Instantiate the parser with just the file path
        parser = AnnotationParser(annotations_path)
        # Parse the file to get fully initialized annotation objects
        annotation_objects: List[BaseAnnotationElement] = parser.parse()

        logger.info(
            f"Loaded {len(annotation_objects)} annotations from {annotations_path}"
        )

        for annotation in annotation_objects:
            logger.debug(
                f"\t> Adding annotation '{annotation.id}' ({annotation.__class__.__name__}) to scene"
            )
            try:
                scene.add_element(annotation)
            except Exception as e:
                logger.error(
                    f"Failed to add annotation '{annotation.id}' to scene: {e}"
                )
                raise SceneCreationError(
                    f"Failed to add annotation '{annotation.id}' to scene: {e}"
                ) from e

    except (
        AnnotationFileNotFoundError,
        MalformedAnnotationError,
        AnnotationError,
    ) as e:
        logger.error(f"Failed to parse annotations from {annotations_path}: {e}")
        # Re-raise parser errors as they indicate a problem with the input file
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while adding annotations: {str(e)}")
        # Re-raise unexpected errors
        raise


def add_position_annotations_to_scene(
    scene: Scene,
    style: Optional[PositionAnnotationStyle] = None,
    annotation_level: str = "full",
) -> None:
    """Adds position annotations (N/C terminus and residue numbers) to the scene.

    Args:
        scene: The Scene object to add position annotations to.
        style: Optional custom style for position annotations.
        annotation_level: Level of annotation detail ('none', 'minimal', 'major', 'full').

    Raises:
        SceneCreationError: If adding an element to the scene fails.
    """
    if style is None:
        style = PositionAnnotationStyle()

    # Return early if no annotations requested
    if annotation_level == "none":
        logger.info("Position annotations disabled")
        return

    # Configure style settings based on annotation level
    style.show_terminus = annotation_level in ["minimal", "major", "full"]
    style.show_residue_numbers = annotation_level in ["major", "full"]

    # Get all structure elements in sequence
    try:
        structure_elements = scene.get_sequential_structure_elements()
    except Exception as e:
        logger.error(f"Failed to get sequential structure elements: {e}")
        raise SceneCreationError(
            f"Failed to get structure elements for position annotations: {e}"
        ) from e

    if not structure_elements:
        logger.warning("No structure elements found for position annotations")
        return

    # Add N-terminus annotation
    if style.show_terminus and structure_elements:
        first_element = structure_elements[0]
        n_terminus_id = f"pos_n_terminus_{first_element.id}"

        try:
            n_terminus = PositionAnnotation(
                id=n_terminus_id,
                target=first_element.residue_range_set,
                position_type=PositionType.N_TERMINUS,
                text="N",
                style=style,
            )
            scene.add_element(n_terminus)
            logger.debug(f"Added N-terminus annotation: {n_terminus_id}")
        except Exception as e:
            logger.error(f"Failed to add N-terminus annotation: {e}")
            raise SceneCreationError(f"Failed to add N-terminus annotation: {e}") from e

    # Add C-terminus annotation
    if style.show_terminus and structure_elements:
        last_element = structure_elements[-1]
        c_terminus_id = f"pos_c_terminus_{last_element.id}"

        try:
            c_terminus = PositionAnnotation(
                id=c_terminus_id,
                target=last_element.residue_range_set,
                position_type=PositionType.C_TERMINUS,
                text="C",
                style=style,
            )
            scene.add_element(c_terminus)
            logger.debug(f"Added C-terminus annotation: {c_terminus_id}")
        except Exception as e:
            logger.error(f"Failed to add C-terminus annotation: {e}")
            raise SceneCreationError(f"Failed to add C-terminus annotation: {e}") from e

    # Add residue number annotations for helices and sheets only
    if style.show_residue_numbers:
        for element in structure_elements:
            # Only add residue numbers for helices and sheets, not coils
            if not isinstance(element, (HelixSceneElement, SheetSceneElement)):
                continue

            if not element.residue_range_set.ranges:
                logger.warning(f"Element {element.id} has no residue ranges")
                continue

            # Get the first range (assuming single range per element)
            residue_range = element.residue_range_set.ranges[0]

            # Apply filtering based on annotation level
            structure_length = residue_range.end - residue_range.start + 1

            if annotation_level == "major" and structure_length < 3:
                # Skip short structures in 'major' mode to reduce clutter
                logger.debug(
                    f"Skipping short structure {element.id} ({structure_length} residues)"
                )
                continue
            elif annotation_level == "full":
                # Include all structures in 'full' mode (original behavior)
                pass

            # Add start position annotation
            start_id = f"pos_start_{element.id}"
            try:
                # Create specific coordinate for start position
                start_coord = ResidueCoordinate(
                    residue_range.chain_id,
                    residue_range.start,
                    None,  # residue_type not needed
                    0,  # coordinate_index
                )
                start_annotation = PositionAnnotation(
                    id=start_id,
                    target=start_coord,
                    position_type=PositionType.RESIDUE_NUMBER,
                    text=str(residue_range.start),
                    style=style,
                )
                scene.add_element(start_annotation)
                logger.debug(f"Added start position annotation: {start_id}")
            except Exception as e:
                logger.error(
                    f"Failed to add start position annotation for {element.id}: {e}"
                )
                # Continue with other annotations even if one fails
                continue

            # Add end position annotation (only if range has more than one residue)
            if residue_range.end != residue_range.start:
                end_id = f"pos_end_{element.id}"
                try:
                    # Create specific coordinate for end position
                    end_coord = ResidueCoordinate(
                        residue_range.chain_id,
                        residue_range.end,
                        None,  # residue_type not needed
                        0,  # coordinate_index
                    )
                    end_annotation = PositionAnnotation(
                        id=end_id,
                        target=end_coord,
                        position_type=PositionType.RESIDUE_NUMBER,
                        text=str(residue_range.end),
                        style=style,
                    )
                    scene.add_element(end_annotation)
                    logger.debug(f"Added end position annotation: {end_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to add end position annotation for {element.id}: {e}"
                    )
                    # Continue with other annotations even if one fails
                    continue

    logger.info(f"Added position annotations to scene (level: {annotation_level})")
