# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Type, Tuple
from pathlib import Path

from flatprot.core import (
    Structure,
    ResidueRange,
    SecondaryStructureType,
    logger,
)
from flatprot.scene import Scene, SceneGroup
from flatprot.scene import CoordinateCalculationError, SceneCreationError
from flatprot.scene import (
    BaseStructureSceneElement,
    BaseStructureStyle,
    HelixSceneElement,
    SheetSceneElement,
    CoilSceneElement,
    HelixStyle,
    SheetStyle,
    CoilStyle,
)

from flatprot.io import AnnotationParser

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
    default_styles: Optional[Dict[str, BaseStructureStyle]] = None,
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
        default_styles: An optional dictionary mapping lowercase secondary structure
                        type names ('helix', 'sheet', 'coil') to specific style
                        instances to be used as defaults for those element types.
                        If not provided or a type is missing, the element's own
                        default style will be used.

    Returns:
        A Scene object representing the structure.

    Raises:
        SceneCreationError: If Structure has no coordinates or other critical errors occur.
        CoordinateCalculationError: If depth calculation fails for an element.
    """
    if structure.coordinates is None or len(structure.coordinates) == 0:
        raise SceneCreationError(f"Structure '{structure.id}' has no coordinates.")

    scene = Scene(id=f"scene_{structure.id}")
    styles = default_styles or {}

    for chain_id, chain in structure:
        chain_group = SceneGroup(id=f"{structure.id}_{chain_id}")
        scene.add_element(chain_group)
        logger.debug(f"\t>Adding chain group {chain_group.id} to scene")

        elements_with_depth = []

        for ss_element in chain.secondary_structure:
            ss_type = ss_element.secondary_structure_type
            element_info = STRUCTURE_ELEMENT_MAP.get(ss_type)

            if not element_info:
                logger.warning(f"Unsupported secondary structure type: {ss_type.value}")
                continue

            ElementClass, DefaultStyleClass = element_info
            ss_range_set = ResidueRange(
                chain_id=chain_id, start=ss_element.start, end=ss_element.end
            ).to_set()

            # Determine the style: Use provided default or element's default
            element_type_key = ss_type.value.lower()
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

        # Add sorted elements to the chain group
        for element, _ in elements_with_depth:
            logger.debug(
                f"\t>Adding element {element.id} to chain group {chain_group.id}"
            )
            scene.add_element(element, parent_id=chain_group.id)

    return scene


def add_annotations_to_scene(annotations_path: Path, scene: Scene) -> None:
    """Process and add annotations to the scene.

    Args:
        annotations_path: Path to annotations file
        scene: The scene to add annotations to
        style_manager: Style manager for styling annotations
    """
    try:
        annotation_parser = AnnotationParser(
            file_path=annotations_path,
            scene=scene,
        )
        annotations = annotation_parser.parse()

        logger.info(f"Loaded {len(annotations)} annotations from {annotations_path}")
        for annotation in annotations:
            logger.debug(
                f"\t> {annotation.label} ({annotation.__class__.__name__}) to scene"
            )
            scene.add_element(annotation)
    except Exception as e:
        logger.error(f"Failed to load annotations: {str(e)}")
