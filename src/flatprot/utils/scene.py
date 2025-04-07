# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Scene utilities for FlatProt."""

from pathlib import Path
from typing import Any
import numpy as np

from flatprot.core import CoordinateManager, CoordinateType
from flatprot.style import StyleManager
from flatprot.scene import Scene, SceneGroup
from flatprot.scene.structure import secondary_structure_to_scene_element
from flatprot.io import AnnotationParser
from flatprot.core import ResidueRange, CoordinateError

from .logger import logger


def process_structure_chain(
    chain: Any,
    coordinate_manager: CoordinateManager,
    style_manager: StyleManager,
    scene: Scene,
) -> Scene:
    """Process a single chain from a structure and add elements to the scene.

    Args:
        chain: The chain to process
        residue_range: ResidueRange for this chain
        coordinate_manager: Coordinate manager with transformed and projected coordinates
        style_manager: Style manager for styling elements
        scene: The scene to add elements to

    Returns:
        scene: The updated scene
    """
    elements_with_z = []

    # Process each secondary structure element
    for element in chain.secondary_structure:
        # Create ResidueRange for this secondary structure element
        ss_range = ResidueRange(chain_id=chain.id, start=element.start, end=element.end)

        try:
            canvas_coords = coordinate_manager.get_range(
                ss_range, CoordinateType.CANVAS
            )
            depth = coordinate_manager.get_range(ss_range, CoordinateType.DEPTH)

            metadata = {
                "chain_id": chain.id,
                "start": element.start,
                "end": element.end,
                "type": element.secondary_structure_type.value,
            }

            # Get the appropriate scene element
            viz_element = secondary_structure_to_scene_element(
                element,
                canvas_coords,
                style_manager,
                metadata,
            )

            elements_with_z.append((viz_element, np.mean(depth)))

        except CoordinateError as e:
            logger.warning(f"Could not get coordinates for {ss_range}: {e}")
            continue

    # Sort elements by depth (farther objects first)
    elements_with_z.sort(key=lambda x: x[1], reverse=True)

    # Create a group for the chain
    chain_group = SceneGroup(id=chain.id)
    chain_group.metadata["chain_id"] = chain.id

    # Add chain group to scene
    scene.add_element(chain_group)
    logger.debug(f"[bold]Added chain {chain.id} to scene[/bold]")

    # Add sorted elements to the chain group
    for element, _ in elements_with_z:
        logger.debug(
            f"\t> {element.metadata['type']} from {element.metadata['start']} to {element.metadata['end']}"
        )
        scene.add_element(
            element,
            chain_group,
            element.metadata["chain_id"],
            element.metadata["start"],
            element.metadata["end"],
        )

    return scene


def process_annotations(
    annotations_path: Path, scene: Scene, style_manager: StyleManager
) -> None:
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
            style_manager=style_manager,
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
