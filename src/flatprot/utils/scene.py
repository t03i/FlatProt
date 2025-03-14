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

from . import console


def process_structure_chain(
    chain: Any,
    offset: int,
    coordinate_manager: CoordinateManager,
    style_manager: StyleManager,
    scene: Scene,
) -> int:
    """Process a single chain from a structure and add elements to the scene.

    Args:
        chain: The chain to process
        offset: The current coordinate offset
        coordinate_manager: Coordinate manager with transformed and projected coordinates
        style_manager: Style manager for styling elements
        scene: The scene to add elements to

    Returns:
        int: The new offset after processing this chain
    """
    elements_with_z = []  # Reset for each chain

    # Process each secondary structure element
    for element in chain.secondary_structure:
        start_idx = offset + element.start
        end_idx = offset + element.end + 1

        canvas_coords = coordinate_manager.get(
            start_idx, end_idx, CoordinateType.CANVAS
        )
        depth = coordinate_manager.get(start_idx, end_idx, CoordinateType.DEPTH)

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

    # Sort elements by depth (farther objects first)
    elements_with_z.sort(key=lambda x: x[1], reverse=True)

    # Create a group for the chain
    chain_group = SceneGroup(id=chain.id)
    chain_group.metadata["chain_id"] = chain.id

    # Add chain group to scene
    scene.add_element(chain_group)

    # Add sorted elements to the chain group
    for element, _ in elements_with_z:
        scene.add_element(
            element,
            chain_group,
            element.metadata["chain_id"],
            element.metadata["start"],
            element.metadata["end"],
        )

    # Return the updated offset for next chain
    return offset + chain.num_residues


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

        console.print(
            f"[blue]Loaded {len(annotations)} annotations from {annotations_path}[/blue]"
        )
        for annotation in annotations:
            console.print(
                f"  Adding {annotation.label} ({annotation.__class__.__name__}) to scene"
            )
            scene.add_element(annotation)
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load annotations: {str(e)}[/yellow]")
