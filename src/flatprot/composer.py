# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np

from flatprot.core import Structure, Coil
from flatprot.transformation import Transformer, TransformParameters
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters
from flatprot.scene import Scene, SceneGroup, secondary_structure_to_scene_element
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.style import StyleManager, StyleType


def coordinate_manager_from_structure(
    structure: Structure,
    transformer: Transformer,
    transform_parameters: TransformParameters,
    projector: OrthographicProjector,
    projection_parameters: OrthographicProjectionParameters,
) -> CoordinateManager:
    coordinate_manager = CoordinateManager()
    transformed_coords = transformer.transform(
        structure.coordinates, transform_parameters
    )
    canvas_coords, depth = projector.project(transformed_coords, projection_parameters)

    coordinate_manager.add(
        0, len(structure.coordinates), structure.coordinates, CoordinateType.COORDINATES
    )
    coordinate_manager.add(
        0, len(structure.coordinates), transformed_coords, CoordinateType.TRANSFORMED
    )
    coordinate_manager.add(
        0, len(structure.coordinates), canvas_coords, CoordinateType.CANVAS
    )
    coordinate_manager.add(0, len(structure.coordinates), depth, CoordinateType.DEPTH)
    return coordinate_manager


def structure_to_scene(
    structure: Structure,
    transformer: Transformer,
    transform_parameters: Optional[TransformParameters] = None,
    style_manager: Optional[StyleManager] = None,
) -> Scene:
    """Convert a structure to a composed scene."""
    style_manager = style_manager or StyleManager.create_default()
    canvas_settings = style_manager.get_style(StyleType.CANVAS)

    projector = OrthographicProjector()
    projection_parameters = OrthographicProjectionParameters(
        width=canvas_settings.width,
        height=canvas_settings.height,
        padding_x=canvas_settings.padding_x,
        padding_y=canvas_settings.padding_y,
        maintain_aspect_ratio=canvas_settings.maintain_aspect_ratio,
    )

    coordinate_manager = coordinate_manager_from_structure(
        structure, transformer, transform_parameters, projector, projection_parameters
    )

    root_scene = Scene()

    # Process each chain
    offset = 0
    for chain in structure:
        elements_with_z = []  # Reset for each chain

        for i, element in enumerate(chain.secondary_structure):
            print(element, type(element))
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
                "type": element.type.value,
            }

            viz_element = secondary_structure_to_scene_element(
                element.type,
                canvas_coords,
                metadata,
                style_manager,
            )

            elements_with_z.append((viz_element, np.mean(depth)))

        # Sort elements by depth (farther objects first)
        sorted_elements = [
            elem
            for elem, z in sorted(elements_with_z, key=lambda x: x[1], reverse=True)
        ]

        # Create a scene for the chain
        chain_scene = SceneGroup(
            id=f"chain_{chain.id}",
        )
        for element in sorted_elements:
            chain_scene.add_element(element)

        root_scene.add_element(chain_scene)
        offset += chain.num_residues

    return root_scene
