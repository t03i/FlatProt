# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate manager utilities for FlatProt."""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np

from flatprot.core import CoordinateManager, CoordinateType
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters
from flatprot.style import StyleManager
from flatprot.projection import ProjectionError
from flatprot.core.components import Structure
from flatprot.transformation import (
    InertiaTransformer,
    MatrixTransformer,
    MatrixTransformParameters,
    TransformationMatrix,
    InertiaTransformParameters,
    BaseTransformer,
)
from flatprot.io import MatrixLoader
from flatprot.transformation import TransformationError

from .logger import logger


def get_projection_parameters(
    style_manager: StyleManager,
) -> OrthographicProjectionParameters:
    """Create orthographic projection parameters based on style manager settings.

    Args:
        style_manager: Style manager containing canvas style information

    Returns:
        OrthographicProjectionParameters configured with style-based settings
    """
    # Default values
    width = 800
    height = 600
    padding_x = 0.05
    padding_y = 0.05
    maintain_aspect_ratio = True

    # Try to get values from style manager
    if hasattr(style_manager, "canvas_style"):
        canvas_style = style_manager.canvas_style
        if hasattr(canvas_style, "width") and canvas_style.width is not None:
            width = canvas_style.width
        if hasattr(canvas_style, "height") and canvas_style.height is not None:
            height = canvas_style.height
        if hasattr(canvas_style, "padding_x") and canvas_style.padding_x is not None:
            padding_x = canvas_style.padding_x
        if hasattr(canvas_style, "padding_y") and canvas_style.padding_y is not None:
            padding_y = canvas_style.padding_y
        if (
            hasattr(canvas_style, "maintain_aspect_ratio")
            and canvas_style.maintain_aspect_ratio is not None
        ):
            maintain_aspect_ratio = canvas_style.maintain_aspect_ratio

    return OrthographicProjectionParameters(
        width=width,
        height=height,
        padding_x=padding_x,
        padding_y=padding_y,
        maintain_aspect_ratio=maintain_aspect_ratio,
    )


def apply_projection(
    coordinate_manager: CoordinateManager,
    style_manager: StyleManager,
) -> CoordinateManager:
    """Apply projection to transformed coordinates based on style settings."""
    try:
        if not coordinate_manager.has_type(CoordinateType.TRANSFORMED):
            logger.warning("No transformed coordinates to project")
            return coordinate_manager

        # Set up projector and parameters
        projector = OrthographicProjector()
        projection_params = get_projection_parameters(style_manager)

        # Process each range of transformed coordinates individually
        for (start, end), transformed_coords in coordinate_manager.coordinates[
            CoordinateType.TRANSFORMED
        ].items():
            # Project this specific range
            canvas_coords, depth = projector.project(
                transformed_coords, projection_params
            )

            # Add projected coordinates directly with the same range
            coordinate_manager.add(start, end, canvas_coords, CoordinateType.CANVAS)
            coordinate_manager.add(start, end, depth, CoordinateType.DEPTH)

        return coordinate_manager

    except Exception as e:
        logger.error(f"Error applying projection: {str(e)}")
        raise ProjectionError(f"Failed to apply projection: {str(e)}")


def create_transformer(
    structure: Structure, matrix_path: Optional[Path] = None
) -> Tuple[
    BaseTransformer, Union[MatrixTransformParameters, InertiaTransformParameters]
]:
    """Create an appropriate transformer based on input.

    Args:
        structure: The protein structure to transform
        matrix_path: Path to a custom transformation matrix file

    Returns:
        Tuple containing the transformer and its parameters

    Raises:
        TransformationError: If there's an error creating the transformer
    """
    has_valid_residues = (
        hasattr(structure, "residues")
        and structure.residues is not None
        and len(structure.residues) > 0
    )

    # Try to use custom matrix if provided
    if matrix_path:
        try:
            matrix_loader = MatrixLoader(matrix_path)
            transformation_matrix = matrix_loader.load()
            return MatrixTransformer(), MatrixTransformParameters(
                matrix=transformation_matrix
            )
        except Exception as e:
            logger.warning(f"Failed to load matrix from {matrix_path}: {str(e)}")
            logger.warning("Falling back to inertia-based transformation")

    # Use inertia transformer if structure has necessary properties
    if (
        hasattr(structure, "coordinates")
        and structure.coordinates is not None
        and has_valid_residues
    ):
        return InertiaTransformer(), InertiaTransformParameters(
            residues=structure.residues
        )

    # Fallback to identity matrix if necessary
    logger.warning(
        "Structure lacks required properties for inertia transformation, using identity matrix"
    )
    rotation = np.eye(3)
    translation = np.zeros(3)
    identity_matrix = TransformationMatrix(rotation=rotation, translation=translation)
    return MatrixTransformer(), MatrixTransformParameters(matrix=identity_matrix)


def create_coordinate_manager(
    structure: Structure, matrix_path: Optional[Path] = None
) -> CoordinateManager:
    """Apply transformation to a protein structure and create a coordinate manager.

    Args:
        structure: The protein structure to transform
        matrix_path: Path to a numpy matrix file for custom transformation.
                    If None, uses InertiaTransformer.

    Returns:
        CoordinateManager preloaded with original and transformed coordinates

    Raises:
        TransformationError: If there's an error during transformation
    """
    try:
        # Check if structure has coordinates
        if not hasattr(structure, "coordinates") or structure.coordinates is None:
            logger.warning("Structure has no coordinates, skipping transformation")
            return CoordinateManager()

        # Create coordinate manager and add original coordinates
        coordinate_manager = CoordinateManager()
        coordinate_manager.add(
            0,
            len(structure.coordinates),
            structure.coordinates,
            CoordinateType.COORDINATES,
        )

        # Create transformer and transform coordinates
        transformer, transform_parameters = create_transformer(structure, matrix_path)
        transformed_coords = transformer.transform(
            structure.coordinates.copy(), transform_parameters
        )

        # Add transformed coordinates
        coordinate_manager.add(
            0,
            len(structure.coordinates),
            transformed_coords,
            CoordinateType.TRANSFORMED,
        )

        return coordinate_manager

    except Exception as e:
        logger.error(f"Error applying transformation: {str(e)}")
        raise TransformationError(f"Failed to apply transformation: {str(e)}")
