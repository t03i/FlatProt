# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate manager utilities for FlatProt."""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np

from flatprot.core import CoordinateManager, CoordinateType
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters
from flatprot.style import StyleManager, StyleType
from flatprot.projection import ProjectionError
from flatprot.core import Structure, ResidueRangeSet
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

    canvas_style = style_manager.get_style(StyleType.CANVAS)

    return OrthographicProjectionParameters(
        width=canvas_style.width,
        height=canvas_style.height,
        padding_x=canvas_style.padding_x,
        padding_y=canvas_style.padding_y,
        maintain_aspect_ratio=canvas_style.maintain_aspect_ratio,
    )


def apply_projection(
    coordinate_manager: CoordinateManager,
    style_manager: StyleManager,
) -> CoordinateManager:
    """Apply projection to transformed coordinates based on style settings using vectorized operations."""
    try:
        if not coordinate_manager.has_type(CoordinateType.TRANSFORMED):
            logger.warning("No transformed coordinates to project")
            return coordinate_manager

        # Retrieve all transformed coordinates at once
        all_transformed_coords = coordinate_manager.get_all(CoordinateType.TRANSFORMED)

        if all_transformed_coords.size == 0:
            logger.warning("Transformed coordinates exist but are empty.")
            # Ensure canvas and depth types are cleared if they existed before
            coordinate_manager._coordinates[CoordinateType.CANVAS].clear()
            coordinate_manager._coordinates[CoordinateType.DEPTH].clear()
            return coordinate_manager

        # Set up projector and parameters
        projector = OrthographicProjector()
        projection_params = get_projection_parameters(style_manager)

        # Perform projection on the entire dataset once
        logger.info("Performing vectorized projection...")
        all_canvas_coords, all_depth_values = projector.project(
            all_transformed_coords, projection_params
        )
        logger.info("Projection complete.")

        # Use apply_vectorized_transform to store the results efficiently.
        # The transform function here just returns the pre-calculated array.
        # It ignores the input coords passed by apply_vectorized_transform
        # because the projection was already done outside.

        # Store Canvas Coordinates
        coordinate_manager.apply_vectorized_transform(
            source_coord_type=CoordinateType.TRANSFORMED,  # Source is needed for keys
            target_coord_type=CoordinateType.CANVAS,
            transform_func=lambda _: all_canvas_coords,
        )
        logger.info("Stored canvas coordinates.")

        # Store Depth Coordinates
        coordinate_manager.apply_vectorized_transform(
            source_coord_type=CoordinateType.TRANSFORMED,  # Source is needed for keys
            target_coord_type=CoordinateType.DEPTH,
            transform_func=lambda _: all_depth_values,
        )
        logger.info("Stored depth coordinates.")

        return coordinate_manager

    except Exception as e:
        logger.error(f"Error applying vectorized projection: {str(e)}", exc_info=True)
        raise ProjectionError(f"Failed to apply vectorized projection: {str(e)}")


def get_inertia_parameters(
    structure: Structure,
) -> Optional[InertiaTransformParameters]:
    """Get parameters for InertiaTransformer if structure is suitable.

    Args:
        structure: The protein structure.

    Returns:
        InertiaTransformParameters if the structure has coordinates and residues,
        otherwise None.
    """
    has_valid_residues = (
        hasattr(structure, "residues")
        and structure.residues is not None
        and len(structure.residues) > 0
    )
    has_coordinates = (
        hasattr(structure, "coordinates") and structure.coordinates is not None
    )

    if has_coordinates and has_valid_residues:
        return InertiaTransformParameters(residues=structure.residues)
    else:
        logger.debug("Structure not suitable for inertia transformation.")
        return None


def get_matrix_parameters(
    matrix_path: Optional[Path] = None,
) -> MatrixTransformParameters:
    """Get parameters for MatrixTransformer, loading from path or using identity.

    Args:
        matrix_path: Optional path to a transformation matrix file.

    Returns:
        MatrixTransformParameters containing the loaded or identity matrix.
    """
    transformation_matrix: TransformationMatrix
    if matrix_path:
        try:
            logger.debug(
                f"Attempting to load transformation matrix from: {matrix_path}"
            )
            matrix_loader = MatrixLoader(matrix_path)
            transformation_matrix = matrix_loader.load()
            logger.debug("Successfully loaded transformation matrix.")
            return MatrixTransformParameters(matrix=transformation_matrix)
        except Exception as e:
            logger.warning(f"Failed to load matrix from {matrix_path}: {str(e)}")
            logger.warning("Falling back to identity matrix.")
            # Fallthrough to identity matrix case

    # Use identity matrix if no path or loading failed
    logger.debug("Using identity matrix for transformation.")
    rotation = np.eye(3)
    translation = np.zeros(3)
    identity_matrix = TransformationMatrix(rotation=rotation, translation=translation)
    return MatrixTransformParameters(matrix=identity_matrix)


def get_transformer_and_parameters(
    structure: Structure, matrix_path: Optional[Path] = None
) -> Tuple[
    BaseTransformer, Union[MatrixTransformParameters, InertiaTransformParameters]
]:
    """Determine and create the appropriate transformer and its parameters.

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


def apply_transformation(
    coordinate_manager: CoordinateManager,
    structure: Structure,
    matrix_path: Optional[Path] = None,
) -> None:
    """Applies the appropriate transformation (inertia or matrix) to coordinates.

    Determines whether to use inertia-based transformation or matrix-based
    (from file or identity) and applies it to the COORDINATES type, storing
    the result in the TRANSFORMED type using vectorized operations.

    Args:
        coordinate_manager: The manager containing original coordinates.
        structure: The protein structure (needed for inertia parameters).
        matrix_path: Optional path to a custom transformation matrix.

    Raises:
        TransformationError: If transformation fails.
    """
    transformer: BaseTransformer
    transform_parameters: Union[MatrixTransformParameters, InertiaTransformParameters]

    # Prioritize custom matrix if provided
    if matrix_path:
        logger.info("Custom matrix path provided. Attempting matrix transformation.")
        transformer = MatrixTransformer()
        transform_parameters = get_matrix_parameters(matrix_path)
    else:
        # Try inertia transformation if possible
        inertia_params = get_inertia_parameters(structure)
        if inertia_params:
            logger.info("Structure suitable for inertia transformation.")
            transformer = InertiaTransformer()
            transform_parameters = inertia_params
        else:
            # Fallback to identity matrix transformation
            logger.info(
                "Structure not suitable for inertia transformation. Using identity matrix."
            )
            transformer = MatrixTransformer()
            transform_parameters = get_matrix_parameters(None)  # Force identity

    # Define the vectorized transformation function
    def transform_func(coords: np.ndarray) -> np.ndarray:
        # Copy coords to prevent potential in-place modification by transformer
        return transformer.transform(coords.copy(), transform_parameters)

    # Apply the transformation using the vectorized method
    logger.info(f"Applying transformation using {transformer.__class__.__name__}...")
    try:
        coordinate_manager.apply_vectorized_transform(
            source_coord_type=CoordinateType.COORDINATES,
            target_coord_type=CoordinateType.TRANSFORMED,
            transform_func=transform_func,
        )
        logger.info("Transformation complete.")
    except Exception as e:
        logger.error(f"Error during vectorized transformation: {str(e)}", exc_info=True)
        # Re-raise as TransformationError for consistent error handling
        raise TransformationError(f"Transformation failed: {str(e)}")


def create_coordinate_manager(
    structure: Structure, matrix_path: Optional[Path] = None
) -> CoordinateManager:
    """Create a coordinate manager and apply transformation to the structure's coordinates.

    Args:
        structure: The protein structure containing coordinates.
        matrix_path: Optional path to a numpy matrix file for custom transformation.
                     If None, attempts inertia transformation, falling back to identity.

    Returns:
        CoordinateManager preloaded with original and (if applicable) transformed coordinates.

    Raises:
        TransformationError: If there's an error during transformation.
    """
    try:
        # Create coordinate manager
        coordinate_manager = CoordinateManager()

        if not hasattr(structure, "coordinates") or structure.coordinates is None:
            logger.warning("Structure has no coordinates. Returning empty manager.")
            return coordinate_manager

        # Create ResidueRangeSet from structure
        ranges = []
        for chain in structure:
            ranges.extend(chain.to_ranges())

        if not ranges:
            logger.warning(
                "No valid residue ranges found in structure. Transformation skipped."
            )
            return coordinate_manager

        range_set = ResidueRangeSet(ranges)

        # Add original coordinates
        try:
            coordinate_manager.add_range_set(
                range_set, structure.coordinates, CoordinateType.COORDINATES
            )
        except Exception as e:
            logger.error(f"Failed to add original coordinates: {str(e)}", exc_info=True)
            # Decide if we should continue without original coords or raise
            raise TransformationError(f"Failed to add original coordinates: {str(e)}")

        # Apply the chosen transformation (inertia or matrix)
        apply_transformation(coordinate_manager, structure, matrix_path)

        return coordinate_manager

    except TransformationError:  # Catch TransformationError specifically
        raise  # Re-raise errors from apply_transformation
    except Exception as e:
        # Catch other unexpected errors during setup
        logger.error(
            f"Unexpected error creating coordinate manager: {str(e)}", exc_info=True
        )
        raise TransformationError(f"Failed to create coordinate manager: {str(e)}")
