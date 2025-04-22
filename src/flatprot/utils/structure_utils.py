# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for transforming and projecting Structure objects."""

from pathlib import Path
from typing import Optional
import numpy as np

from flatprot.core import Structure, logger

from flatprot.io import MatrixLoader

from flatprot.io import MatrixFileNotFoundError, MatrixFileError
from flatprot.projection import (
    OrthographicProjection,
    OrthographicProjectionParameters,
    ProjectionError,
)
from flatprot.transformation import (
    InertiaTransformer,
    InertiaTransformationParameters,
    InertiaTransformationArguments,
    MatrixTransformer,
    MatrixTransformParameters,
    TransformationMatrix,
    TransformationError,
)


def _load_transformation_matrix(
    matrix_path: Optional[Path],
) -> TransformationMatrix:
    """Load a transformation matrix from a file or return the identity matrix.

    Args:
        matrix_path: Optional path to a transformation matrix file (.npy).

    Returns:
        The loaded TransformationMatrix or an identity matrix.

    Raises:
        TransformationError: If loading fails for reasons other than file not found
                             or invalid matrix format handled by MatrixLoader.
    """
    if matrix_path:
        try:
            logger.debug(
                f"Attempting to load transformation matrix from: {matrix_path}"
            )
            # MatrixLoader constructor might raise MatrixFileNotFoundError
            # MatrixLoader.load might raise InvalidMatrixFormatError, InvalidMatrixDimensionsError etc.
            matrix_loader = MatrixLoader(matrix_path)
            transformation_matrix = matrix_loader.load()
            logger.debug("Successfully loaded transformation matrix.")
            return transformation_matrix
        except (MatrixFileNotFoundError, MatrixFileError) as e:
            # Handle file-related errors gracefully by falling back
            logger.warning(f"Failed to load matrix from {matrix_path}: {str(e)}")
            logger.warning("Falling back to identity matrix.")
        except Exception as e:
            # Handle other potential errors during loading (e.g., numpy errors)
            logger.error(
                f"Unexpected error loading matrix file {matrix_path}: {str(e)}",
                exc_info=True,
            )
            # Re-raise as TransformationError for consistent API
            raise TransformationError(
                f"Failed to load matrix file {matrix_path}: {str(e)}"
            ) from e

    # Use identity matrix if no path or loading failed gracefully
    logger.debug("Using identity matrix for transformation.")
    rotation = np.eye(3)
    translation = np.zeros(3)
    identity_matrix = TransformationMatrix(rotation=rotation, translation=translation)
    return identity_matrix


def transform_structure_with_matrix(
    structure: Structure, matrix_path: Optional[Path] = None
) -> Structure:
    """Transforms a Structure using a matrix (loaded from path or identity).

    Args:
        structure: The protein structure to transform.
        matrix_path: Optional path to a numpy matrix file (.npy) for transformation.
                     If None or loading fails gracefully, the identity matrix is used.

    Returns:
        A new Structure object with transformed coordinates.

    Raises:
        TransformationError: If the transformation process fails unexpectedly (e.g.,
                             matrix loading error, issues during transformation math).
        ValueError: If the structure lacks coordinates.
    """
    if not hasattr(structure, "coordinates") or structure.coordinates is None:
        raise ValueError("Structure has no coordinates to transform.")
    if structure.coordinates.size == 0:
        logger.warning("Structure coordinates are empty, returning original structure.")
        return structure  # Return unchanged structure if no coords

    try:
        transformation_matrix = _load_transformation_matrix(matrix_path)
        # Correctly instantiate MatrixTransformer using its parameters class
        transformer_params = MatrixTransformParameters(matrix=transformation_matrix)
        transformer = MatrixTransformer(parameters=transformer_params)

        logger.info(
            f"Applying matrix transformation (source: {matrix_path or 'Identity'})..."
        )

        new_structure = structure.apply_vectorized_transformation(
            lambda coords: transformer.transform(coords, arguments=None)
        )
        logger.info("Matrix transformation complete.")
        return new_structure

    except Exception as e:
        # Catch potential errors from apply_vectorized_transformation or transform_func
        logger.error(f"Error during matrix transformation: {str(e)}", exc_info=True)
        # Re-raise as TransformationError if not already (e.g. ValueError from shape mismatch)
        if isinstance(e, TransformationError):
            raise
        raise TransformationError(f"Matrix transformation failed: {str(e)}") from e


def transform_structure_with_inertia(
    structure: Structure,
    custom_config_params: Optional[InertiaTransformationParameters] = None,
) -> Structure:
    """Transforms a Structure using its principal axes of inertia.

    Args:
        structure: The protein structure to transform.
        custom_config_params: Optional InertiaTransformationParameters to override defaults
                              (e.g., custom residue weights or disabling weights) passed
                              to the InertiaTransformer's initialization.

    Returns:
        A new Structure object with transformed coordinates oriented along principal axes.

    Raises:
        TransformationError: If the structure is unsuitable for inertia calculation
                             (e.g., lacks coordinates or residues), or if the
                             transformation process fails mathematically.
        ValueError: If the structure lacks coordinates or residues required for the process.
    """
    if not hasattr(structure, "coordinates") or structure.coordinates is None:
        raise ValueError("Structure has no coordinates for inertia transformation.")
    if not hasattr(structure, "residues") or not structure.residues:
        raise ValueError("Structure has no residues for inertia transformation.")
    if structure.coordinates.size == 0:
        logger.warning("Structure coordinates are empty, returning original structure.")
        return structure  # Return unchanged structure

    try:
        # Use provided config params or create default ones for the transformer's __init__
        # InertiaTransformationParameters likely holds config like residue_weights dict
        inertia_config_parameters = (
            custom_config_params or InertiaTransformationParameters.default()
        )

        # Instantiate the transformer with its configuration parameters
        transformer = InertiaTransformer(parameters=inertia_config_parameters)

        logger.info("Applying inertia transformation...")

        # Prepare the runtime arguments needed specifically for the transform call
        # InertiaTransformArguments (aliased from InertiaTransformationParameters in inertia.py)
        # likely holds data specific to this call, like the list of residues.
        transform_arguments = InertiaTransformationArguments(
            residues=structure.residues
        )

        # Wrap the transformer's transform method
        def transform_func(coords: np.ndarray) -> np.ndarray:
            # Pass coordinates and the required runtime arguments object
            return transformer.transform(coords, arguments=transform_arguments)

        new_structure = structure.apply_vectorized_transformation(transform_func)
        logger.info("Inertia transformation complete.")
        return new_structure

    except Exception as e:
        logger.error(f"Error during inertia transformation: {str(e)}", exc_info=True)
        # Re-raise as TransformationError if not already one
        if isinstance(e, TransformationError):
            raise
        raise TransformationError(f"Inertia transformation failed: {str(e)}") from e


def project_structure_orthographically(
    structure: Structure,
    width: int,
    height: int,
    padding_x: float = 0.05,
    padding_y: float = 0.05,
    maintain_aspect_ratio: bool = True,
    center_projection: bool = True,
    view_direction: Optional[np.ndarray] = None,
    up_vector: Optional[np.ndarray] = None,
) -> Structure:
    """Projects the coordinates of a Structure orthographically, returning a new Structure.

    Assumes the input structure's coordinates are already appropriately transformed
    (e.g., centered and oriented via inertia or matrix transformation).
    The coordinates of the returned Structure will be the projected (X, Y, Depth) values.

    Args:
        structure: The Structure object whose coordinates are to be projected.
                   Assumes structure.coordinates holds the transformed 3D points.
        width: The width of the target canvas in pixels.
        height: The height of the target canvas in pixels.
        padding_x: Horizontal padding as a fraction of the width (0 to <0.5).
        padding_y: Vertical padding as a fraction of the height (0 to <0.5).
        maintain_aspect_ratio: Whether to scale uniformly to fit while preserving
                               the structure's shape, or stretch to fill padding box.
        center_projection: Whether to center the projected structure within the canvas.
        view_direction: Optional (3,) numpy array for the view direction (camera looking along -view_direction).
                        Defaults to [0, 0, 1] if None.
        up_vector: Optional (3,) numpy array for the initial up vector.
                   Defaults to [0, 1, 0] if None.

    Returns:
        A new Structure object where the coordinates represent the projected data:
        - Column 0: X canvas coordinate (float)
        - Column 1: Y canvas coordinate (float)
        - Column 2: Depth value (float, representing scaled Z for visibility/layering)

    Raises:
        ProjectionError: If the projection process fails (e.g., mathematical error,
                         invalid parameters).
        ValueError: If the structure lacks coordinates or if vector inputs are invalid.
    """
    if not hasattr(structure, "coordinates") or structure.coordinates is None:
        raise ValueError("Structure has no coordinates to project.")

    coords_to_project = structure.coordinates
    if coords_to_project.size == 0:
        logger.warning("Structure coordinates are empty. Returning original structure.")
        # Return original structure if no coordinates to project
        return structure

    try:
        projector = OrthographicProjection()

        # Construct parameters for the projection
        param_kwargs = {
            "width": width,
            "height": height,
            "padding_x": padding_x,
            "padding_y": padding_y,
            "maintain_aspect_ratio": maintain_aspect_ratio,
            "canvas_alignment": "center" if center_projection else "top_left",
        }
        if view_direction is not None:
            param_kwargs["view_direction"] = view_direction
        if up_vector is not None:
            param_kwargs["up_vector"] = up_vector

        projection_params = OrthographicProjectionParameters(**param_kwargs)

        # Define the transformation function that performs the projection
        def projection_func(coords: np.ndarray) -> np.ndarray:
            logger.info("Performing orthographic projection internally...")
            projected = projector.project(coords, projection_params)
            logger.info("Internal projection complete.")
            return projected.astype(np.float32)  # Ensure float32

        # Use apply_vectorized_transformation to create a new structure
        # with the projected coordinates
        logger.info("Applying projection transformation to structure...")
        projected_structure = structure.apply_vectorized_transformation(projection_func)
        logger.info("Projection transformation applied.")

        return projected_structure

    except Exception as e:
        logger.error(f"Error during orthographic projection: {str(e)}", exc_info=True)
        # Re-raise specific errors or wrap in ProjectionError
        if isinstance(e, (ProjectionError, ValueError)):
            raise
        raise ProjectionError(f"Orthographic projection failed: {str(e)}") from e
