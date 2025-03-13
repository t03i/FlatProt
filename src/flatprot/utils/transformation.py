# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Transformation utilities for FlatProt."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from rich.console import Console

from flatprot.core.components import Structure
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.transformation import (
    InertiaTransformer,
    MatrixTransformer,
    MatrixTransformParameters,
    TransformationMatrix,
    InertiaTransformParameters,
    BaseTransformer,
)
from flatprot.io import MatrixLoader
from flatprot.cli.errors import TransformationError

console = Console()


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
            console.print(
                f"[yellow]Warning: Failed to load matrix from {matrix_path}: {str(e)}[/yellow]"
            )
            console.print(
                "[yellow]Falling back to inertia-based transformation[/yellow]"
            )

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
    console.print(
        "[yellow]Warning: Structure lacks required properties for inertia transformation, using identity matrix[/yellow]"
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
            console.print(
                "[yellow]Warning: Structure has no coordinates, skipping transformation[/yellow]"
            )
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
        console.print(f"[red]Error applying transformation: {str(e)}[/red]")
        raise TransformationError(f"Failed to apply transformation: {str(e)}")
