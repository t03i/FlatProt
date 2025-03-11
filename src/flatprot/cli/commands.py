# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for FlatProt visualization tool."""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
import numpy as np

from flatprot.cli.errors import (
    FlatProtCLIError,
    FileNotFoundError,
    InvalidStructureError,
    TransformationError,
)
from flatprot.io import GemmiStructureParser, MatrixLoader
from flatprot.core.components import Structure
from flatprot.transformation import (
    InertiaTransformer,
    MatrixTransformer,
    MatrixTransformParameters,
    TransformationMatrix,
)
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters

console = Console()


def validate_structure_file(path: Path) -> None:
    """Validate that the file exists and is a valid PDB or CIF format.

    Args:
        path: Path to the structure file

    Raises:
        FileNotFoundError: If the file does not exist
        InvalidStructureError: If the file is not a valid PDB or CIF format
    """
    # Check file existence
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Check file extension
    suffix = path.suffix.lower()
    if suffix not in [".pdb", ".cif", ".mmcif", ".ent"]:
        raise InvalidStructureError(
            str(path),
            "PDB or CIF",
            "File does not have a recognized structure file extension (.pdb, .cif, .mmcif, .ent)",
        )

    # Basic content validation
    try:
        with open(path, "r") as f:
            content = f.read(1000)  # Read first 1000 bytes for quick check

            # Basic check for PDB format
            if suffix in [".pdb", ".ent"]:
                if not (
                    "ATOM" in content or "HETATM" in content or "HEADER" in content
                ):
                    raise InvalidStructureError(
                        str(path),
                        "PDB",
                        "File does not contain required PDB records (ATOM, HETATM, or HEADER)",
                    )

            # Basic check for mmCIF format
            if suffix in [".cif", ".mmcif"]:
                if not (
                    "_atom_site." in content or "loop_" in content or "data_" in content
                ):
                    raise InvalidStructureError(
                        str(path),
                        "CIF",
                        "File does not contain required CIF categories (_atom_site, loop_, or data_)",
                    )
    except UnicodeDecodeError:
        raise InvalidStructureError(
            str(path),
            "PDB or CIF",
            "File contains invalid characters and is not a valid text file",
        )


def get_coordinate_manager(
    structure: Structure, matrix_path: Optional[Path] = None
) -> CoordinateManager:
    """Apply transformation to a protein structure and create a coordinate manager.

    Args:
        structure: The protein structure to transform
        matrix_path: Path to a numpy matrix file for custom transformation.
                    If None, uses InertiaTransformer.

    Returns:
        CoordinateManager preloaded with original, transformed, and projected coordinates

    Raises:
        TransformationError: If there's an error during transformation
    """
    try:
        # Check if structure has coordinates
        if not hasattr(structure, "coordinates") or structure.coordinates is None:
            console.print(
                "[yellow]Warning: Structure has no coordinates, skipping transformation[/yellow]"
            )
            # Create empty coordinate manager
            return CoordinateManager()

        # Create coordinate manager
        coordinate_manager = CoordinateManager()

        # Add original coordinates
        coordinate_manager.add(
            0,
            len(structure.coordinates),
            structure.coordinates,
            CoordinateType.COORDINATES,
        )

        # Set up transformer based on input
        transformer = None
        transform_parameters = None

        if matrix_path:
            # Load custom transformation matrix
            try:
                matrix_loader = MatrixLoader(matrix_path)
                transformation_matrix = matrix_loader.load()
                transformer = MatrixTransformer()
                transform_parameters = MatrixTransformParameters(
                    matrix=transformation_matrix
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to load matrix from {matrix_path}: {str(e)}[/yellow]"
                )
                console.print(
                    "[yellow]Falling back to inertia-based transformation[/yellow]"
                )
                matrix_path = None  # Fall back to inertia transformer

        has_valid_residues = (
            hasattr(structure, "residues")
            and structure.residues is not None
            and len(structure.residues) > 0
        )

        if not matrix_path:
            # Use inertia-based transformation
            if (
                hasattr(structure, "coordinates")
                and structure.coordinates is not None
                and has_valid_residues
            ):
                transformer = InertiaTransformer()
                transform_parameters = (
                    None  # InertiaTransformer doesn't need parameters
                )
            else:
                # If structure doesn't have proper attributes, use identity transformation
                console.print(
                    "[yellow]Warning: Structure lacks required properties for inertia transformation, using identity matrix[/yellow]"
                )
                rotation = np.eye(3)
                translation = np.zeros(3)
                identity_matrix = TransformationMatrix(
                    rotation=rotation, translation=translation
                )
                transformer = MatrixTransformer()
                transform_parameters = MatrixTransformParameters(matrix=identity_matrix)

        # Apply transformation
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

        # Set up projector with default parameters
        projector = OrthographicProjector()
        projection_parameters = OrthographicProjectionParameters(
            width=800,  # Default width
            height=600,  # Default height
            padding_x=0.05,  # 5% padding (values must be between 0 and 1)
            padding_y=0.05,  # 5% padding (values must be between 0 and 1)
            maintain_aspect_ratio=True,  # Default setting
        )

        # Project the coordinates
        canvas_coords, depth = projector.project(
            transformed_coords, projection_parameters
        )

        # Add projected coordinates and depth information
        coordinate_manager.add(
            0, len(structure.coordinates), canvas_coords, CoordinateType.CANVAS
        )
        coordinate_manager.add(
            0, len(structure.coordinates), depth, CoordinateType.DEPTH
        )

        return coordinate_manager

    except Exception as e:
        console.print(f"[red]Error applying transformation: {str(e)}[/red]")
        raise TransformationError(f"Failed to apply transformation: {str(e)}")


def main(
    structure_file: Path,
    output_file: Path,
    matrix: Optional[Path] = None,
    annotations: Optional[Path] = None,
    style: Optional[Path] = None,
) -> int:
    """Generate a 2D visualization of a protein structure.

    Args:
        structure_file: Path to a PDB/CIF structure file to visualize.
        output_file: Path for the output SVG file containing the visualization.
        matrix: Path to a numpy matrix file for custom transformation. If not provided, inertia-based projection will be used.
        annotations: Path to a TOML file containing annotation specifications.
        style: Path to a TOML file containing style specifications.

    Examples:
        flatprot structure.pdb output.svg
        flatprot structure.cif output.svg --annotations annotations.toml --style style.toml
        flatprot structure.pdb output.svg --matrix custom_matrix.npy
    """
    try:
        # Validate the structure file
        validate_structure_file(structure_file)

        # Ensure the output directory exists
        output_dir = output_file.parent
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)

        # Verify optional files if specified
        if matrix and not matrix.exists():
            raise FileNotFoundError(str(matrix))
        if annotations and not annotations.exists():
            raise FileNotFoundError(str(annotations))
        if style and not style.exists():
            raise FileNotFoundError(str(style))

        # Load structure
        parser = GemmiStructureParser()
        structure = parser.parse_structure(structure_file)

        # Apply transformation and create coordinate manager
        # ruff: noqa: F841
        coordinate_manager = get_coordinate_manager(structure, matrix)

        # For now, just print a message to indicate successful parsing
        console.print("[bold green]Successfully processed structure:[/bold green]")
        console.print(f"  Structure file: {str(structure_file)}")
        console.print(f"  Output file: {str(output_file)}")
        console.print(
            f"  Transformation: {'Custom matrix' if matrix else 'Inertia-based'}"
        )
        if matrix:
            console.print(f"  Matrix file: {str(matrix)}")
        if annotations:
            console.print(f"  Annotations file: {str(annotations)}")
        if style:
            console.print(f"  Style file: {str(style)}")

        return 0

    except FlatProtCLIError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
        return 1
