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
from flatprot.io import GemmiStructureParser, MatrixLoader, StyleParser
from flatprot.core.components import Structure
from flatprot.transformation import (
    InertiaTransformer,
    MatrixTransformer,
    MatrixTransformParameters,
    TransformationMatrix,
    InertiaTransformParameters,
)
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.projection import OrthographicProjector, OrthographicProjectionParameters
from flatprot.style import StyleManager
from flatprot.scene import Scene, SceneGroup
from flatprot.scene.structure import secondary_structure_to_scene_element
from flatprot.drawing.canvas import Canvas
from flatprot.io import AnnotationParser

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
                # Create proper parameters with structure residues
                transform_parameters = InertiaTransformParameters(
                    residues=structure.residues
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


def generate_svg(
    structure: Structure,
    coordinate_manager: CoordinateManager,
    annotations_path: Optional[Path] = None,
    style_path: Optional[Path] = None,
) -> str:
    """Generate SVG content from a structure and coordinate manager.

    Args:
        structure: The protein structure
        coordinate_manager: Coordinate manager with transformed and projected coordinates
        annotations_path: Optional path to annotations file
        style_path: Optional path to style file

    Returns:
        SVG content as a string
    """
    # Create style manager - either from file or default
    if style_path:
        style_parser = StyleParser(file_path=style_path)
        style_manager = style_parser.get_styles()
        console.print(style_manager, style_parser)
    else:
        style_manager = StyleManager.create_default()
        console.print(style_manager)

    # Create scene from structure
    scene = Scene()

    # Process each chain
    offset = 0
    for chain in structure:
        elements_with_z = []  # Reset for each chain

        for i, element in enumerate(chain.secondary_structure):
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
        chain_group = SceneGroup(id=f"chain_{chain.id}")
        chain_group.metadata["chain_id"] = chain.id

        # Add sorted elements to the chain group
        for element, _ in elements_with_z:
            chain_group.add_element(element)

        # Add chain group to scene
        scene.add_element(chain_group)

        # Update offset for next chain
        offset += chain.num_residues

    # Handle annotations if provided
    if annotations_path:
        annotation_parser = AnnotationParser(file_path=annotations_path, scene=scene)
        annotations = annotation_parser.parse()
        for annotation in annotations:
            annotation.apply(scene, style_manager)

    # Render scene to SVG using Canvas
    canvas = Canvas(scene, style_manager)
    drawing = canvas.render()

    # Convert drawing to SVG string (using drawsvg's functionality)
    svg_content = drawing.as_svg()

    return svg_content


def save_svg(svg_content: str, output_path: Path) -> None:
    """Save SVG content to a file, creating directories if needed.

    Args:
        svg_content: SVG content as a string
        output_path: Path to save the SVG file

    Raises:
        IOError: If the file cannot be saved
    """
    try:
        # Ensure output directory exists
        output_dir = output_path.parent
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)

        # Write SVG content to file
        with open(output_path, "w") as f:
            f.write(svg_content)

        console.print(f"[green]SVG saved to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving SVG to {output_path}: {str(e)}[/red]")
        raise IOError(f"Failed to save SVG: {str(e)}")


def main(
    structure: Path,
    output: Path = None,
    matrix: Path = None,
    style: Path = None,
    annotations: Path = None,
):
    """Generate a flat projection of a protein structure.

    Args:
        structure (Path): Path to the structure file (PDB or similar).
        output (Path, optional): Path to save the SVG output.
            If not provided, the SVG is printed to stdout.
        matrix (Path, optional): Path to a custom transformation matrix.
            If not provided, a default inertia transformation is used.
        style (Path, optional): Path to a custom style file in TOML format.
            If not provided, the default styles are used.
        annotations (Path, optional): Path to a TOML file with annotation definitions.
            The annotation file can define point, line, and area annotations to highlight
            specific structural features. Examples:
            - Point annotations mark single residues with symbols
            - Line annotations connect residues with lines
            - Area annotations highlight regions of the structure
            See examples/annotations.toml for a reference annotation file.

    Examples:
        flatprot structure.pdb output.svg
        flatprot structure.cif output.svg --annotations annotations.toml --style style.toml
        flatprot structure.pdb output.svg --matrix custom_matrix.npy

    Returns:
        int: 0 for success, 1 for errors.
    """
    try:
        # Validate the structure file
        validate_structure_file(structure)

        # Ensure the output directory exists
        output_dir = output.parent
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)

        # Verify optional files if specified
        if matrix and not matrix.exists():
            raise FileNotFoundError(str(matrix))
        if style and not style.exists():
            raise FileNotFoundError(str(style))
        if annotations and not annotations.exists():
            raise FileNotFoundError(str(annotations))

        # Load structure
        parser = GemmiStructureParser()
        structure = parser.parse_structure(structure)

        # Apply transformation and create coordinate manager
        coordinate_manager = get_coordinate_manager(structure, matrix)

        # Generate SVG visualization
        svg_content = generate_svg(structure, coordinate_manager, annotations, style)

        # Save SVG to output file
        save_svg(svg_content, output)

        # Print success message
        console.print("[bold green]Successfully processed structure:[/bold green]")
        console.print(f"  Structure file: {str(structure)}")
        console.print(f"  Output file: {str(output)}")
        console.print(
            f"  Transformation: {'Custom matrix' if matrix else 'Inertia-based'}"
        )
        if matrix:
            console.print(f"  Matrix file: {str(matrix)}")
        if style:
            console.print(f"  Style file: {str(style)}")
        if annotations:
            console.print(f"  Annotations file: {str(annotations)}")

        return 0

    except FlatProtCLIError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
        return 1
