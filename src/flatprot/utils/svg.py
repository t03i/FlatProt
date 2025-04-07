# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""SVG utilities for FlatProt."""

from typing import Optional
from pathlib import Path
import os

from flatprot.core import CoordinateManager, ResidueRange
from flatprot.core.components import Structure
from flatprot.drawing import Canvas
from flatprot.scene import Scene
from flatprot.style import StyleManager

from .scene import process_structure_chain, process_annotations

from .logger import logger


def generate_svg(
    structure: Structure,
    coordinate_manager: CoordinateManager,
    style_manager: StyleManager,
    annotations_path: Optional[Path] = None,
) -> str:
    """Generate SVG content from a structure and coordinate manager.

    Args:
        structure: The protein structure
        coordinate_manager: Coordinate manager with transformed and projected coordinates
        style_manager: Style manager with style information
        annotations_path: Optional path to annotations file

    Returns:
        SVG content as a string
    """
    # Create scene from structure
    scene = Scene()

    # Process each chain
    for chain in structure:
        # Create ResidueRange for this chain
        chain_range = ResidueRange(
            chain_id=chain.id,
            start=chain.residues[0].number if chain.residues else 0,
            end=chain.residues[-1].number if chain.residues else 0,
        )

        process_structure_chain(
            chain=chain,
            residue_range=chain_range,
            coordinate_manager=coordinate_manager,
            style_manager=style_manager,
            scene=scene,
        )

    # Handle annotations if provided
    if annotations_path:
        process_annotations(annotations_path, scene, style_manager)

    # Render scene to SVG using Canvas
    canvas = Canvas(scene, style_manager)
    drawing = canvas.render()

    return drawing.as_svg()


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

        logger.info(f"[bold]SVG saved to {output_path}[/bold]")
    except Exception as e:
        logger.error(f"Error saving SVG to {output_path}: {str(e)}")
        raise IOError(f"Failed to save SVG: {str(e)}")
