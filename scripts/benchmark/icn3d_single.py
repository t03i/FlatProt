#!/usr/bin/env -S uv --quiet run --script

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "selenium",
#   "webdriver-manager"
# ]
# ///

# Copyright 2025 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from Bio.PDB import PDBParser, MMCIFParser
except ImportError as e:
    print(f"Error: Required package not available: {e}", file=sys.stderr)
    print("Install with: uv add matplotlib biopython numpy", file=sys.stderr)
    sys.exit(1)


def create_2d_protein_diagram(structure, output_file):
    """Create a simple 2D protein structure diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract coordinates from all chains
    residue_positions = []
    residue_types = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):  # Alpha carbon
                    ca_atom = residue["CA"]
                    coord = ca_atom.get_coord()
                    residue_positions.append(coord)
                    residue_types.append(residue.get_resname())

    if not residue_positions:
        raise ValueError("No CA atoms found in structure")

    positions = np.array(residue_positions)

    # Project 3D coordinates to 2D (simple PCA-like projection)
    # Use first two principal components for visualization
    mean_pos = np.mean(positions, axis=0)
    centered_pos = positions - mean_pos

    # Use XY projection (ignoring Z for simplicity)
    x_coords = centered_pos[:, 0]
    y_coords = centered_pos[:, 1]

    # Plot the protein backbone as connected line
    ax.plot(x_coords, y_coords, "b-", linewidth=2, alpha=0.7, label="Backbone")

    # Plot CA atoms as dots
    ax.scatter(x_coords, y_coords, c="red", s=20, alpha=0.8, label="C-alpha atoms")

    # Add some styling
    ax.set_xlabel("X Coordinate (Å)")
    ax.set_ylabel("Y Coordinate (Å)")
    ax.set_title(f"2D Protein Structure Diagram\n({len(residue_positions)} residues)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Save the plot
    plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(
    structure_file: Path,
    output_file: Path,
) -> None:
    """Load a structure and create a 2D diagram using BioPython and matplotlib.

    This script creates simple 2D protein structure diagrams from PDB/CIF files.

    Parameters
    ----------
    structure_file
        Path to the input structure file (e.g., CIF, PDB). Must exist.
    output_file
        Path to save the output image (PNG format).
    """
    try:
        # Determine file type and parse structure
        file_extension = structure_file.suffix.lower()

        if file_extension in [".cif", ".mmcif"]:
            parser = MMCIFParser(QUIET=True)
        elif file_extension == ".pdb":
            parser = PDBParser(QUIET=True)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        structure = parser.get_structure("protein", str(structure_file))

        # Create 2D diagram
        create_2d_protein_diagram(structure, output_file)

        print(f"Successfully created 2D diagram: {output_file}")

    except Exception as e:
        print(f"Error processing structure {structure_file}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a structure and create a 2D diagram using BioPython and matplotlib."
    )
    parser.add_argument(
        "structure_file",
        type=Path,
        help="Path to the input structure file (e.g., CIF, PDB).",
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to save the output PNG image."
    )

    try:
        args = parser.parse_args()
        if not args.structure_file.exists():
            print(
                f"Error: structure file not found: {args.structure_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        main(args.structure_file, args.output_file)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
