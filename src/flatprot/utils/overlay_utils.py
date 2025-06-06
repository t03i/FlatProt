"""Utility functions for creating protein structure overlays."""

import collections
import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import drawsvg as draw

from flatprot.core import logger
from flatprot.io import GemmiStructureParser, validate_structure_file, StyleParser
from flatprot.renderers import SVGRenderer
from flatprot.scene import BaseStructureStyle
from flatprot.utils.scene_utils import create_scene_from_structure
from flatprot.utils.structure_utils import (
    project_structure_orthographically,
    transform_structure_with_inertia,
)


@dataclass
class OverlayConfig:
    """Configuration for overlay generation."""

    alignment_mode: str = "family-identity"  # or "inertia"
    target_family: Optional[str] = None  # SCOP ID for fixed family
    min_probability: float = 0.5
    clustering_enabled: bool = True  # Auto-cluster similar structures
    opacity_scaling: bool = True  # Scale opacity by cluster size
    canvas_width: int = 1000
    canvas_height: int = 1000
    style_file: Optional[Path] = None
    output_format: str = "png"
    dpi: int = 300
    quiet: bool = False


def create_overlay(
    input_files: List[Path], output_path: Path, config: OverlayConfig
) -> Path:
    """Create overlay visualization from multiple protein structures."""

    if not config.quiet:
        logger.info(f"Creating overlay from {len(input_files)} structures")

    # Step 1: Optional clustering to reduce visual clutter
    if config.clustering_enabled and len(input_files) > 5:
        representatives = cluster_and_select_representatives(input_files, config)
        if not config.quiet:
            logger.info(
                f"Clustering reduced {len(input_files)} structures to {len(representatives)} representatives"
            )
    else:
        representatives = [(f, 1.0) for f in input_files]  # (file, opacity)

    # Step 2: Generate individual drawings with alignment
    if not config.quiet:
        logger.info("Generating aligned projections...")

    drawings_with_opacity = []
    for file_path, base_opacity in representatives:
        try:
            drawing = generate_aligned_drawing(file_path, config)
            drawings_with_opacity.append((drawing, base_opacity, file_path.stem))
        except Exception as e:
            if not config.quiet:
                logger.warning(f"Failed to generate drawing for {file_path.name}: {e}")
            continue

    if not drawings_with_opacity:
        raise RuntimeError("No drawings could be generated successfully")

    # Step 3: Combine drawings using drawsvg directly
    if not config.quiet:
        logger.info("Combining drawings into overlay...")

    combined_drawing = combine_drawings(drawings_with_opacity, config)

    # Step 4: Export in target format using drawsvg's built-in methods
    if not config.quiet:
        logger.info(f"Exporting to {config.output_format.upper()}...")

    if config.output_format == "png":
        combined_drawing.save_png(str(output_path), dpi=config.dpi)
    elif config.output_format == "pdf":
        combined_drawing.save_pdf(str(output_path))
    elif config.output_format == "svg":
        combined_drawing.save_svg(str(output_path))
    else:
        raise ValueError(f"Unsupported output format: {config.output_format}")

    return output_path


def generate_aligned_drawing(file_path: Path, config: OverlayConfig) -> draw.Drawing:
    """Generate aligned drawing for a single structure using existing FlatProt pipeline."""

    # Parse structure using existing parser
    validate_structure_file(file_path)
    parser = GemmiStructureParser()
    structure = parser.parse_structure(file_path)

    # Apply alignment based on mode
    if config.alignment_mode == "family-identity" and config.target_family:
        # Use existing alignment workflow
        # db_path = ensure_database_available()
        # alignment_db = AlignmentDatabase(db_path / "alignments.h5")
        # foldseek_db_path = db_path / "foldseek" / "db"

        # Align to specific family
        # alignment_result = align_structure_database(
        #     file_path, foldseek_db_path, "foldseek", config.min_probability
        # )

        # Get transformation matrix
        # matrix_result = get_aligned_rotation_database(alignment_result, alignment_db)
        # transformation_matrix = matrix_result[0]  # TODO: implement matrix application

        # Apply transformation (this would need proper implementation in the transformation module)
        # For now, we'll use inertia transformation as fallback
        transformed_structure = transform_structure_with_inertia(structure)
    else:
        # Use inertia-based alignment
        transformed_structure = transform_structure_with_inertia(structure)

    # Project to 2D using existing utilities
    projected_structure = project_structure_orthographically(
        transformed_structure, config.canvas_width, config.canvas_height
    )

    # Load styles if provided
    styles_dict = load_styles(config.style_file) if config.style_file else None

    # Create scene using existing utilities
    scene = create_scene_from_structure(projected_structure, styles_dict)

    # Render using existing SVGRenderer and return Drawing object
    renderer = SVGRenderer(scene, config.canvas_width, config.canvas_height)
    return renderer.render()  # This returns the drawsvg.Drawing object


def cluster_and_select_representatives(
    files: List[Path], config: OverlayConfig
) -> List[Tuple[Path, float]]:
    """Cluster structures and select representatives with opacity based on cluster size."""

    if not config.quiet:
        logger.info("Clustering structures with Foldseek...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create input directory with structure files
        structures_dir = temp_path / "structures"
        structures_dir.mkdir()

        # Copy files to temp directory (Foldseek needs them in one directory)
        for file_path in files:
            temp_file = structures_dir / file_path.name
            temp_file.write_bytes(file_path.read_bytes())

        # Run Foldseek clustering
        cluster_output_prefix = temp_path / "cluster"
        clustering_tmp_dir = temp_path / "clustering_tmp"
        clustering_tmp_dir.mkdir()

        cluster_cmd = [
            "foldseek",
            "easy-cluster",
            str(structures_dir),
            str(cluster_output_prefix),
            str(clustering_tmp_dir),
            "--min-seq-id",
            "0.5",
            "-c",
            "0.9",
            "-v",
            "0",
        ]

        try:
            subprocess.run(cluster_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if not config.quiet:
                logger.warning(f"Clustering failed, using all structures: {e}")
            return [(f, 1.0) for f in files]

        # Parse cluster results
        cluster_file = Path(f"{cluster_output_prefix}_cluster.tsv")
        if not cluster_file.exists():
            if not config.quiet:
                logger.warning("Cluster file not found, using all structures")
            return [(f, 1.0) for f in files]

        clusters = collections.defaultdict(list)
        with open(cluster_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) == 2:
                    representative, member = row
                    rep_fn = f"{Path(representative).stem}.cif"
                    mem_fn = f"{Path(member).stem}.cif"
                    clusters[rep_fn].append(mem_fn)

        # Select representatives from clusters with multiple members
        large_clusters = {
            rep: members for rep, members in clusters.items() if len(members) > 1
        }

        if not large_clusters:
            # No clustering benefit, use all structures
            return [(f, 1.0) for f in files]

        # Calculate opacities based on cluster sizes
        cluster_counts = {rep: len(members) for rep, members in large_clusters.items()}
        opacities = calculate_opacity(cluster_counts)

        # Map back to original file paths
        representatives = []
        for rep_name, opacity in opacities.items():
            for file_path in files:
                if file_path.stem + ".cif" == rep_name:
                    representatives.append((file_path, opacity))
                    break

        return representatives


def calculate_opacity(
    cluster_counts: Dict[str, int], min_opacity: float = 0.1, max_opacity: float = 1.0
) -> Dict[str, float]:
    """Calculate opacity for each representative based on cluster size."""
    if not cluster_counts:
        return {}

    counts = list(cluster_counts.values())
    min_count = min(counts)
    max_count = max(counts)

    opacities = {}
    for representative, count in cluster_counts.items():
        if max_count == min_count:
            normalized_count = 1.0
        else:
            normalized_count = (count - min_count) / (max_count - min_count)
        opacity = min_opacity + normalized_count * (max_opacity - min_opacity)
        opacities[representative] = opacity

    return opacities


def combine_drawings(
    drawings_with_opacity: List[Tuple[draw.Drawing, float, str]], config: OverlayConfig
) -> draw.Drawing:
    """Combine multiple drawings into a single overlay with opacity using drawsvg."""

    # Create combined drawing with same dimensions
    combined_drawing = draw.Drawing(config.canvas_width, config.canvas_height)

    # Set viewbox to match canvas
    combined_drawing.view_box = (0, 0, config.canvas_width, config.canvas_height)

    # Add each drawing as a group with opacity
    for i, (drawing, opacity, name) in enumerate(drawings_with_opacity):
        # Create group with opacity
        group = draw.Group(opacity=opacity, id=f"structure_{i}_{name}")

        # Copy all elements except background and defs
        for element in drawing.elements:
            # Skip background rectangles and defs elements
            if (hasattr(element, "class_") and element.class_ == "background") or (
                hasattr(element, "tag") and element.tag == "defs"
            ):
                continue

            # Add element to group
            group.append(element)

        # Add group to combined drawing
        combined_drawing.append(group)

    return combined_drawing


def load_styles(style_file: Path) -> Optional[Dict[str, BaseStructureStyle]]:
    """Load style configuration from TOML file using existing StyleParser."""
    try:
        parser = StyleParser(style_file)
        return parser.get_element_styles()
    except Exception as e:
        logger.warning(f"Failed to load style file {style_file}: {e}")
        return None
