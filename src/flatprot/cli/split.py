# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Split command implementation for FlatProt CLI."""

from pathlib import Path
from typing import List, Optional
import tempfile

import cyclopts
from pydantic import BaseModel, Field, field_validator

from flatprot.core import ResidueRange, logger
from flatprot.io import (
    validate_structure_file,
    InvalidStructureError,
    GemmiStructureParser,
    OutputFileError,
    StyleParser,
)
from flatprot.transformation import TransformationError
from flatprot.utils.database import ensure_database_available
from flatprot.utils.domain_utils import (
    apply_domain_transformations_masked,
    create_domain_aware_scene,
    extract_structure_regions,
    align_regions_batch,
)
from flatprot.utils.structure_utils import (
    project_structure_orthographically,
    transform_structure_with_inertia,
)
from flatprot.utils.scene_utils import add_position_annotations_to_scene
from flatprot.renderers import SVGRenderer
from flatprot.cli.errors import CLIError, error_handler


class SplitConfig(BaseModel):
    """Configuration for split command."""

    structure_file: Path = Field(
        ..., description="Path to input structure file (PDB/CIF)"
    )
    regions: str = Field(
        ..., description="Comma-separated residue regions (e.g., 'A:1-100,A:150-250')"
    )
    output: Path = Field(
        default=Path("split_output.svg"), description="Output SVG file path"
    )
    alignment_mode: str = Field(default="family-identity", description="Alignment mode")
    layout: str = Field(default="horizontal", description="Layout arrangement")
    spacing: float = Field(
        default=100.0, description="Spacing between regions in pixels"
    )
    style: Optional[Path] = Field(default=None, description="Custom style file (TOML)")
    min_probability: float = Field(
        default=0.5, description="Minimum alignment probability"
    )
    canvas_width: int = Field(default=1000, description="Canvas width in pixels")
    canvas_height: int = Field(default=1000, description="Canvas height in pixels")
    foldseek: str = Field(default="foldseek", description="Foldseek executable path")
    dssp: Optional[Path] = Field(default=None, description="DSSP file for PDB input")
    show_positions: bool = Field(
        default=False, description="Show residue position annotations"
    )
    show_database_alignment: bool = Field(
        default=False,
        description="Enable database alignment and show family area annotations",
    )

    @field_validator("regions")
    @classmethod
    def validate_regions(cls, v: str) -> str:
        """Validate region format."""
        if not v.strip():
            raise ValueError("Regions cannot be empty")

        regions = v.split(",")
        for region in regions:
            region = region.strip()
            if not region:
                continue

            # Basic format validation
            if ":" not in region or "-" not in region:
                raise ValueError(
                    f"Invalid region format: {region}. Expected format: 'CHAIN:START-END'"
                )

            try:
                chain_part, range_part = region.split(":", 1)
                start_str, end_str = range_part.split("-", 1)
                start, end = int(start_str.strip()), int(end_str.strip())

                if start <= 0 or end <= 0 or start > end:
                    raise ValueError(f"Invalid residue numbers in region: {region}")

            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid region format: {region}. Error: {e}")

        return v

    @field_validator("alignment_mode")
    @classmethod
    def validate_alignment_mode(cls, v: str) -> str:
        """Validate alignment mode."""
        valid_modes = {"family-identity", "inertia"}
        if v not in valid_modes:
            raise ValueError(
                f"Invalid alignment mode: {v}. Must be one of {valid_modes}"
            )
        return v

    @field_validator("layout")
    @classmethod
    def validate_layout(cls, v: str) -> str:
        """Validate layout arrangement."""
        valid_layouts = {"horizontal", "vertical"}
        if v not in valid_layouts:
            raise ValueError(f"Invalid layout: {v}. Must be one of {valid_layouts}")
        return v


def parse_regions(regions_str: str) -> List[ResidueRange]:
    """Parse region string into ResidueRange objects.

    Args:
        regions_str: Comma-separated regions like "A:1-100,A:150-250,B:1-80"

    Returns:
        List of ResidueRange objects

    Raises:
        ValueError: If region format is invalid
    """
    regions = []
    for region in regions_str.split(","):
        region = region.strip()
        if not region:
            continue

        try:
            chain_part, range_part = region.split(":", 1)
            chain_id = chain_part.strip().upper()
            start_str, end_str = range_part.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())

            regions.append(ResidueRange(chain_id, start, end))

        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse region '{region}': {e}")

    if not regions:
        raise ValueError("No valid regions found")

    return regions


@error_handler
def split_command(
    structure_file: Path,
    *,
    regions: str,
    output: Path = Path("split_output.svg"),
    alignment_mode: str = "family-identity",
    layout: str = "horizontal",
    spacing: float = 100.0,
    style: Optional[Path] = None,
    min_probability: float = 0.5,
    canvas_width: int = 1000,
    canvas_height: int = 1000,
    foldseek: str = "foldseek",
    dssp: Optional[Path] = None,
    show_positions: bool = False,
    show_database_alignment: bool = False,
) -> None:
    """Split protein structure into regions and create aligned visualization.

    Extracts specified structural regions, aligns them using Foldseek, and creates
    a combined SVG visualization with the aligned regions arranged spatially.

    Args:
        structure_file: Path to input structure file (PDB/CIF)
        regions: Comma-separated residue regions (e.g., 'A:1-100,A:150-250')
        output: Output SVG file path
        alignment_mode: Alignment strategy ('family-identity' or 'inertia')
        layout: Layout arrangement ('horizontal' or 'vertical')
        spacing: Spacing between regions in pixels
        style: Custom style file (TOML format)
        min_probability: Minimum alignment probability threshold
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        foldseek: Foldseek executable path
        dssp: DSSP file for PDB input (required for PDB files)
        show_positions: Show residue position annotations
        show_database_alignment: Enable database alignment and show family area annotations

    Raises:
        CLIError: If command execution fails
    """
    try:
        # Validate configuration
        config = SplitConfig(
            structure_file=structure_file,
            regions=regions,
            output=output,
            alignment_mode=alignment_mode,
            layout=layout,
            spacing=spacing,
            style=style,
            min_probability=min_probability,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            foldseek=foldseek,
            dssp=dssp,
            show_positions=show_positions,
            show_database_alignment=show_database_alignment,
        )

        logger.info(f"Starting split command for {config.structure_file}")
        logger.info(f"Regions: {config.regions}")
        logger.info(f"Output: {config.output}")

        # Validate input files
        validate_structure_file(config.structure_file)

        if config.structure_file.suffix.lower() == ".pdb" and not config.dssp:
            raise CLIError("DSSP file required for PDB input. Use --dssp option.")

        # Parse regions
        region_ranges = parse_regions(config.regions)
        logger.info(f"Parsed {len(region_ranges)} regions")

        # Load original structure
        parser = GemmiStructureParser()
        original_structure = parser.parse_structure(
            config.structure_file, secondary_structure_file=config.dssp
        )

        if original_structure is None:
            raise CLIError("Failed to parse structure file")

        logger.info(
            f"Loaded structure: {original_structure.id} ({len(original_structure)} residues)"
        )

        # Create temporary directory for extracted regions
        with tempfile.TemporaryDirectory(prefix="flatprot_split_") as temp_dir:
            temp_path = Path(temp_dir)

            # Extract regions to temporary files
            logger.info("Extracting regions...")
            region_files = extract_structure_regions(
                config.structure_file, region_ranges, temp_path
            )
            logger.info(f"Extracted {len(region_files)} region files")

            # Always create identity-based domain transformations for consistent layout
            from flatprot.utils.domain_utils import DomainTransformation
            from flatprot.transformation import TransformationMatrix
            import numpy as np

            # Create identity transformation matrix for layout
            identity_rotation = np.eye(3)
            identity_translation = np.zeros(3)
            identity_matrix = TransformationMatrix(
                rotation=identity_rotation, translation=identity_translation
            )

            domain_transformations = []

            # Collect alignment data if database alignment is enabled (for annotations only)
            if config.show_database_alignment:
                # Ensure alignment database is available
                logger.info("Ensuring alignment database is available...")
                db_path = ensure_database_available()
                foldseek_db_path = db_path / "foldseek" / "db"

                # Align regions to get SCOP IDs and probabilities
                logger.info("Aligning regions to collect annotation data...")
                alignment_transformations = align_regions_batch(
                    region_files,
                    region_ranges,
                    foldseek_db_path,
                    config.foldseek,
                    config.min_probability,
                    config.alignment_mode,
                )
                logger.info(
                    f"Successfully aligned {len(alignment_transformations)} regions"
                )

                if not alignment_transformations:
                    raise CLIError("No successful alignments found")

                # Create domain transformations with rotation-only matrices to preserve positioning
                logger.info(
                    "Creating domain definitions with rotation-only transformations..."
                )
                for alignment_transform in alignment_transformations:
                    # Extract only rotation component, zero out translation to preserve positioning
                    rotation_only_matrix = TransformationMatrix(
                        rotation=alignment_transform.transformation_matrix.rotation,
                        translation=np.zeros(
                            3
                        ),  # Zero translation to keep original positions
                    )
                    domain_transform = DomainTransformation(
                        domain_range=alignment_transform.domain_range,
                        transformation_matrix=rotation_only_matrix,  # Use rotation-only transformation
                        domain_id=alignment_transform.domain_id,
                        scop_id=alignment_transform.scop_id,  # Keep SCOP ID for annotations
                        alignment_probability=alignment_transform.alignment_probability,  # Keep probability for annotations
                    )
                    domain_transformations.append(domain_transform)
            else:
                # No alignment - create simple domain transformations for layout only
                logger.info(
                    "Database alignment disabled - creating layout-only domain definitions..."
                )
                for i, region_range in enumerate(region_ranges):
                    domain_id = f"{region_range.chain_id}:{region_range.start}-{region_range.end}"
                    domain_transform = DomainTransformation(
                        domain_range=region_range,
                        transformation_matrix=identity_matrix,
                        domain_id=domain_id,
                        scop_id=None,  # No SCOP ID when database alignment is disabled
                        alignment_probability=None,  # No alignment probability when alignment is disabled
                    )
                    domain_transformations.append(domain_transform)

            logger.info(f"Created {len(domain_transformations)} domain definitions")

            # Apply domain transformations if database alignment is enabled
            if (
                config.show_database_alignment
                and config.alignment_mode == "family-identity"
            ):
                logger.info(
                    "Applying domain transformations for aligned orientations..."
                )
                transformed_structure = apply_domain_transformations_masked(
                    original_structure, domain_transformations
                )
            else:
                # Use original structure for consistent positioning when alignment disabled or inertia mode
                transformed_structure = original_structure

            # Apply inertia transformation for overall orientation
            logger.info("Applying inertia transformation...")
            inertia_transformed = transform_structure_with_inertia(
                transformed_structure
            )

            # Project to 2D
            logger.info(
                f"Projecting to 2D ({config.canvas_width}x{config.canvas_height})..."
            )
            projected_structure = project_structure_orthographically(
                inertia_transformed, config.canvas_width, config.canvas_height
            )

            # Load custom styles if provided
            logger.info("Loading styles...")
            default_styles = None
            if config.style:
                try:
                    style_parser = StyleParser(config.style)
                    style_data = style_parser.parse()
                    default_styles = style_data
                    logger.info(f"Loaded custom styles from {config.style}")
                except Exception as e:
                    logger.warning(f"Failed to load style file {config.style}: {e}")

            # Extract SCOP IDs and alignment probabilities from domain transformations
            logger.info(
                "Extracting SCOP IDs and alignment probabilities from alignment results..."
            )
            domain_scop_ids = {}
            domain_alignment_probabilities = {}
            for dt in domain_transformations:
                if dt.domain_id and dt.scop_id:
                    domain_scop_ids[dt.domain_id] = dt.scop_id
                    logger.info(f"Mapped domain {dt.domain_id} -> SCOP ID {dt.scop_id}")

                if dt.domain_id and dt.alignment_probability is not None:
                    domain_alignment_probabilities[
                        dt.domain_id
                    ] = dt.alignment_probability
                    logger.info(
                        f"Mapped domain {dt.domain_id} -> Alignment probability {dt.alignment_probability:.3f}"
                    )

            # Create domain-aware scene
            logger.info(f"Creating scene with {config.layout} layout...")

            # Control family annotations based on flag
            scop_ids_to_use = (
                domain_scop_ids if config.show_database_alignment else None
            )
            if config.show_database_alignment:
                logger.info("Database alignment and family annotations enabled")
            else:
                logger.info("Database alignment and family annotations disabled")

            scene = create_domain_aware_scene(
                projected_structure=projected_structure,
                domain_definitions=domain_transformations,
                spacing=config.spacing,
                arrangement=config.layout,
                default_styles=default_styles,  # Use loaded styles
                domain_scop_ids=scop_ids_to_use,  # Only pass SCOP IDs if enabled
                domain_alignment_probabilities=domain_alignment_probabilities
                if config.show_database_alignment
                else None,
            )

            # Add position annotations if requested
            if config.show_positions:
                logger.info("Adding position annotations...")
                position_style = None
                if default_styles and "position_annotation" in default_styles:
                    position_style = default_styles["position_annotation"]
                add_position_annotations_to_scene(scene, position_style)
                logger.info("Position annotations added")

            # Render SVG
            logger.info(f"Rendering SVG to {config.output}...")

            # Adjust canvas size for layout
            render_width = config.canvas_width
            render_height = config.canvas_height

            if config.layout == "horizontal":
                render_width += int(config.spacing * len(domain_transformations))
            elif config.layout == "vertical":
                render_height += int(config.spacing * len(domain_transformations))

            renderer = SVGRenderer(scene, render_width, render_height)
            renderer.save_svg(config.output)

            logger.info("Split command completed successfully")

    except (
        ValueError,
        InvalidStructureError,
        TransformationError,
        OutputFileError,
    ) as e:
        raise CLIError(f"Split command failed: {e}")
    except Exception as e:
        logger.exception("Unexpected error in split command")
        raise CLIError(f"Unexpected error: {e}")


# Register with cyclopts
app = cyclopts.App()
app.command(split_command)

if __name__ == "__main__":
    app()
