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
from flatprot.renderers import SVGRenderer
from flatprot.cli.errors import CLIError, error_handler
from flatprot.scene import (
    PositionAnnotation,
    PositionType,
    HelixSceneElement,
    SheetSceneElement,
)
from flatprot.scene.annotation.position import PositionAnnotationStyle
from flatprot.core import ResidueCoordinate


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
    show_positions: str = Field(
        default="minimal",
        description="Position annotation level: 'none', 'minimal', 'major', 'full'",
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

    @field_validator("show_positions")
    @classmethod
    def validate_show_positions(cls, v: str) -> str:
        """Validate position annotation level."""
        valid_levels = {"none", "minimal", "major", "full"}
        if v not in valid_levels:
            raise ValueError(
                f"Invalid position annotation level: {v}. Must be one of {valid_levels}"
            )
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


def add_domain_aware_position_annotations(
    scene, domain_transformations, style=None, annotation_level="major"
):
    """Add cleaner position annotations to domain groups in a domain-aware scene.

    This function creates a cleaner annotation approach for domain-separated visualizations:
    - 'none': No position annotations
    - 'minimal': Only N/C terminus per domain
    - 'major': N/C terminus + major secondary structures (≥3 residues)
    - 'full': All position annotations (original behavior)

    Args:
        scene: The Scene object with domain groups
        domain_transformations: List of DomainTransformation objects defining domains
        style: Optional PositionAnnotationStyle
        annotation_level: Level of annotation detail ('none', 'minimal', 'major', 'full')
    """
    if style is None:
        style = PositionAnnotationStyle()

    # Return early if no annotations requested
    if annotation_level == "none":
        logger.info("Position annotations disabled")
        return

    # Get all structure elements organized by domain group
    try:
        all_structure_elements = scene.get_sequential_structure_elements()
    except Exception as e:
        logger.error(f"Failed to get sequential structure elements: {e}")
        return

    if not all_structure_elements:
        logger.warning("No structure elements found for position annotations")
        return

    # Find which group each structure element belongs to
    element_to_group = {}
    for element in all_structure_elements:
        if element.parent:
            element_to_group[element.id] = element.parent.id

    # Group elements by their parent domain group
    elements_by_group = {}
    for element in all_structure_elements:
        group_id = element_to_group.get(element.id)
        if group_id:
            if group_id not in elements_by_group:
                elements_by_group[group_id] = []
            elements_by_group[group_id].append(element)

    # Add cleaner position annotations for each domain group
    for group_id, group_elements in elements_by_group.items():
        if not group_elements:
            continue

        # Sort elements by sequence to find first and last
        group_elements.sort(
            key=lambda e: (
                e.residue_range_set.ranges[0].chain_id
                if e.residue_range_set.ranges
                else "Z",
                e.residue_range_set.ranges[0].start
                if e.residue_range_set.ranges
                else 99999,
            )
        )

        # Add N-terminus annotation for this domain (always show)
        first_element = group_elements[0]
        n_terminus_id = f"pos_n_terminus_{group_id}_{first_element.id}"

        try:
            n_terminus = PositionAnnotation(
                id=n_terminus_id,
                target=first_element.residue_range_set,
                position_type=PositionType.N_TERMINUS,
                text="N",
                style=style,
            )
            scene.add_element(n_terminus, parent_id=group_id)
            logger.debug(
                f"Added N-terminus annotation to group {group_id}: {n_terminus_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to add N-terminus annotation to group {group_id}: {e}"
            )

        # Add C-terminus annotation for this domain (always show)
        last_element = group_elements[-1]
        c_terminus_id = f"pos_c_terminus_{group_id}_{last_element.id}"

        try:
            c_terminus = PositionAnnotation(
                id=c_terminus_id,
                target=last_element.residue_range_set,
                position_type=PositionType.C_TERMINUS,
                text="C",
                style=style,
            )
            scene.add_element(c_terminus, parent_id=group_id)
            logger.debug(
                f"Added C-terminus annotation to group {group_id}: {c_terminus_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to add C-terminus annotation to group {group_id}: {e}"
            )

        # Add residue number annotations based on annotation level
        if annotation_level in ["major", "full"] and style.show_residue_numbers:
            for element in group_elements:
                # Only add residue numbers for helices and sheets, not coils
                if not isinstance(element, (HelixSceneElement, SheetSceneElement)):
                    continue

                if not element.residue_range_set.ranges:
                    logger.warning(f"Element {element.id} has no residue ranges")
                    continue

                # Get the first range (assuming single range per element)
                residue_range = element.residue_range_set.ranges[0]

                # Apply filtering based on annotation level
                structure_length = residue_range.end - residue_range.start + 1

                if annotation_level == "major" and structure_length < 3:
                    # Skip short structures in 'major' mode to reduce clutter
                    logger.debug(
                        f"Skipping short structure {element.id} ({structure_length} residues)"
                    )
                    continue
                elif annotation_level == "full":
                    # Include all structures in 'full' mode (original behavior)
                    pass

                # Add start position annotation
                start_id = f"pos_start_{element.id}"
                try:
                    # Create specific coordinate for start position
                    start_coord = ResidueCoordinate(
                        residue_range.chain_id,
                        residue_range.start,
                        None,  # residue_type not needed
                        0,  # coordinate_index
                    )
                    start_annotation = PositionAnnotation(
                        id=start_id,
                        target=start_coord,
                        position_type=PositionType.RESIDUE_NUMBER,
                        text=str(residue_range.start),
                        style=style,
                    )
                    scene.add_element(start_annotation, parent_id=group_id)
                    logger.debug(
                        f"Added start position annotation to group {group_id}: {start_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to add start position annotation for {element.id}: {e}"
                    )
                    continue

                # Add end position annotation (only if different from start)
                if residue_range.end != residue_range.start:
                    end_id = f"pos_end_{element.id}"
                    try:
                        # Create specific coordinate for end position
                        end_coord = ResidueCoordinate(
                            residue_range.chain_id,
                            residue_range.end,
                            None,  # residue_type not needed
                            0,  # coordinate_index
                        )
                        end_annotation = PositionAnnotation(
                            id=end_id,
                            target=end_coord,
                            position_type=PositionType.RESIDUE_NUMBER,
                            text=str(residue_range.end),
                            style=style,
                        )
                        scene.add_element(end_annotation, parent_id=group_id)
                        logger.debug(
                            f"Added end position annotation to group {group_id}: {end_id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to add end position annotation for {element.id}: {e}"
                        )
                        continue


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
    show_positions: str = "none",
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
        show_positions: Position annotation level controlling residue numbering and terminus labels for each domain.
            Available levels:
            - 'none': No position annotations
            - 'minimal': Only N/C terminus labels per domain (default)
            - 'major': N/C terminus + residue numbers for major secondary structures (≥3 residues)
            - 'full': All position annotations including short structures
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

            # Perform database alignment if family-identity mode is selected
            if config.alignment_mode == "family-identity":
                # Ensure alignment database is available
                logger.info("Ensuring alignment database is available...")
                db_path = ensure_database_available()
                foldseek_db_path = db_path / "foldseek" / "db"

                # Align regions using family-identity mode
                logger.info("Aligning regions using family-identity mode...")
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
            else:  # inertia mode
                # No database alignment - create simple domain transformations for layout only
                logger.info(
                    "Using inertia mode - creating layout-only domain definitions..."
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

            # Apply domain transformations if family-identity mode is used
            if config.alignment_mode == "family-identity":
                logger.info(
                    "Applying domain transformations for aligned orientations..."
                )
                transformed_structure = apply_domain_transformations_masked(
                    original_structure, domain_transformations
                )
            else:
                # Use original structure for inertia mode
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

            # Add position annotations based on level
            if config.show_positions != "none":
                logger.info(
                    f"Adding position annotations to domain groups (level: {config.show_positions})..."
                )
                position_style = None
                if default_styles and "position_annotation" in default_styles:
                    position_style = default_styles["position_annotation"]
                add_domain_aware_position_annotations(
                    scene, domain_transformations, position_style, config.show_positions
                )
                logger.info(
                    f"Position annotations added to domain groups (level: {config.show_positions})"
                )

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
