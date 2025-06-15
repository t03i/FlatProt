# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Utilities related to protein domain handling and transformations."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
import os

import numpy as np
import gemmi

from flatprot.core import Structure, logger
from flatprot.core.coordinates import ResidueRange, ResidueRangeSet
from flatprot.transformation import TransformationMatrix, TransformationError
from flatprot.scene import (
    Scene,
    SceneGroup,
    GroupTransform,
    BaseStructureStyle,
)
from flatprot.utils.scene_utils import STRUCTURE_ELEMENT_MAP
from flatprot.scene.connection import Connection, ConnectionStyle
from flatprot.scene import AreaAnnotation, AreaAnnotationStyle
from flatprot.alignment import (
    AlignmentDatabase,
    align_structure_database,
    get_aligned_rotation_database,
)
from flatprot.utils.database import DEFAULT_DB_DIR


@dataclass
class DomainTransformation:
    """
    Encapsulates a transformation matrix applied to a specific protein domain.

    Attributes:
        domain_range: The specific residue range defining the domain.
        transformation_matrix: The matrix used to transform this domain.
        domain_id: An optional identifier for the domain (e.g., 'Domain1', 'N-term').
        scop_id: An optional SCOP family identifier from alignment (e.g., '3000114').
        alignment_probability: The alignment probability/quality score (0.0-1.0).
    """

    domain_range: ResidueRange
    transformation_matrix: TransformationMatrix
    domain_id: Optional[str] = None
    scop_id: Optional[str] = None
    alignment_probability: Optional[float] = None

    def __post_init__(self):
        """Validate inputs."""
        if not isinstance(self.domain_range, ResidueRange):
            raise TypeError("domain_range must be a ResidueRange object.")
        if not isinstance(self.transformation_matrix, TransformationMatrix):
            raise TypeError(
                "transformation_matrix must be a TransformationMatrix object."
            )
        if self.domain_id is not None and not isinstance(self.domain_id, str):
            raise TypeError("domain_id must be a string if provided.")
        if self.scop_id is not None and not isinstance(self.scop_id, str):
            raise TypeError("scop_id must be a string if provided.")

    def __repr__(self) -> str:
        """Provide a concise representation."""
        name = f" '{self.domain_id}'" if self.domain_id else ""
        return f"<DomainTransformation{name} range={self.domain_range}>"


def apply_domain_transformations_masked(
    structure: Structure,
    domain_transforms: List[DomainTransformation],
) -> Structure:
    """
    Applies specific transformation matrices sequentially to defined domains.

    Uses boolean masks to identify coordinates for each domain. If domains overlap,
    transformations applied later in the `domain_transforms` list will overwrite
    those applied earlier for the overlapping coordinates.

    Creates a new Structure object with transformed 3D coordinates. The original
    structure remains unchanged.

    Args:
        structure: The original Structure object.
        domain_transforms: An ordered list of DomainTransformation objects.
                           The order determines precedence in case of overlaps.

    Returns:
        A new Structure object with coordinates transformed domain-specifically.

    Raises:
        TransformationError: If matrix application fails.
        ValueError: If structure lacks coordinates or if coordinates are not 3D.
    """
    if not hasattr(structure, "coordinates") or structure.coordinates is None:
        raise ValueError("Structure has no coordinates to transform.")
    if structure.coordinates.shape[1] != 3:
        raise ValueError(
            f"Expected 3D coordinates, but got shape {structure.coordinates.shape}"
        )

    original_coords = structure.coordinates
    num_atoms = original_coords.shape[0]

    # 1. Create Transformation Masks (one per domain)
    domain_masks: List[np.ndarray] = [
        np.zeros(num_atoms, dtype=bool) for _ in domain_transforms
    ]

    # 2. & 3. Iterate through structure coords and populate masks
    for chain in structure.values():
        for residue in chain:
            for domain_idx, domain_tf in enumerate(domain_transforms):
                if residue in domain_tf.domain_range:
                    if residue.coordinate_index < domain_masks[domain_idx].shape[0]:
                        domain_masks[domain_idx][residue.coordinate_index] = True

    transformed_coords = original_coords.copy()

    for mask, domain_tf in zip(domain_masks, domain_transforms):
        if not np.any(mask):
            print(
                f"No coordinates found for domain {domain_tf.domain_range}. Skipping transformation."
            )
            continue

        matrix = domain_tf.transformation_matrix
        domain_id_str = f" '{domain_tf.domain_id}'" if domain_tf.domain_id else ""
        print(
            f"Applying transformation for domain{domain_id_str} {domain_tf.domain_range} (affecting {np.sum(mask)} coordinates)..."
        )

        # Apply transformation to the *original* coordinates selected by the mask
        try:
            coords_subset = original_coords[mask, :]
            transformed_subset = matrix.apply(coords_subset)

            if transformed_subset.shape != coords_subset.shape:
                raise TransformationError(
                    f"Transformation resulted in unexpected shape change for domain {domain_tf.domain_range}. "
                    f"Input shape: {coords_subset.shape}, Output shape: {transformed_subset.shape}"
                )

            transformed_coords[mask, :] = transformed_subset

        except Exception as e:
            raise TransformationError(
                f"Failed to apply transformation matrix for domain {domain_tf.domain_range}: {e}"
            ) from e

    # Create a new structure object with the same topology but new coordinates

    new_structure = structure.with_coordinates(transformed_coords)

    return new_structure


# --- Domain-Aware Scene Builder (Reversed Flow) ---


def create_domain_aware_scene(
    projected_structure: Structure,
    domain_definitions: List[DomainTransformation],
    spacing: float = 0.0,
    arrangement: str = "horizontal",
    default_styles: Optional[
        Dict[str, Union[BaseStructureStyle, ConnectionStyle, AreaAnnotationStyle]]
    ] = None,
    domain_scop_ids: Optional[Dict[str, str]] = None,
    domain_alignment_probabilities: Optional[Dict[str, float]] = None,
) -> Scene:
    """Creates a Scene containing only specified domains, each in its own group.

    Elements (structure, connections, annotations) are assigned to their respective
    domain group. Elements not belonging to any defined domain are discarded.
    Domain groups are translated by a fixed margin based on the `spacing` parameter.

    Args:
        projected_structure: The Structure object with final 2D projected coordinates.
        domain_definitions: List of DomainTransformation objects defining the domains.
                            The domain_id attribute is crucial.
        spacing: Fixed margin used to separate domain groups in pixels.
        arrangement: How to arrange domain groups ('horizontal' or 'vertical').
        default_styles: Optional dictionary mapping element type names to style instances.
        domain_scop_ids: Optional dictionary mapping domain_id to an annotation string
                         (e.g., SCOP ID) used for AreaAnnotation labels.
        domain_alignment_probabilities: Optional dictionary mapping domain_id to alignment
                                        probability (0.0-1.0) for display in annotations.

    Returns:
        A Scene object containing only the specified domain groups, laid out.

    Raises:
        ValueError: If arrangement is invalid, structure lacks coordinates, or
                    domain_definitions have issues (missing IDs when needed, duplicates).
        TypeError: If incompatible style types are provided in default_styles.
    """
    if (
        projected_structure.coordinates is None
        or projected_structure.coordinates.size == 0
    ):
        raise ValueError("Input projected_structure has no coordinates.")
    if arrangement not in ["horizontal", "vertical"]:
        raise ValueError("Arrangement must be 'horizontal' or 'vertical'.")
    if domain_scop_ids and any(d.domain_id is None for d in domain_definitions):
        raise ValueError(
            "All domain_definitions must have a domain_id if domain_scop_ids is provided."
        )
    defined_domain_ids = [d.domain_id for d in domain_definitions if d.domain_id]
    if len(defined_domain_ids) != len(set(defined_domain_ids)):
        raise ValueError("Duplicate domain_ids found in domain_definitions.")
    if domain_scop_ids:
        scop_keys = set(domain_scop_ids.keys())
        if not scop_keys.issubset(set(defined_domain_ids)):
            missing = scop_keys - set(defined_domain_ids)
            raise ValueError(f"domain_scop_ids keys not in domain_ids: {missing}")

    scene = Scene(structure=projected_structure)
    styles = default_styles or {}
    domain_groups: Dict[str, SceneGroup] = {}  # Map domain_id to SceneGroup
    domain_ids_in_order: List[str] = []  # Maintain order for fixed layout
    domain_tf_lookup: Dict[str, DomainTransformation] = {}

    # --- 1. Create Domain Groups Only ---
    print("Creating scene groups for defined domains...")
    for domain_tf in domain_definitions:
        domain_id = domain_tf.domain_id or str(domain_tf.domain_range)
        if domain_id in domain_groups:
            logger.warning(f"Skipping duplicate domain ID '{domain_id}'.")
            continue
        domain_group = SceneGroup(id=domain_id, transforms=GroupTransform())
        scene.add_element(domain_group)
        domain_groups[domain_id] = domain_group
        domain_ids_in_order.append(domain_id)
        domain_tf_lookup[domain_id] = domain_tf
    print(f"Created {len(domain_groups)} domain groups.")
    if not domain_groups:
        logger.warning("No domain groups created. Scene will be empty.")
        return scene  # Return early if no domains defined/created

    # --- 2. Assign Structure Elements ONLY to Domain Groups ---
    print("Assigning structure elements to domain groups...")
    element_map: Dict[str, SceneGroup] = {}  # Map element ID to its parent group
    elements_assigned_count = 0
    elements_discarded_count = 0

    for _, chain in projected_structure:
        for ss_element in chain.secondary_structure:
            ss_type = ss_element.secondary_structure_type
            element_info = STRUCTURE_ELEMENT_MAP.get(ss_type)

            if not element_info:
                elements_discarded_count += 1
                continue  # Skip unsupported types

            ElementClass, _ = element_info
            ss_range_set = ResidueRangeSet([ss_element])
            element_type_key = ss_type.name.lower()

            assigned_group: Optional[SceneGroup] = None
            # Find the domain this element belongs to
            for domain_tf in domain_definitions:
                if ss_element in domain_tf.domain_range:
                    domain_id = domain_tf.domain_id or str(domain_tf.domain_range)
                    assigned_group = domain_groups.get(domain_id)
                    break  # Assign to first matching domain

            # If element belongs to a defined domain, create and add it
            if assigned_group:
                try:
                    base_style = styles.get(element_type_key)
                    # Type check base style
                    if base_style is not None and not isinstance(
                        base_style, BaseStructureStyle
                    ):
                        logger.warning(
                            f"Invalid style type for '{element_type_key}'. Using default."
                        )
                        base_style = None

                    viz_element = ElementClass(
                        residue_range_set=ss_range_set,
                        style=base_style,  # Pass style or None
                    )
                    scene.add_element(viz_element, parent_id=assigned_group.id)
                    element_map[viz_element.id] = assigned_group
                    elements_assigned_count += 1
                except Exception as e:
                    logger.error(
                        f"Error creating/assigning element {ss_element}: {e}",
                        exc_info=True,
                    )
                    elements_discarded_count += 1
            else:
                # Element does not belong to any defined domain, discard it
                elements_discarded_count += 1

    print(
        f"Assigned {elements_assigned_count} elements to domain groups. Discarded {elements_discarded_count}."
    )

    # --- 3. Add Connections ONLY Within the Same Domain Group ---
    print("Adding connections within domain groups...")
    all_structure_elements = scene.get_sequential_structure_elements()
    connections_added_count = 0
    connections_discarded_count = 0

    for i in range(len(all_structure_elements) - 1):
        element_i = all_structure_elements[i]
        element_i_plus_1 = all_structure_elements[i + 1]

        # Check adjacency first
        if not element_i.is_adjacent_to(element_i_plus_1):
            continue

        # Find the groups these elements belong to (if they were added)
        group_i = element_map.get(element_i.id)
        group_i_plus_1 = element_map.get(element_i_plus_1.id)

        # Only add connection if both elements exist and are in the SAME group
        if group_i is not None and group_i is group_i_plus_1:
            try:
                base_conn_style = styles.get("connection")
                # Type check
                if base_conn_style is not None and not isinstance(
                    base_conn_style, ConnectionStyle
                ):
                    logger.warning(
                        "Invalid type for 'connection' style. Using default."
                    )
                    base_conn_style = None

                conn = Connection(
                    start_element=element_i,
                    end_element=element_i_plus_1,
                    style=base_conn_style,
                )
                print(
                    f"Adding connection between {element_i.id} and {element_i_plus_1.id} to group {group_i.id}"
                )
                scene.add_element(conn, parent_id=group_i.id)
                connections_added_count += 1
            except Exception as e:
                logger.error(
                    f"Failed adding connection between {element_i.id}/{element_i_plus_1.id}: {e}",
                    exc_info=True,
                )
                connections_discarded_count += 1  # Count as discarded if creation fails
        else:
            # Connection spans groups or involves discarded elements
            connections_discarded_count += 1

    print(
        f"Added {connections_added_count} connections within groups. Discarded {connections_discarded_count}."
    )

    # --- 4. Add Domain Annotations to Respective Groups ---
    if domain_scop_ids:
        print("Adding domain annotations...")
        annotations_added_count = 0
        base_area_style = styles.get("area_annotation")
        print(f"Base area style: {base_area_style}")
        # Type check
        if base_area_style is not None and not isinstance(
            base_area_style, AreaAnnotationStyle
        ):
            logger.warning("Invalid type for 'area_annotation' style. Using default.")
            base_area_style = None

        for domain_id, scop_id in domain_scop_ids.items():
            group = domain_groups.get(domain_id)
            domain_tf = domain_tf_lookup.get(domain_id)
            if not group or not domain_tf:
                logger.warning(
                    f"Cannot add annotation for domain '{domain_id}': missing group/definition."
                )
                continue
            try:
                target_range_set = ResidueRangeSet([domain_tf.domain_range])

                # Create label with SCOP ID and alignment probability
                label = scop_id
                if (
                    domain_alignment_probabilities
                    and domain_id in domain_alignment_probabilities
                ):
                    probability = domain_alignment_probabilities[domain_id]
                    label = (
                        f"{scop_id}\n({probability:.1%})"  # e.g., "3000622\n(85.3%)"
                    )

                annotation = AreaAnnotation(
                    id=f"{domain_id}_area",
                    residue_range_set=target_range_set,
                    style=base_area_style,  # Pass style or None
                    label=label,
                )
                # Add annotation as child of the specific domain group
                scene.add_element(annotation, parent_id=group.id)
                annotations_added_count += 1
            except Exception as e:
                logger.error(
                    f"Failed adding area annotation for domain {domain_id}: {e}",
                    exc_info=True,
                )
        print(f"Added {annotations_added_count} domain area annotations.")

    # --- 5. Apply Fixed Layout Translation to Domain Groups ---
    print("Applying fixed layout translations...")
    layout_applied_count = 0
    # Use the ordered list for consistent layout
    total = len(domain_ids_in_order)
    for i, domain_id in enumerate(domain_ids_in_order):
        group = domain_groups.get(domain_id)
        if not group:
            continue  # Should not happen based on checks

        # Calculate fixed translation based on index and spacing (margin)
        if arrangement == "horizontal":
            idx = total - i - 1  # Reverse for horizontal (right to left)
            translate_x = idx * spacing
            translate_y = 0.0
        elif arrangement == "vertical":
            idx = total - i - 1  # Reverse for vertical (bottom to top)
            translate_x = 0.0
            translate_y = idx * spacing

        # Apply the translation
        if group.transforms is None:
            group.transforms = GroupTransform()  # Should exist, but safety check
        group.transforms.translate = (translate_x, translate_y)
        print(
            f"Applied fixed translate to group {domain_id}: ({translate_x:.2f}, {translate_y:.2f})"
        )
        layout_applied_count += 1

    print(f"Applied fixed layout transforms to {layout_applied_count} domain groups.")

    return scene


def extract_structure_regions(
    structure_file: Path, regions: List[ResidueRange], output_dir: Path
) -> List[Tuple[Path, ResidueRange]]:
    """Extract specified regions from a structure file to separate CIF files.

    Args:
        structure_file: Path to input structure file (PDB/CIF)
        regions: List of ResidueRange objects defining regions to extract
        output_dir: Directory to write extracted region files

    Returns:
        List of tuples (output_file_path, residue_range) for successfully extracted regions

    Raises:
        ValueError: If structure file cannot be read or regions are invalid
        FileNotFoundError: If structure file does not exist
    """
    if not structure_file.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")

    if not regions:
        raise ValueError("No regions provided for extraction")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load structure using GEMMI
        structure = gemmi.read_structure(
            str(structure_file), merge_chain_parts=True, format=gemmi.CoorFormat.Detect
        )
    except Exception as e:
        raise ValueError(f"Failed to read structure file {structure_file}: {e}")

    extracted_files = []
    structure_id = structure_file.stem

    for i, region in enumerate(regions):
        try:
            # Create output filename
            region_id = f"{region.chain_id}_{region.start}_{region.end}"
            output_file = output_dir / f"{structure_id}_region_{region_id}.cif"

            # Extract region using helper function
            _extract_region_to_file(structure, region, output_file)

            extracted_files.append((output_file, region))
            logger.info(f"Extracted region {region} to {output_file.name}")

        except Exception as e:
            logger.warning(f"Failed to extract region {region}: {e}")
            continue

    if not extracted_files:
        raise ValueError("No regions were successfully extracted")

    logger.info(f"Successfully extracted {len(extracted_files)} regions")
    return extracted_files


def _extract_region_to_file(
    structure: gemmi.Structure, region: ResidueRange, output_file: Path
) -> None:
    """Extract a single region from a GEMMI structure to a CIF file.

    Args:
        structure: GEMMI Structure object
        region: ResidueRange defining the region to extract
        output_file: Path where to write the extracted region

    Raises:
        ValueError: If chain not found or no residues in range
    """
    # Create new structure for the domain
    domain_structure = gemmi.Structure()
    domain_structure.name = output_file.stem.replace(":", "_").replace("-", "_")

    # Create new model
    model = gemmi.Model("1")

    # Find the original chain
    original_chain = None
    if structure and len(structure) > 0:
        for chain in structure[0]:
            if chain.name == region.chain_id:
                original_chain = chain
                break

    if original_chain is None:
        raise ValueError(f"Chain '{region.chain_id}' not found in structure")

    # Create new chain and extract residues
    new_chain = gemmi.Chain(region.chain_id)
    extracted_count = 0

    for residue in original_chain:
        seq_id = residue.seqid.num
        if region.start <= seq_id <= region.end:
            new_chain.add_residue(residue.clone())
            extracted_count += 1

    if extracted_count == 0:
        raise ValueError(
            f"No residues found in range {region.start}-{region.end} "
            f"for chain '{region.chain_id}'"
        )

    # Assemble and write structure
    model.add_chain(new_chain)
    domain_structure.add_model(model)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    domain_structure.make_mmcif_document().write_file(str(output_file))


def align_regions_batch(
    region_files: List[Tuple[Path, ResidueRange]],
    region_ranges: List[ResidueRange],
    foldseek_db_path: Path,
    foldseek_executable: str,
    min_probability: float,
    alignment_mode: str = "family-identity",
) -> List[DomainTransformation]:
    """Align extracted regions using Foldseek and return transformations.

    Args:
        region_files: List of (file_path, residue_range) tuples from extract_structure_regions
        region_ranges: Original list of ResidueRange objects for identification
        foldseek_db_path: Path to Foldseek database
        foldseek_executable: Path to Foldseek executable
        min_probability: Minimum alignment probability threshold
        alignment_mode: Alignment strategy ('family-identity' or 'inertia')

    Returns:
        List of DomainTransformation objects for successfully aligned regions

    Raises:
        ValueError: If no successful alignments found
        RuntimeError: If alignment database is not accessible
    """
    if not region_files:
        raise ValueError("No region files provided for alignment")

    # Initialize alignment database
    db_file_path = DEFAULT_DB_DIR / "alignments.h5"
    if not db_file_path.exists():
        raise RuntimeError(f"Alignment database not found: {db_file_path}")

    try:
        alignment_db = AlignmentDatabase(db_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize alignment database: {e}")

    domain_transformations = []
    failed_regions = []

    logger.info(f"Aligning {len(region_files)} regions using {alignment_mode} mode")

    for file_path, region in region_files:
        try:
            region_id = f"{region.chain_id}:{region.start}-{region.end}"
            logger.info(f"Aligning region {region_id} from {file_path.name}")

            # Perform Foldseek alignment
            alignment_result = align_structure_database(
                file_path, foldseek_db_path, foldseek_executable, min_probability
            )

            logger.info(
                f"Alignment hit for {region_id}: {alignment_result.db_id} "
                f"(P={alignment_result.probability:.3f})"
            )

            # Get transformation matrix based on alignment mode
            scop_id = None
            if alignment_mode == "family-identity":
                # Get rotation matrix from database
                matrix_result = get_aligned_rotation_database(
                    alignment_result, alignment_db
                )
                transformation_matrix = matrix_result[0]
                db_entry = matrix_result[1]

                if transformation_matrix is None:
                    logger.warning(f"No transformation matrix found for {region_id}")
                    failed_regions.append(region)
                    continue

                # Extract SCOP ID from database entry
                if db_entry and hasattr(db_entry, "entry_id") and db_entry.entry_id:
                    scop_id = str(db_entry.entry_id)
                    logger.info(f"Retrieved SCOP ID for {region_id}: {scop_id}")
                else:
                    # Fallback to alignment result DB ID
                    scop_id = (
                        str(alignment_result.db_id) if alignment_result.db_id else None
                    )
                    if scop_id:
                        logger.info(
                            f"Using alignment DB ID as SCOP ID for {region_id}: {scop_id}"
                        )
                    else:
                        logger.warning(f"No SCOP ID available for {region_id}")

                logger.info(f"Retrieved transformation matrix for {region_id}")

            else:  # inertia mode
                # Use identity matrix for inertia-based alignment
                transformation_matrix = TransformationMatrix(
                    rotation=np.eye(3), translation=np.zeros(3)
                )
                # For inertia mode, still capture the alignment DB ID as SCOP ID
                scop_id = (
                    str(alignment_result.db_id) if alignment_result.db_id else None
                )
                logger.info(f"Using identity matrix for inertia mode: {region_id}")
                if scop_id:
                    logger.info(f"Captured alignment DB ID for {region_id}: {scop_id}")
                else:
                    logger.warning(f"No SCOP ID available for {region_id}")

            # Create DomainTransformation object
            domain_transformation = DomainTransformation(
                domain_range=region,
                transformation_matrix=transformation_matrix,
                domain_id=region_id,
                scop_id=scop_id,
                alignment_probability=alignment_result.probability,
            )

            domain_transformations.append(domain_transformation)
            logger.info(f"Successfully processed region {region_id}")

        except Exception as e:
            logger.warning(f"Failed to align region {region}: {e}")
            failed_regions.append(region)
            continue

        finally:
            # Clean up temporary file
            try:
                if file_path.exists():
                    os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

    if not domain_transformations:
        raise ValueError("No regions were successfully aligned")

    logger.info(
        f"Batch alignment completed: {len(domain_transformations)} successful, "
        f"{len(failed_regions)} failed"
    )

    return domain_transformations
