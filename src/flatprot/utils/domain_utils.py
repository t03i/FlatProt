# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Utilities related to protein domain handling and transformations."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union

import numpy as np

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


@dataclass
class DomainTransformation:
    """
    Encapsulates a transformation matrix applied to a specific protein domain.

    Attributes:
        domain_range: The specific residue range defining the domain.
        transformation_matrix: The matrix used to transform this domain.
        domain_id: An optional identifier for the domain (e.g., 'Domain1', 'N-term').
    """

    domain_range: ResidueRange
    transformation_matrix: TransformationMatrix
    domain_id: Optional[str] = None

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
                annotation = AreaAnnotation(
                    id=f"{domain_id}_area",
                    residue_range_set=target_range_set,
                    style=base_area_style,  # Pass style or None
                    label=scop_id,
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
        idx = total - i - 1
        if not group:
            continue  # Should not happen based on checks

        # Calculate fixed translation based on index and spacing (margin)
        if arrangement == "horizontal":
            translate_x = idx * spacing
            translate_y = 0.0
        else:  # vertical
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
