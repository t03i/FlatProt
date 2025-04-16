# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Utilities related to protein domain handling and transformations."""

from dataclasses import dataclass
from typing import Optional, List, Dict

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
from flatprot.core import CoordinateCalculationError


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
    spacing: float = 50.0,
    arrangement: str = "horizontal",
    default_styles: Optional[Dict[str, BaseStructureStyle]] = None,
) -> Scene:
    """
    Creates a Scene object with elements grouped by domain and spatially arranged.

    Elements not belonging to any defined domain are skipped. Domain groups
    are added directly under the scene root.

    Args:
        projected_structure: The Structure object with final 2D projected coordinates.
        domain_definitions: List of DomainTransformation objects defining the domains.
                            Used to group elements and identify ranges.
        spacing: Spacing between domain groups in pixels.
        arrangement: How to arrange domains ('horizontal' or 'vertical').
        default_styles: Optional dictionary mapping element type names to style instances.

    Returns:
        A Scene object with domain groups containing translated elements.

    Raises:
        SceneCreationError: If critical errors occur during scene building.
        ValueError: If arrangement is invalid or structure lacks coordinates.
    """
    if (
        projected_structure.coordinates is None
        or projected_structure.coordinates.size == 0
    ):
        raise ValueError("Input projected_structure has no coordinates.")
    if arrangement not in ["horizontal", "vertical"]:
        raise ValueError("Arrangement must be 'horizontal' or 'vertical'.")

    scene = Scene(structure=projected_structure)
    styles = default_styles or {}
    domain_groups: Dict[str, SceneGroup] = {}  # Map domain key to SceneGroup
    domain_keys_in_order: List[str] = []  # Maintain order for layout

    # --- 1. Create Domain Groups under Root ---
    for domain_tf in domain_definitions:
        domain_key = domain_tf.domain_id or str(domain_tf.domain_range)
        if domain_key in domain_groups:
            logger.warning(
                f"Duplicate domain key '{domain_key}' found. Using first encountered."
            )
            continue
        domain_group = SceneGroup(
            id=domain_key, transforms=GroupTransform()
        )  # Initialize empty transform
        try:
            scene.add_element(domain_group)  # Add directly to scene root
            domain_groups[domain_key] = domain_group
            domain_keys_in_order.append(domain_key)
        except Exception as e:
            print(f"Failed to add group for domain {domain_key}: {e}")
            # Decide if this is fatal or recoverable

    # --- 2. Iterate Structure and Assign Elements to Groups ---
    print("Assigning structure elements to domain groups...")
    elements_added_count = 0
    elements_skipped_count = 0
    for _, chain in projected_structure:
        # Use chain.secondary_structure which includes coils and handles gaps
        for ss_element in chain.secondary_structure:
            ss_type = ss_element.secondary_structure_type
            element_info = STRUCTURE_ELEMENT_MAP.get(ss_type)

            if not element_info:
                print(
                    f"Unsupported secondary structure type: {ss_type.value}. Skipping element."
                )
                elements_skipped_count += 1
                continue

            ElementClass, _ = element_info
            ss_range_set = ResidueRangeSet([ss_element])
            element_type_key = ss_type.name.lower()
            style_instance = styles.get(element_type_key, None)

            try:
                # Create the visualization element
                viz_element = ElementClass(
                    residue_range_set=ss_range_set,
                    style=style_instance,
                )

                # Find the domain group this element belongs to
                assigned_parent_id = None
                for domain_tf in domain_definitions:
                    # Check containment using the ResidueRange __contains__
                    if ss_element in domain_tf.domain_range:
                        domain_key = domain_tf.domain_id or str(domain_tf.domain_range)
                        # Get the pre-created SceneGroup ID
                        if domain_key in domain_groups:
                            assigned_parent_id = domain_groups[domain_key].id
                        else:
                            # Should not happen if groups were created correctly
                            logger.error(
                                f"Logic error: Group for domain key '{domain_key}' not found."
                            )
                        break  # Assign to the first matching domain

                # Add element to the scene under the matched group, or skip if no match
                if assigned_parent_id:
                    try:
                        scene.add_element(viz_element, parent_id=assigned_parent_id)
                        elements_added_count += 1
                    except Exception as e:
                        print(
                            f"Failed to add element {viz_element.id} to group {assigned_parent_id}: {e}"
                        )
                        elements_skipped_count += 1
                else:
                    # Skip element as it doesn't belong to any defined domain
                    print(
                        f"Skipping element {viz_element.id} ({ss_element}): Not in any defined domain."
                    )
                    elements_skipped_count += 1

            except Exception as e:
                # Catch errors during element creation
                print(f"Error creating scene element for {ss_element}: {e}")
                elements_skipped_count += 1
                continue  # Skip this element

    print(
        f"Assigned {elements_added_count} elements to domain groups, skipped {elements_skipped_count}."
    )

    # --- 3. Calculate Bounding Boxes and Translations ---
    # This part requires the Scene and its Resolver to get coordinates of elements *within* groups
    print("Calculating domain bounding boxes and translations...")
    domain_bboxes: Dict[str, Dict[str, float]] = {}
    try:
        for domain_key, group in domain_groups.items():
            group_coords_list = []
            # Iterate through the actual children added to the group
            for child_element in group.children:
                # Use resolver to get final projected coordinates for each child
                child_coords = child_element.get_coordinates(projected_structure)
                if child_coords is not None and child_coords.size > 0:
                    group_coords_list.append(child_coords[:, :2])  # Add X,Y coords

            if group_coords_list:
                all_coords = np.vstack(group_coords_list)
                if all_coords.size > 0:
                    min_xy = np.min(all_coords, axis=0)
                    max_xy = np.max(all_coords, axis=0)
                    domain_bboxes[domain_key] = {
                        "min_x": min_xy[0],
                        "min_y": min_xy[1],
                        "max_x": max_xy[0],
                        "max_y": max_xy[1],
                        "width": max_xy[0] - min_xy[0],
                        "height": max_xy[1] - min_xy[1],
                    }
                else:
                    print(
                        f"No valid child coordinates found for domain group {domain_key}"
                    )
            else:
                print(
                    f"No children with coordinates found for domain group {domain_key}"
                )

    except (AttributeError, CoordinateCalculationError, Exception) as e:
        print(
            f"Failed during bounding box calculation: {e}. Cannot apply layout transforms.",
            exc_info=True,
        )
        # Return the scene without layout transforms if bbox calculation fails
        return scene

    # Calculate and apply transforms
    current_offset_x = 0.0
    current_offset_y = 0.0
    for i, domain_key in enumerate(domain_keys_in_order):
        group = domain_groups.get(domain_key)
        bbox = domain_bboxes.get(domain_key)
        if not group or not bbox:
            continue  # Skip if group or bbox missing

        translate_x = 0.0
        translate_y = 0.0
        if i > 0:
            prev_domain_key = domain_keys_in_order[i - 1]
            prev_bbox = domain_bboxes.get(prev_domain_key)
            if prev_bbox:
                if arrangement == "horizontal":
                    current_offset_x += prev_bbox["width"] + spacing
                    translate_x = current_offset_x - bbox["min_x"]
                else:  # vertical
                    current_offset_y += prev_bbox["height"] + spacing
                    translate_y = current_offset_y - bbox["min_y"]

        # Update the transforms attribute of the existing SceneGroup object
        if not np.isclose(translate_x, 0.0) or not np.isclose(translate_y, 0.0):
            # Ensure transforms object exists (should from init)
            if group.transforms is None:
                group.transforms = GroupTransform()
            group.transforms.translate = (translate_x, translate_y)
            print(
                f"Set translate transform for group {domain_key}: ({translate_x:.2f}, {translate_y:.2f})"
            )

    print(f"Applied layout transforms to {len(domain_bboxes)} domain groups.")
    return scene
