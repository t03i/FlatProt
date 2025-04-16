# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""
Coordinate Resolver for mapping ResidueCoordinates to rendered coordinates
within the context of SceneElements.
"""

from typing import Dict, Optional
import numpy as np

from flatprot.core import (
    Structure,
    ResidueCoordinate,
    CoordinateCalculationError,
    logger,
)
from .errors import TargetResidueNotFoundError

from .base_element import BaseSceneElement
from .structure import BaseStructureSceneElement


class CoordinateResolver:
    """
    Resolves ResidueCoordinates to their final rendered coordinates.

    This class iterates through relevant scene elements to find the one
    covering the target residue and asks that element for the coordinate
    in its specific rendered space.
    """

    def __init__(
        self, structure: Structure, element_registry: Dict[str, BaseSceneElement]
    ):
        """
        Initializes the CoordinateResolver.

        Args:
            structure: The core Structure object.
            element_registry: The Scene's dictionary mapping element IDs to elements.
        """
        self._structure = structure
        # Filter the registry to only contain structure elements for efficiency
        self._structure_elements = [
            element
            for element in element_registry.values()
            if isinstance(element, BaseStructureSceneElement)
        ]

    def resolve(self, target_residue: ResidueCoordinate) -> np.ndarray:
        """
        Finds the covering structure element and gets the rendered coordinate.

        Args:
            target_residue: The ResidueCoordinate to resolve.

        Returns:
            A NumPy array [3,] with the resolved (X, Y, Z) coordinate.

        Raises:
            TargetResidueNotFoundError: If the residue is not found within any
                                        covering structure element's range.
            CoordinateCalculationError: If the covering element exists but fails
                                        to calculate the specific coordinate, or
                                        if no covering element is found.
        """
        covering_element: Optional[BaseStructureSceneElement] = None
        for element in self._structure_elements:
            # Check if the element's range set exists and contains the target
            if (
                element.residue_range_set
                and target_residue in element.residue_range_set
            ):
                covering_element = element
                break  # Use the first one found

        if covering_element is None:
            logger.warning(
                f"No structure element found covering target residue {target_residue}."
            )
            # Raise specific error indicating no element coverage
            raise CoordinateCalculationError(
                f"Target residue {target_residue} is not covered by any structure element in the scene."
            )

        # Ask the covering element for the coordinate
        try:
            resolved_coord = covering_element.get_coordinate_at_residue(
                target_residue, self._structure
            )

            if resolved_coord is None:
                # Element covered the range but couldn't resolve the specific point
                logger.warning(
                    f"Element '{covering_element.id}' could not provide coordinate for {target_residue}."
                )
                raise CoordinateCalculationError(
                    f"Element '{covering_element.id}' failed to resolve coordinate for {target_residue}."
                )

            # Validate shape
            if not isinstance(resolved_coord, np.ndarray) or resolved_coord.shape != (
                3,
            ):
                logger.error(
                    f"Element '{covering_element.id}' returned invalid coordinate shape for {target_residue}: {type(resolved_coord)} shape {getattr(resolved_coord, 'shape', 'N/A')}"
                )
                raise CoordinateCalculationError(
                    f"Element '{covering_element.id}' returned invalid coordinate data for {target_residue}."
                )

            return resolved_coord

        except TargetResidueNotFoundError as e:
            # This can happen if the element's internal lookup fails
            logger.warning(
                f"Element '{covering_element.id}' could not find {target_residue} internally: {e}"
            )
            raise  # Re-raise the specific error

        except CoordinateCalculationError as e:
            logger.error(
                f"Coordinate calculation error within element '{covering_element.id}' for {target_residue}: {e}",
                exc_info=True,
            )
            raise  # Re-raise calculation errors from the element

        except Exception as e:
            # Catch unexpected errors from the element's method
            logger.error(
                f"Unexpected error in get_coordinate_at_residue for element '{covering_element.id}' and {target_residue}: {e}",
                exc_info=True,
            )
            raise CoordinateCalculationError(
                f"Unexpected error resolving coordinate for {target_residue} via element '{covering_element.id}'."
            ) from e
