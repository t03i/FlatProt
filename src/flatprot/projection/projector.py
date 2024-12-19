# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from pathlib import Path
from typing import Optional

from ..structure.components import Structure
from .utils import ProjectionMatrix


class ProjectionScope(Enum):
    STRUCTURE = "structure"  # Project entire structure together
    CHAIN = "chain"  # Project each chain independently


class Projector(ABC):
    """Base class for all projectors using template method pattern."""

    def __init__(self, scope: ProjectionScope = ProjectionScope.STRUCTURE):
        self.scope = scope
        self._cached_projections: dict[Optional[str], ProjectionMatrix] = {}

    def project(self, structure: Structure) -> dict[str, np.ndarray]:
        """Template method defining the projection workflow.

        Returns:
            dictionary mapping chain IDs to their 2D coordinates
        """
        if self.scope == ProjectionScope.STRUCTURE:
            return self._project_whole_structure(structure)
        else:
            return self._project_individual_chains(structure)

    def _project_whole_structure(self, structure: Structure) -> dict[str, np.ndarray]:
        """Projects entire structure using single transformation."""
        if None not in self._cached_projections:
            # Get all coordinates
            all_coords = np.vstack([chain.coordinates for chain in structure.values()])
            self._cached_projections[None] = self._calculate_projection(
                structure, all_coords
            )

        # Apply same transformation to each chain
        return {
            chain_id: self._apply_cached_projection(
                chain, self._cached_projections[None]
            )
            for chain_id, chain in structure.items()
        }

    def _project_individual_chains(self, structure: Structure) -> dict[str, np.ndarray]:
        """Projects each chain independently."""
        projections = {}
        for chain_id, chain in structure.items():
            if chain_id not in self._cached_projections:
                coords = chain.coordinates
                self._cached_projections[chain_id] = self._calculate_projection(
                    structure, coords, chain_id
                )

            projections[chain_id] = self._apply_cached_projection(
                chain, self._cached_projections[chain_id]
            )
        return projections

    @abstractmethod
    def _calculate_projection(
        self,
        structure: Structure,
        coordinates: np.ndarray,
        chain_id: Optional[str] = None,
    ) -> np.ndarray:
        """Calculate projection for given coordinates.

        Args:
            structure: Complete structure (for context)
            coordinates: Coordinates to project
            chain_id: Chain ID if projecting single chain, None for whole structure

        Returns:
            Projected 2D coordinates
        """
        pass

    @abstractmethod
    def _apply_cached_projection(
        self, chain, cached_projection: np.ndarray
    ) -> np.ndarray:
        """Apply cached projection to chain coordinates."""
        pass

    def save(self, path: Path) -> None:
        """Saves projection parameters.

        Args:
            path: Where to save the projection

        Raises:
            ValueError: If the file extension is not '.npz'
        """
        if path.suffix != ".npz":
            raise ValueError(
                f"Projection must be saved as .npz file, got '{path.suffix}'"
            )

        # Convert ProjectionMatrix objects to their underlying numpy arrays
        save_dict = {
            "cached_projections": {
                str(k) if k is not None else "None": v.to_array()
                for k, v in self._cached_projections.items()
            },
            "scope": self.scope.value,
        }
        np.savez(path, **save_dict)

    def load(self, path: Path) -> None:
        """Loads projection parameters.

        Args:
            path: From where to load the projection

        Raises:
            ValueError: If the file extension is not '.npz'
        """
        if path.suffix != ".npz":
            raise ValueError(
                f"Projection must be loaded from .npz file, got '{path.suffix}'"
            )

        loaded = np.load(path, allow_pickle=True)

        # Convert back from numpy arrays to ProjectionMatrix objects
        cached_dict = loaded["cached_projections"].item()
        self._cached_projections = {
            None if k == "None" else k: ProjectionMatrix.from_array(v)
            for k, v in cached_dict.items()
        }
        self.scope = ProjectionScope(loaded["scope"].item())
