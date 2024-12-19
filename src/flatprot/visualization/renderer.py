# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from ..core.structure import Structure
from .elements.helix import HelixElement
from .elements.sheet import SheetElement


class VisualizationRenderer(ABC):
    """Interface for structure visualization renderers."""

    @abstractmethod
    def render_structure(
        self,
        structure: Structure,
        elements: List[VisualizationElement],
        output_path: Path,
    ) -> None:
        """Renders structure visualization to file.

        Args:
            structure: Structure to visualize
            elements: Visual elements to render
            output_path: Where to save the visualization
        """
        pass

    @abstractmethod
    def add_element(self, element: VisualizationElement) -> None:
        """Adds a visual element to the rendering.

        Args:
            element: Element to add
        """
        pass
