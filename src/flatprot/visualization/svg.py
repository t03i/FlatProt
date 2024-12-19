# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import svgwrite
from pathlib import Path
from typing import List
from .renderer import VisualizationRenderer
from .elements.helix import HelixElement


class SvgRenderer(VisualizationRenderer):
    """Renders structure visualizations as SVG."""

    def __init__(self):
        self._elements = []
        self._drawing = None

    def render_structure(
        self,
        structure: Structure,
        elements: List[VisualizationElement],
        output_path: Path,
    ) -> None:
        self._elements = elements
        self._drawing = self._create_drawing(structure)

        for element in sorted(self._elements, key=lambda e: e.z_order):
            self._render_element(element)

        self._drawing.save(output_path)

    def _render_element(self, element: VisualizationElement) -> None:
        """Renders a single visual element."""
        if isinstance(element, HelixElement):
            self._render_helix(element)
        # Add other element type handling

    def _render_helix(self, helix: HelixElement) -> None:
        """Renders a helix element."""
        points = helix.get_path_points()
        self._drawing.add(
            self._drawing.polyline(
                points=points,
                stroke=helix.style.color,
                stroke_width=helix.style.thickness,
                fill="none",
                opacity=helix.style.opacity,
            )
        )
