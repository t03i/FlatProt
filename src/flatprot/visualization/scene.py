# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from drawsvg import Drawing
from typing import Optional

from flatprot.visualization.elements import VisualizationElement
from flatprot.visualization.elements.group import Group
from .utils import CanvasSettings


class Scene:
    """Manages the overall visualization scene and rendering."""

    def __init__(
        self,
        canvas_settings: Optional[CanvasSettings] = None,
    ):
        self.canvas_settings = canvas_settings or CanvasSettings()
        self.root = Group([])  # Root group containing all elements

    def add_element(self, element: VisualizationElement) -> None:
        """Add an element to the scene's root group."""
        self.root.add_element(element)

    def render(self, output_path: Optional[str] = None) -> Drawing:
        """Render the scene to SVG.

        Args:
            output_path: If provided, saves the SVG to this path

        Returns:
            The rendered Drawing object
        """
        drawing = Drawing(
            self.canvas_settings.width,
            self.canvas_settings.height,
            origin="center",  # This makes (0,0) the center of the canvas
        )

        # Add background if specified
        if self.canvas_settings.background_color:
            drawing.append(
                drawing.rect(
                    -self.canvas_settings.width / 2,  # Account for centered origin
                    -self.canvas_settings.height / 2,
                    self.canvas_settings.width,
                    self.canvas_settings.height,
                    fill=self.canvas_settings.background_color,
                )
            )

        # Render and add the root group
        drawing.append(self.root.render())

        if output_path:
            drawing.save_svg(output_path)

        return drawing
