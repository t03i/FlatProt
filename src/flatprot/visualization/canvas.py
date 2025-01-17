# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from drawsvg import Drawing, Rectangle
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

from flatprot.core.manager import CoordinateManager
from .elements import VisualizationElement, StyleManager


class CanvasSettings(BaseModel):
    """Settings for the canvas"""

    width: int = Field(default=1024, description="Canvas width in pixels")
    height: int = Field(default=1024, description="Canvas height in pixels")
    background_color: Color = Field(
        default=Color("#FFFFFF"), description="Background color in hex format"
    )
    background_opacity: float = Field(default=0, description="Background opacity (0-1)")
    padding: float = Field(
        default=0.05, ge=0, le=1, description="Padding as fraction of canvas size"
    )

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns canvas dimensions as (width, height)"""
        return (self.width, self.height)

    @property
    def padding_pixels(self) -> tuple[float, float]:
        """Returns padding in pixels as (x_padding, y_padding)"""
        return (self.width * self.padding, self.height * self.padding)


class Canvas:
    """Manages the overall visualization canvas and rendering."""

    def __init__(
        self,
        coordinate_manager: CoordinateManager,
        canvas_settings: Optional[CanvasSettings] = None,
        style_manager: Optional[StyleManager] = None,
    ):
        self.canvas_settings = canvas_settings or CanvasSettings()
        self.style_manager = style_manager or StyleManager.create_default()
        self.elements: list[VisualizationElement] = []

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
                Rectangle(
                    -self.canvas_settings.width / 2,  # Account for centered origin
                    -self.canvas_settings.height / 2,
                    self.canvas_settings.width,
                    self.canvas_settings.height,
                    fill=self.canvas_settings.background_color,
                    opacity=self.canvas_settings.background_opacity,
                    class_="background",
                )
            )

        # Render and add the root group
        drawing.append(self.root.render())

        if output_path:
            drawing.save_svg(output_path)

        return drawing
