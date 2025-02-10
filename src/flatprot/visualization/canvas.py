# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union
from dataclasses import dataclass

from drawsvg import Drawing, Rectangle, Group
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

from flatprot.core import CoordinateManager, CoordinateType
from .structure import VisualizationElement, StyleManager


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


@dataclass
class CanvasElement:
    render_element: VisualizationElement
    start_idx: int
    end_idx: int
    coord_type: CoordinateType = CoordinateType.CANVAS  # Default to canvas coordinates

    def render(self, coord_manager: CoordinateManager) -> Drawing:
        """Render this element using coordinates from the coordinate manager."""

        element_coords = coord_manager.get(
            self.start_idx, self.end_idx, self.coord_type
        )
        return self.render_element.render(element_coords)


class CanvasGroup:
    """A group of visualization elements that can be nested."""

    def __init__(
        self, elements: list[Union[CanvasElement, "CanvasGroup"]] = None, **kwargs
    ):
        self.transforms = kwargs
        self._validate_elements(elements or [])
        self.elements = elements or []

    def _validate_elements(
        self, elements: list[Union[CanvasElement, "CanvasGroup"]]
    ) -> None:
        """Validate that all elements are either CanvasElement or CanvasGroup instances."""
        for element in elements:
            if not isinstance(element, (CanvasElement, CanvasGroup)):
                raise TypeError(
                    f"Element must be CanvasElement or CanvasGroup, got {type(element)}"
                )

    def add_element(self, element: Union[CanvasElement, "CanvasGroup"]) -> None:
        """Add an element or group to this group."""
        if not isinstance(element, (CanvasElement, CanvasGroup)):
            raise TypeError(
                f"Element must be CanvasElement or CanvasGroup, got {type(element)}"
            )
        self.elements.append(element)

    def render(self, coord_manager: CoordinateManager) -> Group:
        """Render this group using the coordinate manager."""
        group = Group(**self.transforms)
        for element in self.elements:
            group.append(element.render(coord_manager))
        return group


class Canvas:
    """Manages the overall visualization canvas and rendering."""

    def __init__(
        self,
        coordinate_manager: CoordinateManager,
        canvas_settings: Optional[CanvasSettings] = None,
        style_manager: Optional[StyleManager] = None,
    ):
        self.coordinate_manager = coordinate_manager
        self.canvas_settings = canvas_settings or CanvasSettings()
        self.style_manager = style_manager or StyleManager.create_default()
        self.root = CanvasGroup(id="root")

    def add_element(
        self,
        element: VisualizationElement | CanvasGroup,
    ) -> None:
        """Add an element to the scene's root group with its coordinate range."""
        self.root.add_element(element=element)

    def render(self, output_path: Optional[str] = None) -> Drawing:
        """Render the scene to SVG."""
        drawing = Drawing(
            self.canvas_settings.width,
            self.canvas_settings.height,
            origin="center",
        )

        # Add background if specified
        if self.canvas_settings.background_color:
            drawing.append(
                Rectangle(
                    -self.canvas_settings.width / 2,
                    -self.canvas_settings.height / 2,
                    self.canvas_settings.width,
                    self.canvas_settings.height,
                    fill=self.canvas_settings.background_color,
                    opacity=self.canvas_settings.background_opacity,
                    class_="background",
                )
            )

        # Render and add the root group
        drawing.append(self.root.render(self.coordinate_manager))

        if output_path:
            drawing.save_svg(output_path)

        return drawing
