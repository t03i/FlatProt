# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Protocol

from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color
import numpy as np
import drawsvg as draw

from ..elements import VisualStyle


class AnnotationTarget(Protocol):
    def get_coordinates_for_index(self, index: int) -> np.ndarray:
        pass

    def get_bounding_box(self) -> np.ndarray:
        pass

    def get_visual_style(self) -> VisualStyle:
        pass


class AnnotationStyle(BaseModel):
    """Base style for annotations"""

    text_size: float = 12
    fill_color: Color = Field(
        default=Color("#000000"), description="Element fill color"
    )
    connector_color: Color = Field(
        default=Color("#666666"), description="Connector color"
    )
    connector_width: float = Field(default=1, description="Connector width")
    connector_opacity: float = Field(default=0.6, description="Connector opacity")
    padding: float = Field(default=5, description="Padding")


class Annotation:
    def __init__(
        self,
        label: str,
        target: AnnotationTarget,
        style: Optional[AnnotationStyle] = None,
    ):
        self.label = label
        self.target = target
        self.style = style

    def render(self) -> draw.DrawingElement:
        pass
