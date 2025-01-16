# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color


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
