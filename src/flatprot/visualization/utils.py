# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
import pydantic


class CanvasSettings(BaseModel):
    """Settings for the canvas"""

    width: int = Field(default=1024, description="Canvas width in pixels")
    height: int = Field(default=1024, description="Canvas height in pixels")
    background_color: pydantic.ColorType = Field(
        default="#FFFFFF", description="Background color in hex format"
    )
    padding: float = Field(
        default=0.05, description="Padding as fraction of canvas size"
    )

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns canvas dimensions as (width, height)"""
        return (self.width, self.height)

    @property
    def padding_pixels(self) -> tuple[float, float]:
        """Returns padding in pixels as (x_padding, y_padding)"""
        return (self.width * self.padding, self.height * self.padding)
