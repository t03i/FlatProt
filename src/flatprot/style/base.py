# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict
from pydantic_extra_types.color import Color


class ElementStyle(BaseModel):
    """Base class for styling visualization elements"""

    fill_color: Color | Literal["none"] = Field(
        default="none", description="Element fill color"
    )
    stroke_color: Color | Literal["none"] = Field(
        default=Color("#000000"), description="Element stroke color"
    )
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Opacity value between 0 and 1"
    )
    line_width: float = Field(default=5.0, gt=0, description="Width of lines")
    smoothing_factor: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Percentage of points to keep"
    )

    model_config = ConfigDict(
        title="Base Visualization Style",
        frozen=True,  # Makes instances immutable
    )
