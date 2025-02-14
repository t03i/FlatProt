# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, ConfigDict
from pydantic_extra_types.color import Color
from typing import Literal

from .base import Style


class ElementStyle(Style):
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


class HelixStyle(ElementStyle):
    """Settings specific to helix visualization"""

    ribbon_thickness_factor: float = Field(
        default=10.0, gt=0, description="Thickness of helix relative to line width"
    )
    amplitude: float = Field(
        default=5,
        gt=0,
        description="Height factor for helix wave",
    )
    min_helix_length: int = Field(
        default=4, gt=0, description="Minimum length of the helix for rendering"
    )
    stroke_width_factor: float = Field(
        default=0.5, gt=0, description="Width of the stroke relative to line width"
    )
    wavelength: float = Field(
        default=15, gt=0, description="Width of the cross relative to line width"
    )


class SheetStyle(ElementStyle):
    """Settings specific to sheet visualization"""

    arrow_width_factor: float = Field(
        default=10, gt=0, description="Width factor for sheet arrows"
    )
    stroke_width_factor: float = Field(
        default=0.5, gt=0, description="Width of the stroke relative to line width"
    )
    min_sheet_length: int = Field(
        default=3, gt=0, description="Minimum length of the sheet for rendering"
    )


class CoilStyle(ElementStyle):
    """Settings specific to coil visualization"""

    stroke_width_factor: float = Field(
        default=1.0, gt=0, description="Thickness relative to line width"
    )
