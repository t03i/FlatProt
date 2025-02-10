# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field
from .base import ElementStyle


class HelixStyle(ElementStyle):
    """Settings specific to helix visualization"""

    ribbon_thickness_factor: float = Field(
        default=15.0, gt=0, description="Thickness of helix relative to line width"
    )
    wave_height_factor: float = Field(
        default=0.8,
        gt=0,
        description="Height factor for helix wave",
    )
    min_helix_length: int = Field(
        default=4, gt=0, description="Minimum length of the helix for rendering"
    )
    stroke_width_factor: float = Field(
        default=0.5, gt=0, description="Width of the stroke relative to line width"
    )


class SheetStyle(ElementStyle):
    """Settings specific to sheet visualization"""

    ribbon_thickness_factor: float = Field(
        default=1.0, gt=0, description="Thickness factor for sheet ribbon"
    )
    arrow_width_factor: float = Field(
        default=1.5, gt=0, description="Width factor for sheet arrows"
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
