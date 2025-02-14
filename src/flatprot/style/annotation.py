# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pydantic_extra_types.color import Color
from pydantic import Field
from typing import Literal
from .base import Style


class AnnotationStyle(Style):
    """Base style for annotations"""

    fill_color: Color | Literal["none"] = Field(
        default="none", description="Element fill color"
    )
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Opacity value between 0 and 1"
    )

    text_size: float = 12
    connector_color: Color = Field(
        default=Color("#666666"), description="Connector color"
    )
    connector_width: float = Field(default=1, description="Connector width")
    connector_opacity: float = Field(default=0.6, description="Connector opacity")
    padding: float = Field(default=5, description="Padding")
