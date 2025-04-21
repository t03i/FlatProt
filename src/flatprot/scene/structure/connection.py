# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

from .base_structure import BaseStructureSceneElement


class ConnectionStyle(BaseModel):
    """Defines the visual style for a connection between residues."""

    stroke: Color = Field(default=Color("black"), description="Connection line color.")
    stroke_width: float = Field(
        default=1.0, ge=0.0, description="Connection line width."
    )
    stroke_dasharray: Optional[str] = Field(
        default=None, description="SVG stroke-dasharray pattern (e.g., '5,5')."
    )
    opacity: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Connection line opacity."
    )
    z_index: int = Field(
        default=-1,
        description="Relative Z-index for rendering order (lower is further back).",
    )


class Connection(BaseModel):
    """Represents a connection (e.g., disulfide bond, crosslink) between two residues."""

    start_element: BaseStructureSceneElement
    end_element: BaseStructureSceneElement
    type: str = Field(
        default="custom",
        description="Type of connection (e.g., 'disulfide', 'salt_bridge', 'custom').",
    )
    style: ConnectionStyle = Field(default_factory=ConnectionStyle)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"Connection({self.type}: {self.start_element} <-> {self.end_element})"

    def __repr__(self) -> str:
        return f"Connection(start_element={self.start_element!r}, end_element={self.end_element!r}, type='{self.type}', style={self.style!r})"
