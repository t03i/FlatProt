# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Literal

from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core.structure import Structure

from .structure.base_structure import BaseStructureSceneElement
from .base_element import BaseSceneElement, SceneGroupType, BaseSceneStyle


class ConnectionStyle(BaseSceneStyle):
    """Defines the visual style for a connection between elements."""

    color: Color = Field(default=Color("#5b5859"), description="Connection line color.")
    stroke_width: float = Field(
        default=1.0, ge=0.0, description="Connection line width."
    )
    # Use Literal for specific allowed values and rename field
    line_style: Literal["solid", "dashed", "dotted"] = Field(
        default="solid",
        description="Connection line style ('solid', 'dashed', or 'dotted').",
    )
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Connection line opacity."
    )


class Connection(BaseSceneElement[ConnectionStyle]):
    """Represents a connection (e.g., disulfide bond, crosslink) between two residues."""

    start_element: BaseStructureSceneElement
    end_element: BaseStructureSceneElement

    def __init__(
        self,
        start_element: BaseStructureSceneElement,
        end_element: BaseStructureSceneElement,
        style: Optional[ConnectionStyle] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        self.start_element = start_element
        self.end_element = end_element
        super().__init__(
            id=f"connection_{start_element.id}_{end_element.id}",
            style=style,
            parent=parent,
        )

    def get_depth(self, structure: Structure) -> float | None:
        return (
            self.start_element.get_depth(structure)
            + self.end_element.get_depth(structure) / 2
        )

    def __str__(self) -> str:
        return f"Connection({self.type}: {self.start_element} <-> {self.end_element})"

    def __repr__(self) -> str:
        return f"Connection(start_element={self.start_element!r}, end_element={self.end_element!r}, type='{self.type}', style={self.style!r})"

    @property
    def default_style(self) -> ConnectionStyle:
        return ConnectionStyle()
