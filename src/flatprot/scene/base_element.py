# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

from pydantic import BaseModel, Field

from flatprot.core.structure import Structure

# Forward declaration for type hinting cycle
# Using TypeVar for forward reference to SceneGroup which should inherit from BaseSceneElement
SceneGroupType = TypeVar("SceneGroupType", bound="BaseSceneElement")

# Generic Type Variable for the Style
StyleType = TypeVar("StyleType", bound="BaseSceneStyle")


class BaseSceneStyle(BaseModel):
    """Base class for all scene element style definitions using Pydantic."""

    visibility: bool = Field(
        default=True, description="Whether the element is visible."
    )
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Opacity of the element (0.0 to 1.0)."
    )
    # Add other common style attributes here if needed later

    model_config = {"extra": "forbid"}  # Forbid extra fields


class BaseSceneElement(ABC, Generic[StyleType]):
    """Abstract base class for all elements within a scene graph.

    This class is generic and requires a specific StyleType that inherits
    from BaseSceneStyle.
    """

    def __init__(
        self,
        id: str,
        style: Optional[StyleType] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a BaseSceneElement.

        Args:
            id: A unique identifier for this scene element.
            style: A style instance for this element.
            parent: The parent SceneGroup in the scene graph, if any.
        """
        if not isinstance(id, str) or not id:
            raise ValueError("SceneElement ID must be a non-empty string.")

        self.id: str = id
        self._style: Optional[StyleType] = style or self.default_style
        self._parent: Optional[SceneGroupType] = parent

    @property
    def parent(self) -> Optional[SceneGroupType]:
        """Get the parent group of this element."""
        return self._parent

    # Keep internal setter for parent relationship management by Scene/SceneGroup
    def _set_parent(self, value: Optional[SceneGroupType]) -> None:
        """Internal method to set the parent group. Should be called by Scene/SceneGroup."""
        # Basic type check, assumes SceneGroup will inherit from BaseSceneElement
        if value is not None and not isinstance(value, BaseSceneElement):
            # A more specific check like isinstance(value, SceneGroup) would be ideal
            # but causes circular dependency issues without careful structuring or protocols.
            # This provides a basic safeguard. We expect SceneGroup to inherit BaseSceneElement.
            raise TypeError(
                "Parent must be a SceneGroup (subclass of BaseSceneElement)."
            )
        self._parent = value

    @property
    @abstractmethod
    def default_style(self) -> StyleType:
        """Provides the default style instance for this element type.

        Subclasses must implement this property.

        Returns:
            An instance of the specific StyleType for this element.
        """
        raise NotImplementedError

    @property
    def style(self) -> StyleType:
        """Get the effective style for this element (instance-specific or default)."""
        return self._style if self._style is not None else self.default_style()

    def update_style(self, new_style: StyleType) -> None:
        """Update the instance-specific style of this element.

        Args:
            new_style: The new style object to apply.
        """
        # Ensure the provided style is compatible
        # Note: isinstance check might be too strict if subclasses of the style are allowed.
        # Adjust check if necessary based on desired style inheritance behavior.
        expected_style_type = self.default_style().__class__
        if not isinstance(new_style, expected_style_type):
            raise TypeError(
                f"Invalid style type. Expected {expected_style_type.__name__}, "
                f"got {type(new_style).__name__}."
            )
        self._style = new_style

    @abstractmethod
    def get_depth(self, structure: Structure) -> Optional[float]:
        """Calculate or retrieve the representative depth for Z-ordering.

        Depth should typically be derived from the pre-projected coordinates
        (column 2) in the provided structure object.
        Lower values are typically closer to the viewer.

        Args:
            structure: The core Structure object containing pre-projected
                       2D + Depth coordinate data.

        Returns:
            A float representing the depth, or None if depth cannot be determined
            or is not applicable (e.g., for groups).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Provide a string representation of the scene element."""
        parent_id = f"'{self._parent.id}'" if self._parent else None
        style_source = "default" if self._style is None else "instance"
        # Safely get range representation, default to 'N/A' if not present
        range_repr = str(getattr(self, "residue_range_set", "N/A"))
        range_str = f" range='{range_repr}'" if range_repr != "N/A" else ""
        target_repr = str(getattr(self, "target", "N/A"))
        target_str = f" target='{target_repr}'" if target_repr != "N/A" else ""

        return (
            f"<{self.__class__.__name__} id='{self.id}'{range_str}{target_str} "
            f"parent={parent_id} style_source={style_source}>"
        )
