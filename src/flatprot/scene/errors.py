# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from flatprot.core import FlatProtError, Structure, ResidueCoordinate


class SceneError(FlatProtError):
    """Base class for all errors in the scene module."""

    pass


class SceneAnnotationError(SceneError):
    """Error related to scene annotations."""

    pass


class ElementNotFoundError(SceneError):
    """Raised when a scene element is not found, typically by ID."""

    pass


class DuplicateElementError(SceneError):
    """Raised when attempting to add an element that already exists."""

    pass


class ParentNotFoundError(ElementNotFoundError):  # Inherits as it's a specific case
    """Raised when a specified parent element ID is not found."""

    pass


class ElementTypeError(SceneError, TypeError):  # Inherit TypeError for type context
    """Raised when an element is not of the expected type (e.g., expecting SceneGroup)."""

    pass


class CircularDependencyError(SceneError, ValueError):  # Inherit ValueError for context
    """Raised when an operation would create a circular parent-child relationship."""

    pass


class SceneGraphInconsistencyError(SceneError, RuntimeError):  # Inherit RuntimeError
    """Raised when an internal inconsistency in the scene graph state is detected."""

    pass


class InvalidSceneOperationError(SceneError, ValueError):  # Inherit ValueError
    """Raised for operations that are invalid given the current element state (e.g., adding already parented element)."""

    pass


class SceneCreationError(SceneError):
    """Raised when creation of a scene fails."""

    pass


class TargetResidueNotFoundError(SceneError):
    """Error raised when a target residue is not found in the structure."""

    def __init__(self, structure: Structure, residue: ResidueCoordinate):
        message = f"Residue {residue} not found in structure {structure}"
        super().__init__(message)
