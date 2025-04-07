# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .orthographic import OrthographicProjector, OrthographicProjectionParameters
from .base import Projector, BaseProjectionParameters
from flatprot.core.error import FlatProtError

__all__ = [
    "OrthographicProjector",
    "OrthographicProjectionParameters",
    "Projector",
    "BaseProjectionParameters",
    "ProjectionError",
]


# Projection-related errors
class ProjectionError(FlatProtError):
    """Exception raised when a projection operation fails."""

    def __init__(self, message: str):
        super().__init__(f"Projection error: {message}")
