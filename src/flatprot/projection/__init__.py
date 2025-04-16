# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .orthographic import OrthographicProjection, OrthographicProjectionParameters
from .base import BaseProjection, BaseProjectionParameters
from flatprot.core.errors import FlatProtError

__all__ = [
    "OrthographicProjection",
    "OrthographicProjectionParameters",
    "BaseProjection",
    "BaseProjectionParameters",
    "ProjectionError",
]


# Projection-related errors
class ProjectionError(FlatProtError):
    """Exception raised when a projection operation fails."""

    def __init__(self, message: str):
        super().__init__(f"Projection error: {message}")
