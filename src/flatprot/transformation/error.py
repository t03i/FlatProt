# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from flatprot.core import FlatProtError


# Transformation-related errors
class TransformationError(FlatProtError):
    """Exception raised when a transformation operation fails."""

    def __init__(self, message: str):
        super().__init__(f"Transformation error: {message}")
