# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


from flatprot.core import FlatProtError


class AlignmentError(FlatProtError):
    """Base class for alignment-related errors."""

    pass


class NoSignificantAlignmentError(AlignmentError):
    """Raised when no significant alignment is found."""

    pass


class DatabaseEntryNotFoundError(AlignmentError):
    """Raised when a database entry is not found."""

    pass
