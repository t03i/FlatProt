# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Error classes for the FlatProt CLI."""

from rich.console import Console

console = Console()


class FlatProtCLIError(Exception):
    """Base exception class for FlatProt CLI errors.

    This class is used as the base for all custom exceptions raised by the CLI.
    It provides a consistent interface for error handling and formatting.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class FileNotFoundError(FlatProtCLIError):
    """Exception raised when a required file is not found."""

    def __init__(self, file_path: str):
        message = f"File not found: {file_path}"
        super().__init__(message)


class InvalidFileFormatError(FlatProtCLIError):
    """Exception raised when a file has an invalid format."""

    def __init__(self, file_path: str, expected_format: str):
        message = f"Invalid file format for {file_path}. Expected {expected_format}."
        super().__init__(message)


class InvalidStructureError(FlatProtCLIError):
    """Exception raised when a structure file has an invalid format."""

    def __init__(self, file_path: str, expected_format: str, details: str = None):
        message = f"Invalid {expected_format} file: {file_path}"
        if details:
            message += f"\n{details}"
        super().__init__(message)


class TransformationError(FlatProtCLIError):
    """Exception raised when a transformation operation fails."""

    def __init__(self, message: str):
        super().__init__(f"Transformation error: {message}")
