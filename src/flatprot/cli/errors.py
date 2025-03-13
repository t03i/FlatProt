# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Error classes for the FlatProt CLI."""

import functools
import os
import sys
import traceback
from typing import Any, Callable, TypeVar, cast

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

console = Console()

F = TypeVar("F", bound=Callable[..., Any])


class FlatProtCLIError(Exception):
    """Base exception class for FlatProt CLI errors.

    This class is used as the base for all custom exceptions raised by the CLI.
    It provides a consistent interface for error handling and formatting.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# Structure-related errors
class StructureError(FlatProtCLIError):
    """Base class for structure-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Structure error: {message}")


class FileNotFoundError(FlatProtCLIError):
    """Exception raised when a required file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"File not found: {file_path}"
        suggestion = "Please check that the file exists and the path is correct."
        super().__init__(f"{message}\n{suggestion}")


class InvalidFileFormatError(FlatProtCLIError):
    """Exception raised when a file has an invalid format."""

    def __init__(self, file_path: str, expected_format: str):
        self.file_path = file_path
        self.expected_format = expected_format
        message = f"Invalid file format for {file_path}. Expected {expected_format}."
        super().__init__(message)


class InvalidStructureError(StructureError):
    """Exception raised when a structure file has an invalid format."""

    def __init__(self, file_path: str, expected_format: str, details: str = None):
        self.file_path = file_path
        self.expected_format = expected_format
        self.details = details

        message = f"Invalid {expected_format} file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = f"\nPlease ensure the file is a valid {expected_format} format. "
        if expected_format == "PDB":
            suggestion += "PDB files should contain ATOM, HETATM, or HEADER records."
        elif expected_format == "CIF":
            suggestion += "CIF files should contain _atom_site categories, loop_, or data_ sections."

        super().__init__(f"{message}{suggestion}")


class TransformationError(FlatProtCLIError):
    """Exception raised when a transformation operation fails."""

    def __init__(self, message: str):
        super().__init__(f"Transformation error: {message}")


# Annotation-related errors
class AnnotationError(FlatProtCLIError):
    """Base class for annotation-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Annotation error: {message}")


class AnnotationFileError(AnnotationError):
    """Exception raised when there's an issue with an annotation file."""

    def __init__(self, file_path: str, details: str = None):
        self.file_path = file_path
        self.details = details

        message = f"Invalid annotation file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = (
            "\nAnnotation files should be in TOML format with an 'annotations' list."
        )
        suggestion += "\nExample annotation format:\n"
        suggestion += """
        [[annotations]]
        type = "point"
        label = "Active site"
        chain = "A"
        indices = 123
        color = "#FF0000"
        """

        super().__init__(f"{message}{suggestion}")


# Style-related errors
class StyleError(FlatProtCLIError):
    """Base class for style-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Style error: {message}")


class StyleFileError(StyleError):
    """Exception raised when there's an issue with a style file."""

    def __init__(self, file_path: str, details: str = None):
        self.file_path = file_path
        self.details = details

        message = f"Invalid style file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = "\nStyle files should be in TOML format with sections for different elements."
        suggestion += "\nExample style format:\n"
        suggestion += """
        [helix]
        fill_color = "#FF0000"
        stroke_color = "#800000"

        [sheet]
        fill_color = "#00FF00"
        stroke_color = "#008000"
        """

        super().__init__(f"{message}{suggestion}")


class InvalidColorError(StyleError):
    """Exception raised when an invalid color is specified."""

    def __init__(self, color_value: str, element_type: str):
        self.color_value = color_value
        self.element_type = element_type

        message = f"Invalid color value '{color_value}' for {element_type}."
        suggestion = "\nColors should be specified as hex (#RRGGBB), RGB (rgb(r,g,b)), or named colors."

        super().__init__(f"{message}{suggestion}")


# Output-related errors
class OutputError(FlatProtCLIError):
    """Base class for output-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Output error: {message}")


class OutputFileError(OutputError):
    """Exception raised when there's an issue with an output file."""

    def __init__(self, file_path: str, details: str = None):
        self.file_path = file_path
        self.details = details

        message = f"Error writing to output file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = "\nPlease check that you have write permissions to the directory and sufficient disk space."

        super().__init__(f"{message}{suggestion}")


# Matrix-related errors
class MatrixError(FlatProtCLIError):
    """Base class for matrix-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Matrix error: {message}")


class InvalidMatrixError(MatrixError):
    """Exception raised when a matrix file has an invalid format."""

    def __init__(self, file_path: str, details: str = None):
        self.file_path = file_path
        self.details = details

        message = f"Invalid matrix file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = "\nMatrix files should be NumPy .npy files containing a 4x4 transformation matrix."
        suggestion += "\nAlternatively, you can use separate .npy files for rotation (3x3) and translation (3x1)."

        super().__init__(f"{message}{suggestion}")


def error_handler(func: F) -> F:
    """Decorator to handle exceptions in CLI functions.

    This decorator catches exceptions raised by the decorated function and
    formats error messages using rich formatting. It also provides suggestions
    for fixing common issues.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FlatProtCLIError as e:
            # Get stack trace information
            tb = sys.exc_info()[2]
            frame = traceback.extract_tb(tb)[-1]
            filename = os.path.basename(frame.filename)
            lineno = frame.lineno

            # Create rich text for the error message
            title = Text("FlatProt Error", style="bold red")
            error_type = Text(f"[{e.__class__.__name__}]", style="red")
            location = Text(f" at {filename}:{lineno}", style="dim")
            header = Text.assemble(title, " ", error_type, location)

            # Create panel with the error message
            panel = Panel(
                Text(e.message), title=header, border_style="red", padding=(1, 2)
            )

            # Print the error panel
            console.print(panel)
            return 1
        except Exception as e:
            # For unexpected exceptions, show more detailed traceback
            console.print("[bold red]Unexpected Error:[/bold red]", str(e))
            console.print(Traceback())
            return 1

    return cast(F, wrapper)
