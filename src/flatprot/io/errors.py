# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Error classes for the FlatProt IO module."""

from typing import Optional
from rich.console import Console

from flatprot.core import FlatProtError

console = Console()


# Base IO error classes
class IOError(FlatProtError):
    """Base class for all IO-related errors in FlatProt."""

    def __init__(self, message: str):
        super().__init__(f"IO error: {message}")


class FileError(IOError):
    """Base class for file-related errors."""

    def __init__(self, message: str):
        super().__init__(message)


class FileNotFoundError(FileError):
    """Exception raised when a required file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"File not found: {file_path}"
        suggestion = "Please check that the file exists and the path is correct."
        super().__init__(f"{message}\n{suggestion}")


class InvalidFileFormatError(FileError):
    """Exception raised when a file has an invalid format."""

    def __init__(self, file_path: str, expected_format: str):
        self.file_path = file_path
        self.expected_format = expected_format
        message = f"Invalid file format for {file_path}. Expected {expected_format}."
        super().__init__(message)


# Structure-related errors
class StructureError(IOError):
    """Base class for structure-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Structure error: {message}")


class StructureFileNotFoundError(StructureError):
    """Exception raised when a structure file is not found."""

    def __init__(self, file_path: str):
        message = f"Structure file not found: {file_path}"
        suggestion = "Please check that the file exists and the path is correct."
        super().__init__(f"{message}\n{suggestion}")


class InvalidStructureError(StructureError):
    """Exception raised when a structure file has an invalid format."""

    def __init__(
        self, file_path: str, expected_format: str, details: Optional[str] = None
    ):
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

        super().__init__(f"{message}\n{suggestion}")


# Annotation-related errors
class AnnotationError(IOError):
    """Base class for annotation-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Annotation error: {message}")


class AnnotationFileNotFoundError(AnnotationError):
    """Exception raised when an annotation file is not found."""

    def __init__(self, file_path: str):
        message = f"Annotation file not found: {file_path}"
        super().__init__(message)


class MalformedAnnotationError(AnnotationError):
    """Exception raised when an annotation file has an invalid format."""

    def __init__(self, file_path: str, details: str):
        message = f"Malformed annotation file: {file_path}\n{details}"
        super().__init__(message)


class MissingRequiredFieldError(AnnotationError):
    """Exception raised when a required field is missing from an annotation."""

    def __init__(self, annotation_type: str, field_name: str, annotation_index: int):
        message = f"Missing required field '{field_name}' for {annotation_type} annotation at index {annotation_index}"
        super().__init__(message)


class InvalidFieldTypeError(AnnotationError):
    """Exception raised when a field has an invalid type."""

    def __init__(
        self,
        annotation_type: str,
        field_name: str,
        expected_type: str,
        annotation_index: int,
    ):
        message = f"Invalid type for field '{field_name}' in {annotation_type} annotation at index {annotation_index}. Expected {expected_type}."
        super().__init__(message)


class InvalidReferenceError(AnnotationError):
    """Exception raised when an annotation references a nonexistent chain or residue."""

    def __init__(
        self,
        annotation_type: str,
        reference_type: str,
        reference: str,
        annotation_index: int,
    ):
        message = f"Invalid {reference_type} reference '{reference}' in {annotation_type} annotation at index {annotation_index}."
        super().__init__(message)


class AnnotationFileError(AnnotationError):
    """Exception raised when there's an issue with an annotation file."""

    def __init__(self, file_path: str, details: Optional[str] = None):
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
class StyleError(IOError):
    """Base class for style-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Style error: {message}")


class StyleFileNotFoundError(StyleError):
    """Exception raised when a style file is not found."""

    def __init__(self, file_path: str):
        message = f"Style file not found: {file_path}"
        super().__init__(message)


class StyleParsingError(StyleError):
    """Base error for style parsing issues."""

    pass


class InvalidTomlError(StyleParsingError):
    """Error for malformed TOML files."""

    pass


class StyleValidationError(StyleParsingError):
    """Error for invalid style field types or values."""

    pass


class StyleFileError(StyleError):
    """Exception raised when there's an issue with a style file."""

    def __init__(self, file_path: str, details: Optional[str] = None):
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


# Matrix-related errors
class MatrixError(IOError):
    """Base class for matrix-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Matrix error: {message}")


class MatrixFileNotFoundError(MatrixError):
    """Exception raised when a matrix file is not found."""

    def __init__(self, file_path: str):
        message = f"Matrix file not found: {file_path}"
        super().__init__(message)


class InvalidMatrixDimensionsError(MatrixError):
    """Error raised when a matrix has invalid dimensions."""

    pass


class InvalidMatrixFormatError(MatrixError):
    """Error raised when a matrix file has an invalid format."""

    pass


class MatrixFileError(MatrixError):
    """Error raised when a matrix file can't be read."""

    pass


class InvalidMatrixError(MatrixError):
    """Exception raised when a matrix file has an invalid format."""

    def __init__(self, file_path: str, details: Optional[str] = None):
        self.file_path = file_path
        self.details = details

        message = f"Invalid matrix file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = "\nMatrix files should be NumPy .npy files containing a 4x4 transformation matrix."
        suggestion += "\nAlternatively, you can use separate .npy files for rotation (3x3) and translation (3x1)."

        super().__init__(f"{message}{suggestion}")


# Output-related errors
class OutputError(IOError):
    """Base class for output-related errors."""

    def __init__(self, message: str):
        super().__init__(f"Output error: {message}")


class OutputFileError(OutputError):
    """Exception raised when there's an issue with an output file."""

    def __init__(self, file_path: str, details: Optional[str] = None):
        self.file_path = file_path
        self.details = details

        message = f"Error writing to output file: {file_path}"
        if details:
            message += f"\n{details}"

        suggestion = "\nPlease check that you have write permissions to the directory and sufficient disk space."

        super().__init__(f"{message}{suggestion}")
