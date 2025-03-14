# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Input/output utilities for FlatProt."""

from typing import List, Optional
from pathlib import Path

from flatprot.io.annotations import AnnotationParser
from flatprot.io.styles import StyleParser
from flatprot.io.matrix import MatrixLoader
from flatprot.io.structure import validate_structure_file
from flatprot.io.structure_gemmi_adapter import GemmiStructureParser
from flatprot.io.errors import (
    # Base classes
    IOError,
    FileError,
    FileNotFoundError,
    InvalidFileFormatError,
    # Structure errors
    StructureError,
    StructureFileNotFoundError,
    InvalidStructureError,
    # Annotation errors
    AnnotationError,
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    MissingRequiredFieldError,
    InvalidFieldTypeError,
    InvalidReferenceError,
    AnnotationFileError,
    # Style errors
    StyleError,
    StyleFileNotFoundError,
    StyleParsingError,
    InvalidTomlError,
    StyleValidationError,
    StyleFileError,
    InvalidColorError,
    # Matrix errors
    MatrixError,
    MatrixFileNotFoundError,
    InvalidMatrixDimensionsError,
    InvalidMatrixFormatError,
    MatrixFileError,
    InvalidMatrixError,
    # Output errors
    OutputError,
    OutputFileError,
)


__all__ = [
    # Classes
    "AnnotationParser",
    "StyleParser",
    "GemmiStructureParser",
    "MatrixLoader",
    # Functions
    "validate_optional_files",
    "validate_structure_file",
    # Errors
    "IOError",
    "FileError",
    "FileNotFoundError",
    "InvalidFileFormatError",
    "StructureError",
    "StructureFileNotFoundError",
    "InvalidStructureError",
    "AnnotationError",
    "AnnotationFileNotFoundError",
    "MalformedAnnotationError",
    "MissingRequiredFieldError",
    "InvalidFieldTypeError",
    "InvalidReferenceError",
    "AnnotationFileError",
    "StyleError",
    "StyleFileNotFoundError",
    "StyleParsingError",
    "InvalidTomlError",
    "StyleValidationError",
    "StyleFileError",
    "InvalidColorError",
    "MatrixError",
    "MatrixFileNotFoundError",
    "InvalidMatrixDimensionsError",
    "InvalidMatrixFormatError",
    "MatrixFileError",
    "InvalidMatrixError",
    "OutputError",
    "OutputFileError",
]


def validate_optional_files(
    file_paths: List[Optional[Path]],
) -> None:
    """Validate that optional files exist if specified.

    Args:
        file_paths: List of file paths to check (can include None values)

    Raises:
        FileNotFoundError: If any specified file does not exist
    """
    for path in file_paths:
        if path and not path.exists():
            raise FileNotFoundError(str(path))
