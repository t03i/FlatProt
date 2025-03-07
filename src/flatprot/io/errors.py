# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console

console = Console()


class AnnotationError(Exception):
    """Base exception class for annotation parsing errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


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
