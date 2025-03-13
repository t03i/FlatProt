# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Parser for annotation files in TOML format."""

from pathlib import Path
from typing import List, Union, Literal

import toml
from pydantic import BaseModel, field_validator, ConfigDict, ValidationError
from rich.console import Console

from .errors import (
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    InvalidReferenceError,
    AnnotationError,
)
from flatprot.scene import PointAnnotation, LineAnnotation, AreaAnnotation, Scene
from flatprot.style import StyleManager

console = Console()


# Pydantic models for validation


class PointAnnotationData(BaseModel):
    """Model for point annotation data in TOML files."""

    label: str
    type: Literal["point"]
    index: int
    chain: str

    model_config = ConfigDict(frozen=True)


class LineAnnotationData(BaseModel):
    """Model for line/pair annotation data in TOML files."""

    label: str
    type: Literal["line", "pair"]
    indices: List[int]
    chain: str

    @field_validator("indices", mode="after")
    @classmethod
    def validate_indices(cls, v):
        if len(v) != 2:
            raise ValueError("Line/pair annotation must have exactly 2 indices")
        return v

    model_config = ConfigDict(frozen=True)


class AreaRange(BaseModel):
    """Model for area range data in TOML files."""

    start: int
    end: int

    @field_validator("end", mode="after")
    @classmethod
    def validate_end(cls, v, values):
        if "start" in values.data and v < values.data["start"]:
            raise ValueError("End must be greater than or equal to start")
        return v


class AreaAnnotationData(BaseModel):
    """Model for area annotation data in TOML files."""

    label: str
    type: Literal["area"]
    range: AreaRange
    chain: str

    model_config = ConfigDict(frozen=True)


class AnnotationsFile(BaseModel):
    """Model for the entire annotations file."""

    annotations: List[
        Union[PointAnnotationData, LineAnnotationData, AreaAnnotationData]
    ]

    model_config = ConfigDict(
        json_discriminator="type",
        json_schema_extra={
            "discriminator": {
                "propertyName": "type",
                "mapping": {
                    "point": "PointAnnotationData",
                    "line": "LineAnnotationData",
                    "pair": "LineAnnotationData",
                    "area": "AreaAnnotationData",
                },
            }
        },
    )


class AnnotationParser:
    """Parser for annotation files in TOML format."""

    def __init__(
        self,
        file_path: Path,
        scene: Scene = None,
        style_manager: StyleManager = None,
    ):
        """Initialize the parser with a file path and  scene.

        Args:
            file_path: Path to the TOML file containing annotations
            scene: Scene object to map annotations to

        Raises:
            AnnotationFileNotFoundError: If the file does not exist
        """
        self.file_path = file_path
        self.scene = scene
        self.style_manager = style_manager or StyleManager.create_default()

        # Check file existence
        if not file_path.exists():
            raise AnnotationFileNotFoundError(str(file_path))

    def parse(self) -> List[Union[PointAnnotation, LineAnnotation, AreaAnnotation]]:
        """Parse the annotation file and create annotation objects.

        Returns:
            List of annotation objects

        Raises:
            MalformedAnnotationError: If the TOML file is malformed or missing required structure
            InvalidReferenceError: If an annotation references a nonexistent chain or residue
        """
        # If no scene provided, we just validate the TOML but can't create objects
        if self.scene is None:
            raise AnnotationError("Scene is required to create annotation objects")

        try:
            # Parse TOML content
            content = toml.load(self.file_path)

            # Validate using Pydantic
            annotations_file = AnnotationsFile.model_validate(content)

        except toml.TomlDecodeError as e:
            raise MalformedAnnotationError(
                str(self.file_path), f"Invalid TOML syntax: {str(e)}"
            )
        except ValidationError as e:
            raise MalformedAnnotationError(str(self.file_path), str(e))

        # Create annotation objects
        annotation_objects = []

        for i, data in enumerate(annotations_file.annotations):
            # Validate scene references and create annotation objects
            if isinstance(data, PointAnnotationData):
                annotation_objects.append(
                    self._validate_and_create_point_annotation(data, i)
                )
            elif isinstance(data, LineAnnotationData):
                annotation_objects.append(
                    self._validate_and_create_line_annotation(data, i)
                )
            elif isinstance(data, AreaAnnotationData):
                annotation_objects.append(
                    self._validate_and_create_area_annotation(data, i)
                )

        return annotation_objects

    def _validate_and_create_point_annotation(
        self, data: PointAnnotationData, index: int
    ) -> PointAnnotation:
        """Validate and create a point annotation object.

        Args:
            data: Annotation data
            index: Index of the annotation in the file

        Returns:
            PointAnnotation object

        Raises:
            InvalidReferenceError: If the reference doesn't exist in the scene
        """
        elements = self.scene.get_elements_for_residue(data.chain, data.index)
        element_idx = self.scene.get_element_index_from_global_index(
            data.index, elements[0]
        )
        if not elements:
            raise InvalidReferenceError(
                "point", "residue", f"{data.index} in chain {data.chain}", index
            )

        return PointAnnotation(
            label=data.label,
            targets=[elements[0]],
            indices=[element_idx],
            style_manager=self.style_manager,
        )

    def _validate_and_create_line_annotation(
        self, data: LineAnnotationData, index: int
    ) -> LineAnnotation:
        """Validate and create a line annotation object.

        Args:
            data: Annotation data
            index: Index of the annotation in the file

        Returns:
            LineAnnotation object

        Raises:
            InvalidReferenceError: If the references don't exist in the scene
        """
        targets = []
        for idx in data.indices:
            elements = self.scene.get_elements_for_residue(data.chain, idx)
            element_idx = self.scene.get_element_index_from_global_index(
                idx, elements[0]
            )
            if not elements:
                raise InvalidReferenceError(
                    "line/pair", "residue", f"{idx} in chain {data.chain}", index
                )
            # Store the first element for the residue
            targets.append((elements[0], element_idx))

        return LineAnnotation(
            label=data.label,
            targets=[t for t, _ in targets],
            indices=[i for _, i in targets],
            style_manager=self.style_manager,
        )

    def _validate_and_create_area_annotation(
        self, data: AreaAnnotationData, index: int
    ) -> AreaAnnotation:
        """Validate and create an area annotation object.

        Args:
            data: Annotation data
            index: Index of the annotation in the file

        Returns:
            AreaAnnotation object

        Raises:
            InvalidReferenceError: If the reference range doesn't exist in the scene
        """
        elements = self.scene.get_elements_for_residue_range(
            data.chain, data.range.start, data.range.end
        )
        if not elements:
            raise InvalidReferenceError(
                "area",
                "residue range",
                f"{data.range.start}-{data.range.end} in chain {data.chain}",
                index,
            )

        return AreaAnnotation(
            label=data.label,
            targets=elements,
            style_manager=self.style_manager,
        )
