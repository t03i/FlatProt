# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .scene import Scene
from .connection import Connection, ConnectionStyle
from .base_element import BaseSceneElement, BaseSceneStyle
from .group import SceneGroup, GroupTransform, GroupStyle
from .structure import (
    HelixSceneElement,
    HelixStyle,
    SheetSceneElement,
    SheetStyle,
    CoilSceneElement,
    CoilStyle,
    BaseStructureSceneElement,
    BaseStructureStyle,
)
from .annotation import (
    BaseAnnotationElement,
    BaseAnnotationStyle,
    PointAnnotation,
    PointAnnotationStyle,
    LineAnnotation,
    LineAnnotationStyle,
    AreaAnnotation,
    AreaAnnotationStyle,
)
from .errors import (
    SceneError,
    SceneCreationError,
    SceneAnnotationError,
    ElementNotFoundError,
    DuplicateElementError,
    ParentNotFoundError,
    ElementTypeError,
    CircularDependencyError,
    SceneGraphInconsistencyError,
    InvalidSceneOperationError,
    TargetResidueNotFoundError,
)

from .resolver import CoordinateResolver

__all__ = [
    "CoordinateResolver",
    "Connection",
    "ConnectionStyle",
    "Scene",
    "SceneGroup",
    "GroupTransform",
    "GroupStyle",
    "BaseSceneElement",
    "BaseSceneStyle",
    "BaseStructureSceneElement",
    "BaseStructureStyle",
    "HelixSceneElement",
    "HelixStyle",
    "SheetSceneElement",
    "SheetStyle",
    "CoilSceneElement",
    "CoilStyle",
    "BaseAnnotationElement",
    "BaseAnnotationStyle",
    "PointAnnotation",
    "PointAnnotationStyle",
    "LineAnnotation",
    "LineAnnotationStyle",
    "AreaAnnotation",
    "AreaAnnotationStyle",
    "SceneError",
    "SceneCreationError",
    "SceneAnnotationError",
    "ElementNotFoundError",
    "DuplicateElementError",
    "ParentNotFoundError",
    "ElementTypeError",
    "CircularDependencyError",
    "SceneGraphInconsistencyError",
    "InvalidSceneOperationError",
    "TargetResidueNotFoundError",
]
