# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import numpy as np

from ..elements import SceneElement, SceneGroup
from flatprot.style import StyleManager, StyleType


class Annotation(SceneElement):
    """An annotation connecting one or more scene elements with content."""

    def __init__(
        self,
        content: Any,
        indices: list[int],
        targets: list[SceneElement],
        metadata: dict = {},
        style_manager: Optional[StyleManager] = None,
        style_type: Optional[StyleType] = None,
    ):
        # Get span from targets

        super().__init__(
            metadata=metadata,
            style_manager=style_manager,
            style_type=style_type,
        )
        self.content = content
        self.indices = indices
        self.targets = targets

    def display_coordinates(self) -> Optional[np.ndarray]:
        return None


class GroupAnnotation(SceneGroup):
    """A group that annotates its contained elements."""

    def __init__(
        self,
        annotation_type: str,
        content: Any,
        elements: list[SceneElement],
        metadata: dict = {},
        style_manager: Optional[StyleManager] = None,
        style_type: Optional[StyleType] = None,
    ):
        # Get span from elements
        start = min(e._start for e in elements if hasattr(e, "_start"))
        end = max(e._end for e in elements if hasattr(e, "_end"))
        chain_id = elements[0]._chain_id if hasattr(elements[0], "_chain_id") else ""

        super().__init__(
            id=f"{annotation_type}_{chain_id}_{start}_{end}",
            metadata=metadata,
            style_manager=style_manager,
            style_type=style_type,
        )
        self.annotation_type = annotation_type
        self.content = content

        # Add elements to group
        for element in elements:
            self.add_element(element)
