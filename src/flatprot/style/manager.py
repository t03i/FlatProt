# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

from .base import Style, CanvasStyle
from .types import StyleType
from .structure import HelixStyle, SheetStyle, CoilStyle, ElementStyle
from .annotation import (
    AnnotationStyle,
    AreaAnnotationStyle,
    PointAnnotationStyle,
    LineAnnotationStyle,
)

STYLE_MAP = {
    StyleType.HELIX: HelixStyle,
    StyleType.SHEET: SheetStyle,
    StyleType.COIL: CoilStyle,
    StyleType.CANVAS: CanvasStyle,
    StyleType.ANNOTATION: AnnotationStyle,
    StyleType.ELEMENT: ElementStyle,
    StyleType.AREA_ANNOTATION: AreaAnnotationStyle,
    StyleType.POINT_ANNOTATION: PointAnnotationStyle,
    StyleType.LINE_ANNOTATION: LineAnnotationStyle,
}


@dataclass
class StyleManager:
    """Manages styles for protein visualization elements"""

    # Predefined themes/presets
    DEFAULT_THEME: ClassVar[dict[str, dict]] = {
        StyleType.ELEMENT.value: {
            "stroke_color": "#333333",
        },
        StyleType.HELIX.value: {
            "fill_color": "#FF4444",
        },
        StyleType.SHEET.value: {
            "fill_color": "#4444FF",
        },
        StyleType.COIL.value: {
            "stroke_width_factor": 1.0,
        },
        StyleType.CANVAS.value: {
            "width": 1024,
            "height": 1024,
            "background_color": "#FFFFFF",
            "background_opacity": 0,
            "padding_x": 0.05,
            "padding_y": 0.05,
            "maintain_aspect_ratio": True,
        },
        StyleType.ANNOTATION.value: {
            "fill_color": "#000000",
            "opacity": 1.0,
            "connector_color": "#666666",
            "connector_width": 1,
            "connector_opacity": 0.6,
        },
    }

    _element_styles: dict[StyleType, Style] = field(
        default_factory=lambda: {
            stype: style_class() for stype, style_class in STYLE_MAP.items()
        }
    )

    @classmethod
    def create_default(cls) -> "StyleManager":
        """Create a StyleManager with default settings"""
        return cls.from_dict(cls.DEFAULT_THEME)

    @classmethod
    def from_dict(cls, config: dict) -> "StyleManager":
        """Create a StyleManager from a configuration dictionary"""
        manager = cls()

        # Apply global style if present
        if StyleType.ELEMENT.value in config:
            manager._element_styles[StyleType.ELEMENT] = ElementStyle.model_validate(
                config[StyleType.ELEMENT.value]
            )

        # Apply element-specific styles
        for element_type, style_class in STYLE_MAP.items():
            key = element_type.value
            if key in config:
                style = style_class.model_validate(config[key])
                manager._element_styles[element_type] = style

        return manager

    def get_style(self, element_type: StyleType) -> Style:
        """Get the style for a specific element type, applying global overrides"""
        base_style = self._element_styles[element_type]
        if self._element_styles[StyleType.ELEMENT]:
            return base_style.model_copy(
                update=self._element_styles[StyleType.ELEMENT].model_dump(
                    exclude_unset=True
                )
            )
        return base_style

    def update_style(self, element_type: StyleType, **style_kwargs) -> None:
        """Update style for a specific element type"""
        current_style = self._element_styles[element_type]
        self._element_styles[element_type] = current_style.model_copy(
            update=style_kwargs
        )

    @classmethod
    def available_themes(cls) -> list[str]:
        """Get list of available predefined themes"""
        return list(cls.THEMES.keys())

    def save_as_theme(self, theme_name: str) -> None:
        """Save current style configuration as a new theme"""
        config = {}

        if self.global_style:
            config["global_style"] = self.global_style.model_dump(exclude_unset=True)

        for element_type, style in self.element_styles.items():
            config[element_type.value] = style.model_dump(exclude_unset=True)

        self.THEMES[theme_name] = config
