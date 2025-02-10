# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, ClassVar

from .base import ElementStyle
from .types import StyleType
from .structure import HelixStyle, SheetStyle, CoilStyle

STYLE_MAP = {
    StyleType.HELIX: HelixStyle,
    StyleType.SHEET: SheetStyle,
    StyleType.COIL: CoilStyle,
}


@dataclass
class StyleManager:
    """Manages styles for protein visualization elements"""

    # Predefined themes/presets
    DEFAULT_THEME: ClassVar[dict[str, dict]] = {
        "default": {
            "global_style": {
                "stroke_color": "#333333",
                "opacity": 1.0,
                "line_width": 3.0,
            },
            StyleType.HELIX.value: {
                "fill_color": "#FF4444",
                "wave_height_factor": 0.8,
                "ribbon_thickness_factor": 0.6,
            },
            StyleType.SHEET.value: {
                "fill_color": "#4444FF",
                "ribbon_thickness_factor": 1.0,
                "arrow_width_factor": 1.5,
            },
            StyleType.COIL.value: {
                "stroke_width_factor": 1.0,
            },
        },
    }

    element_styles: dict[StyleType, ElementStyle] = field(
        default_factory=lambda: {
            stype: style_class() for stype, style_class in STYLE_MAP.items()
        }
    )
    global_style: Optional[ElementStyle] = None

    @classmethod
    def create_default(cls) -> "StyleManager":
        """Create a StyleManager with default settings"""
        return cls.from_theme("default")

    @classmethod
    def from_theme(cls, theme_name: str) -> "StyleManager":
        """Create a StyleManager from a predefined theme"""
        if theme_name not in cls.THEMES:
            raise ValueError(
                f"Unknown theme: {theme_name}. Available themes: {list(cls.THEMES.keys())}"
            )
        return cls.from_dict(cls.THEMES[theme_name])

    @classmethod
    def from_dict(cls, config: dict) -> "StyleManager":
        """Create a StyleManager from a configuration dictionary"""
        manager = cls()

        # Apply global style if present
        if "global_style" in config:
            manager.global_style = ElementStyle.model_validate(config["global_style"])

        # Apply element-specific styles
        for element_type, style_class in cls._element_map.items():
            key = element_type.name.lower()
            if key in config:
                style = style_class.model_validate(config[key])
                manager.element_styles[element_type] = style

        return manager

    def get_style(self, element_type: StyleType) -> ElementStyle:
        """Get the style for a specific element type, applying global overrides"""
        base_style = self.element_styles[element_type]
        if self.global_style:
            return base_style.model_copy(
                update=self.global_style.model_dump(exclude_unset=True)
            )
        return base_style

    def update_style(self, element_type: StyleType, **style_kwargs) -> None:
        """Update style for a specific element type"""
        current_style = self.element_styles[element_type]
        self.element_styles[element_type] = current_style.model_copy(
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
