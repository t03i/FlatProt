# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, ClassVar

from flatprot.visualization.structure.helix import HelixStyle
from flatprot.visualization.structure.sheet import SheetStyle
from flatprot.visualization.structure.coil import CoilStyle
from flatprot.visualization.structure.base import VisualizationStyle as BaseStyle
from flatprot.core import SecondaryStructureType


@dataclass
class StyleManager:
    """Manages styles for protein visualization elements"""

    # Map element types to their style classes
    _element_map = {
        SecondaryStructureType.HELIX: HelixStyle,
        SecondaryStructureType.SHEET: SheetStyle,
        SecondaryStructureType.COIL: CoilStyle,
    }

    # Predefined themes/presets
    THEMES: ClassVar[dict[str, dict]] = {
        "default": {
            "global_style": {
                "stroke_color": "#333333",
                "opacity": 1.0,
                "line_width": 3.0,
            },
            "helix": {
                "fill_color": "#FF4444",
                "wave_height_factor": 0.8,
                "ribbon_thickness_factor": 0.6,
            },
            "sheet": {
                "fill_color": "#4444FF",
                "ribbon_thickness_factor": 1.0,
                "arrow_width_factor": 1.5,
            },
            "coil": {
                "stroke_width_factor": 1.0,
            },
        },
        "colorful": {
            "helix": {
                "fill_color": "#FF0000",
                "wave_height_factor": 1.0,
            },
            "sheet": {
                "fill_  color": "#00FF00",
                "arrow_width_factor": 1.8,
            },
            "coil": {
                "stroke_color": "#0000FF",
                "stroke_width_factor": 0.5,
            },
        },
    }

    element_styles: dict[SecondaryStructureType, BaseStyle] = field(
        default_factory=lambda: {
            stype: style_class()
            for stype, style_class in StyleManager._element_map.items()
        }
    )
    global_style: Optional[BaseStyle] = None

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
            manager.global_style = BaseStyle.model_validate(config["global_style"])

        # Apply element-specific styles
        for element_type, style_class in cls._element_map.items():
            key = element_type.name.lower()
            if key in config:
                style = style_class.model_validate(config[key])
                manager.element_styles[element_type] = style

        return manager

    def get_style(self, element_type: SecondaryStructureType) -> BaseStyle:
        """Get the style for a specific element type, applying global overrides"""
        base_style = self.element_styles[element_type]
        if self.global_style:
            return base_style.model_copy(
                update=self.global_style.model_dump(exclude_unset=True)
            )
        return base_style

    def update_style(
        self, element_type: SecondaryStructureType, **style_kwargs
    ) -> None:
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
            key = element_type.name.lower()
            config[key] = style.model_dump(exclude_unset=True)

        self.THEMES[theme_name] = config
