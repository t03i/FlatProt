# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, ClassVar

from flatprot.visualization.elements.helix import HelixStyle
from flatprot.visualization.elements.sheet import SheetStyle
from flatprot.visualization.elements.coil import CoilStyle
from flatprot.visualization.elements.base import VisualizationStyle as BaseStyle
from flatprot.structure import SecondaryStructureType


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
                "color": "#000000",
                "opacity": 1.0,
                "line_width": 1.0,
            },
        },
        "publication": {
            "global_style": {
                "color": "#333333",
                "opacity": 1.0,
                "line_width": 2.0,
            },
            "helix": {
                "color": "#FF4444",
                "wave_height_factor": 0.8,
                "ribbon_thickness_factor": 0.6,
            },
            "sheet": {
                "color": "#4444FF",
                "ribbon_thickness_factor": 1.0,
                "arrow_width_factor": 1.5,
            },
            "coil": {
                "color": "#444444",
                "line_thickness_factor": 0.4,
            },
        },
        "colorful": {
            "helix": {
                "color": "#FF0000",
                "wave_height_factor": 1.0,
            },
            "sheet": {
                "color": "#00FF00",
                "arrow_width_factor": 1.8,
            },
            "coil": {
                "color": "#0000FF",
                "line_thickness_factor": 0.5,
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
