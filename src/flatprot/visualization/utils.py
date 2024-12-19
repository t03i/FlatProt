# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class CanvasSettings:
    """Settings for the canvas"""

    width: int = 1024
    height: int = 1024
    background_color: str = "#FFFFFF"
    padding: float = 0.05  # Padding as fraction of canvas size

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns canvas dimensions as (width, height)"""
        return (self.width, self.height)

    @property
    def padding_pixels(self) -> tuple[float, float]:
        """Returns padding in pixels as (x_padding, y_padding)"""
        return (self.width * self.padding, self.height * self.padding)
