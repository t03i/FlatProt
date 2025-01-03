# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import drawsvg as draw

from flatprot.visualization.elements import VisualizationElement


class GroupVisualization(VisualizationElement):
    """A group of visualization elements"""

    def __init__(self, elements: list[VisualizationElement]):
        self.elements = elements

    def add_element(self, element: VisualizationElement) -> None:
        self.elements.append(element)

    def insert_element(self, index: int, element: VisualizationElement) -> None:
        self.elements.insert(index, element)

    def remove_element(self, element: VisualizationElement) -> None:
        self.elements.remove(element)

    def render(self) -> draw.Group:
        group = draw.Group()
        for element in self.elements:
            group.add(element.render())
        return group
