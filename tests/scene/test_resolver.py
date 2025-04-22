# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CoordinateResolver class."""

import pytest
import numpy as np
from pytest_mock import MockerFixture

from flatprot.core import (
    Structure,
    ResidueCoordinate,
    ResidueRangeSet,
    CoordinateCalculationError,
)
from flatprot.scene import (
    BaseSceneElement,
    BaseStructureSceneElement,
    CoordinateResolver,
    TargetResidueNotFoundError,
)

# --- Fixtures ---


@pytest.fixture
def mock_structure(mocker: MockerFixture) -> Structure:
    """Basic mock Structure."""
    mock_struct = mocker.MagicMock(spec=Structure)
    # Structure might be needed by element.get_coordinate_at_residue
    return mock_struct


@pytest.fixture
def mock_element_registry(mocker: MockerFixture) -> dict[str, BaseSceneElement]:
    """Mock element registry containing structure elements."""
    # Element 1: Covers A:1-10
    elem1 = mocker.MagicMock(spec=BaseStructureSceneElement)
    elem1.id = "elem1"
    elem1.residue_range_set = ResidueRangeSet.from_string("A:1-10")
    elem1.get_coordinate_at_residue = mocker.MagicMock(
        return_value=np.array([1.0, 1.0, 1.0])
    )

    # Element 2: Covers A:11-20, but will fail coord lookup for A:15
    elem2 = mocker.MagicMock(spec=BaseStructureSceneElement)
    elem2.id = "elem2"
    elem2.residue_range_set = ResidueRangeSet.from_string("A:11-20")

    def elem2_get_coord(res, struct):
        if res.residue_index == 15:
            return None  # Simulate failure
        return np.array([2.0, 2.0, 2.0])

    elem2.get_coordinate_at_residue = mocker.MagicMock(side_effect=elem2_get_coord)

    # Element 3: Covers B:1-5, raises TargetResidueNotFoundError for B:3
    elem3 = mocker.MagicMock(spec=BaseStructureSceneElement)
    elem3.id = "elem3"
    elem3.residue_range_set = ResidueRangeSet.from_string("B:1-5")

    def elem3_get_coord(res, struct):
        if res.residue_index == 3:
            raise TargetResidueNotFoundError(struct, res)
        return np.array([3.0, 3.0, 3.0])

    elem3.get_coordinate_at_residue = mocker.MagicMock(side_effect=elem3_get_coord)

    # Element 4: Covers C:1-5, returns invalid shape coord for C:2
    elem4 = mocker.MagicMock(spec=BaseStructureSceneElement)
    elem4.id = "elem4"
    elem4.residue_range_set = ResidueRangeSet.from_string("C:1-5")

    def elem4_get_coord(res, struct):
        if res.residue_index == 2:
            return np.array([4.0, 4.0])  # Invalid shape
        return np.array([4.0, 4.0, 4.0])

    elem4.get_coordinate_at_residue = mocker.MagicMock(side_effect=elem4_get_coord)

    # Element 5: Covers D:1-5, raises unexpected Exception for D:4
    elem5 = mocker.MagicMock(spec=BaseStructureSceneElement)
    elem5.id = "elem5"
    elem5.residue_range_set = ResidueRangeSet.from_string("D:1-5")

    def elem5_get_coord(res, struct):
        if res.residue_index == 4:
            raise ValueError("Unexpected element error")
        return np.array([5.0, 5.0, 5.0])

    elem5.get_coordinate_at_residue = mocker.MagicMock(side_effect=elem5_get_coord)

    # Non-structure element (should be ignored by resolver init)
    non_struct_elem = mocker.MagicMock(spec=BaseSceneElement)  # Not a StructureElement
    non_struct_elem.id = "non_struct"

    return {
        elem1.id: elem1,
        elem2.id: elem2,
        elem3.id: elem3,
        elem4.id: elem4,
        elem5.id: elem5,
        non_struct_elem.id: non_struct_elem,
    }


@pytest.fixture
def resolver(
    mock_structure: Structure, mock_element_registry: dict[str, BaseSceneElement]
) -> CoordinateResolver:
    """Fixture for CoordinateResolver instance."""
    return CoordinateResolver(mock_structure, mock_element_registry)


# --- Test Cases ---


def test_resolver_init_filters_elements(
    mock_structure: Structure, mock_element_registry: dict[str, BaseSceneElement]
):
    """Test that the resolver only keeps structure elements during init."""
    resolver_instance = CoordinateResolver(mock_structure, mock_element_registry)
    assert len(resolver_instance._structure_elements) == 5  # Excludes non_struct_elem
    assert all(
        isinstance(el, BaseStructureSceneElement)
        for el in resolver_instance._structure_elements
    )


def test_resolve_success(
    resolver: CoordinateResolver,
    mock_structure: Structure,
    mock_element_registry: dict[str, BaseSceneElement],
):
    """Test successful coordinate resolution."""
    target = ResidueCoordinate("A", 5)
    expected_coord = np.array([1.0, 1.0, 1.0])
    resolved_coord = resolver.resolve(target)

    np.testing.assert_array_equal(resolved_coord, expected_coord)
    # Check that the correct element's method was called
    elem1 = mock_element_registry["elem1"]
    elem1.get_coordinate_at_residue.assert_called_once_with(target, mock_structure)


def test_resolve_no_covering_element(resolver: CoordinateResolver):
    """Test CoordinateCalculationError when no element covers the residue."""
    target = ResidueCoordinate("X", 1)  # No element covers chain X
    with pytest.raises(
        CoordinateCalculationError, match="not covered by any structure element"
    ):
        resolver.resolve(target)


def test_resolve_element_returns_none(
    resolver: CoordinateResolver,
    mock_structure: Structure,
    mock_element_registry: dict[str, BaseSceneElement],
):
    """Test CoordinateCalculationError when the covering element returns None."""
    target = ResidueCoordinate("A", 15)  # Covered by elem2, which returns None for 15
    with pytest.raises(
        CoordinateCalculationError, match="failed to resolve coordinate"
    ):
        resolver.resolve(target)
    # Check that the correct element's method was called
    elem2 = mock_element_registry["elem2"]
    elem2.get_coordinate_at_residue.assert_called_once_with(target, mock_structure)


def test_resolve_element_returns_invalid_shape(
    resolver: CoordinateResolver,
    mock_structure: Structure,
    mock_element_registry: dict[str, BaseSceneElement],
):
    """Test CoordinateCalculationError when element returns wrong coordinate shape."""
    target = ResidueCoordinate("C", 2)  # Covered by elem4, returns wrong shape
    with pytest.raises(CoordinateCalculationError, match="invalid coordinate data"):
        resolver.resolve(target)
    # Check that the correct element's method was called
    elem4 = mock_element_registry["elem4"]
    elem4.get_coordinate_at_residue.assert_called_once_with(target, mock_structure)


def test_resolve_element_raises_target_not_found(
    resolver: CoordinateResolver,
    mock_structure: Structure,
    mock_element_registry: dict[str, BaseSceneElement],
):
    """Test TargetResidueNotFoundError is propagated from the element."""
    target = ResidueCoordinate(
        "B", 3
    )  # Covered by elem3, raises TargetResidueNotFoundError
    with pytest.raises(TargetResidueNotFoundError):
        resolver.resolve(target)
    # Check that the correct element's method was called
    elem3 = mock_element_registry["elem3"]
    elem3.get_coordinate_at_residue.assert_called_once_with(target, mock_structure)


def test_resolve_element_raises_unexpected_error(
    resolver: CoordinateResolver,
    mock_structure: Structure,
    mock_element_registry: dict[str, BaseSceneElement],
):
    """Test unexpected errors from element are wrapped in CoordinateCalculationError."""
    target = ResidueCoordinate("D", 4)  # Covered by elem5, raises ValueError
    with pytest.raises(
        CoordinateCalculationError, match="Unexpected error resolving coordinate"
    ) as excinfo:
        resolver.resolve(target)

    # Check that the original ValueError is chained
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Unexpected element error" in str(excinfo.value.__cause__)

    # Check that the correct element's method was called
    elem5 = mock_element_registry["elem5"]
    elem5.get_coordinate_at_residue.assert_called_once_with(target, mock_structure)
