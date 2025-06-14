# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for position annotation functionality."""

import pytest
import numpy as np
from pydantic_extra_types.color import Color

from flatprot.core import (
    ResidueCoordinate,
    ResidueRange,
    ResidueRangeSet,
)
from flatprot.scene import (
    PositionAnnotation,
    PositionAnnotationStyle,
    PositionType,
)


class TestPositionAnnotationStyle:
    """Test PositionAnnotationStyle configuration."""

    def test_default_style(self):
        """Test default style values."""
        style = PositionAnnotationStyle()

        assert style.font_size == 8.0
        assert style.font_weight == "normal"
        assert style.font_family == "Arial, sans-serif"
        assert style.text_offset == 5.0
        assert style.show_terminus is True
        assert style.show_residue_numbers is True
        assert style.terminus_font_size == 10.0
        assert style.terminus_font_weight == "bold"

    def test_custom_style(self):
        """Test custom style configuration."""
        style = PositionAnnotationStyle(
            font_size=12.0,
            font_weight="bold",
            font_family="Helvetica",
            text_offset=5.0,
            show_terminus=False,
            show_residue_numbers=False,
            terminus_font_size=14.0,
            terminus_font_weight="normal",
            color=Color("#FF0000"),
        )

        assert style.font_size == 12.0
        assert style.font_weight == "bold"
        assert style.font_family == "Helvetica"
        assert style.text_offset == 5.0
        assert style.show_terminus is False
        assert style.show_residue_numbers is False
        assert style.terminus_font_size == 14.0
        assert style.terminus_font_weight == "normal"
        assert style.color == Color("#FF0000")


class TestPositionAnnotation:
    """Test PositionAnnotation functionality."""

    def test_create_n_terminus_annotation(self):
        """Test creating an N-terminus position annotation."""
        coord = ResidueCoordinate("A", 1, None, 0)
        annotation = PositionAnnotation(
            id="n_terminus",
            target=coord,
            position_type=PositionType.N_TERMINUS,
            text="N",
        )

        assert annotation.id == "n_terminus"
        assert annotation.target == coord
        assert annotation.position_type == PositionType.N_TERMINUS
        assert annotation.text == "N"
        assert annotation.label == "N"

    def test_create_c_terminus_annotation(self):
        """Test creating a C-terminus position annotation."""
        coord = ResidueCoordinate("A", 100, None, 99)
        annotation = PositionAnnotation(
            id="c_terminus",
            target=coord,
            position_type=PositionType.C_TERMINUS,
            text="C",
        )

        assert annotation.id == "c_terminus"
        assert annotation.target == coord
        assert annotation.position_type == PositionType.C_TERMINUS
        assert annotation.text == "C"
        assert annotation.label == "C"

    def test_create_residue_number_annotation(self):
        """Test creating a residue number position annotation."""
        residue_range = ResidueRange("A", 42, 48, 41, None)
        annotation = PositionAnnotation(
            id="residue_42",
            target=residue_range,
            position_type=PositionType.RESIDUE_NUMBER,
            text="42",
        )

        assert annotation.id == "residue_42"
        assert annotation.target == residue_range
        assert annotation.position_type == PositionType.RESIDUE_NUMBER
        assert annotation.text == "42"
        assert annotation.label == "42"

    def test_display_properties_terminus(self):
        """Test display properties for terminus annotations."""
        style = PositionAnnotationStyle(
            terminus_font_size=12.0,
            terminus_font_weight="bold",
            font_family="Arial",
            text_offset=4.0,
        )

        annotation = PositionAnnotation(
            id="n_term",
            target=ResidueCoordinate("A", 1, None, 0),
            position_type=PositionType.N_TERMINUS,
            text="N",
            style=style,
        )

        props = annotation.get_display_properties()
        assert props["font_size"] == 12.0
        assert props["font_weight"] == "bold"
        assert props["font_family"] == "Arial"
        assert props["text"] == "N"
        assert props["offset"] == 4.0

    def test_display_properties_residue_number(self):
        """Test display properties for residue number annotations."""
        style = PositionAnnotationStyle(
            font_size=8.0,
            font_weight="normal",
            font_family="Helvetica",
            text_offset=3.0,
        )

        annotation = PositionAnnotation(
            id="residue_42",
            target=ResidueCoordinate("A", 42, None, 41),
            position_type=PositionType.RESIDUE_NUMBER,
            text="42",
            style=style,
        )

        props = annotation.get_display_properties()
        assert props["font_size"] == 8.0
        assert props["font_weight"] == "normal"
        assert props["font_family"] == "Helvetica"
        assert props["text"] == "42"
        assert props["offset"] == 3.0

    def test_coordinate_resolution_single_residue(self, mocker):
        """Test coordinate resolution for single residue target."""
        # Mock resolver
        mock_resolver = mocker.MagicMock()
        expected_coord = np.array([10.0, 20.0, 5.0])
        mock_resolver.resolve.return_value = expected_coord

        coord = ResidueCoordinate("A", 42, None, 41)
        annotation = PositionAnnotation(
            id="test_pos",
            target=coord,
            position_type=PositionType.RESIDUE_NUMBER,
            text="42",
        )

        result = annotation.get_coordinates(mock_resolver)

        # Should return the coordinate as a single-row array
        expected_result = np.array([expected_coord])
        np.testing.assert_array_equal(result, expected_result)
        mock_resolver.resolve.assert_called_once_with(coord)

    def test_coordinate_resolution_residue_range_n_terminus(self, mocker):
        """Test coordinate resolution for residue range with N-terminus type."""
        # Mock resolver
        mock_resolver = mocker.MagicMock()
        expected_coord = np.array([15.0, 25.0, 8.0])
        mock_resolver.resolve.return_value = expected_coord

        residue_range = ResidueRange("A", 10, 20, 9, None)
        annotation = PositionAnnotation(
            id="n_term",
            target=residue_range,
            position_type=PositionType.N_TERMINUS,
            text="N",
        )

        result = annotation.get_coordinates(mock_resolver)

        # Should resolve the start of the range
        expected_result = np.array([expected_coord])
        np.testing.assert_array_equal(result, expected_result)

        # Check that resolver was called with start coordinate
        call_args = mock_resolver.resolve.call_args[0][0]
        assert call_args.chain_id == "A"
        assert call_args.residue_index == 10  # Start of range

    def test_coordinate_resolution_residue_range_c_terminus(self, mocker):
        """Test coordinate resolution for residue range with C-terminus type."""
        # Mock resolver
        mock_resolver = mocker.MagicMock()
        expected_coord = np.array([25.0, 35.0, 12.0])
        mock_resolver.resolve.return_value = expected_coord

        residue_range = ResidueRange("A", 10, 20, 9, None)
        annotation = PositionAnnotation(
            id="c_term",
            target=residue_range,
            position_type=PositionType.C_TERMINUS,
            text="C",
        )

        result = annotation.get_coordinates(mock_resolver)

        # Should resolve the end of the range
        expected_result = np.array([expected_coord])
        np.testing.assert_array_equal(result, expected_result)

        # Check that resolver was called with end coordinate
        call_args = mock_resolver.resolve.call_args[0][0]
        assert call_args.chain_id == "A"
        assert call_args.residue_index == 20  # End of range

    def test_coordinate_resolution_residue_range_set(self, mocker):
        """Test coordinate resolution for residue range set."""
        # Mock resolver
        mock_resolver = mocker.MagicMock()
        expected_coord = np.array([30.0, 40.0, 15.0])
        mock_resolver.resolve.return_value = expected_coord

        range1 = ResidueRange("A", 10, 20, 9, None)
        range2 = ResidueRange("A", 30, 40, 29, None)
        range_set = ResidueRangeSet([range1, range2])

        annotation = PositionAnnotation(
            id="range_set_n",
            target=range_set,
            position_type=PositionType.N_TERMINUS,
            text="N",
        )

        result = annotation.get_coordinates(mock_resolver)

        # Should resolve the start of the first range
        expected_result = np.array([expected_coord])
        np.testing.assert_array_equal(result, expected_result)

        # Check that resolver was called with first range start
        call_args = mock_resolver.resolve.call_args[0][0]
        assert call_args.chain_id == "A"
        assert call_args.residue_index == 10  # Start of first range

    def test_coordinate_resolution_residue_range_set_c_terminus(self, mocker):
        """Test coordinate resolution for residue range set with C-terminus."""
        # Mock resolver
        mock_resolver = mocker.MagicMock()
        expected_coord = np.array([45.0, 55.0, 20.0])
        mock_resolver.resolve.return_value = expected_coord

        range1 = ResidueRange("A", 10, 20, 9, None)
        range2 = ResidueRange("A", 30, 40, 29, None)
        range_set = ResidueRangeSet([range1, range2])

        annotation = PositionAnnotation(
            id="range_set_c",
            target=range_set,
            position_type=PositionType.C_TERMINUS,
            text="C",
        )

        result = annotation.get_coordinates(mock_resolver)

        # Should resolve the end of the last range
        expected_result = np.array([expected_coord])
        np.testing.assert_array_equal(result, expected_result)

        # Check that resolver was called with last range end
        call_args = mock_resolver.resolve.call_args[0][0]
        assert call_args.chain_id == "A"
        assert call_args.residue_index == 40  # End of last range

    def test_invalid_target_type(self):
        """Test that invalid target types raise appropriate errors."""
        with pytest.raises(ValueError, match="Unsupported target type"):
            _ = PositionAnnotation(
                id="invalid",
                target="invalid_target",  # String is not a valid target
                position_type=PositionType.RESIDUE_NUMBER,
                text="42",
            )

    def test_empty_range_set_error(self, mocker):
        """Test that empty range set raises appropriate error."""
        mock_resolver = mocker.MagicMock()
        empty_range_set = ResidueRangeSet([])

        annotation = PositionAnnotation(
            id="empty_range",
            target=empty_range_set,
            position_type=PositionType.N_TERMINUS,
            text="N",
        )

        with pytest.raises(ValueError, match="Empty ResidueRangeSet"):
            annotation.get_coordinates(mock_resolver)


class TestPositionType:
    """Test PositionType enum."""

    def test_position_type_values(self):
        """Test that PositionType enum has expected values."""
        assert PositionType.N_TERMINUS == "n_terminus"
        assert PositionType.C_TERMINUS == "c_terminus"
        assert PositionType.RESIDUE_NUMBER == "residue_number"

    def test_position_type_membership(self):
        """Test PositionType enum membership."""
        assert "n_terminus" in PositionType
        assert "c_terminus" in PositionType
        assert "residue_number" in PositionType
        assert "invalid_type" not in PositionType
