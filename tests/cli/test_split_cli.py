# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for split CLI command."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from flatprot.cli.split import (
    split_command,
    parse_regions,
    SplitConfig,
)
from flatprot.core import ResidueRange
from flatprot.transformation import TransformationMatrix
import numpy as np


class TestParseRegions:
    """Test region parsing functionality."""

    def test_parse_single_region(self):
        """Test parsing a single region."""
        regions = parse_regions("A:1-100")
        assert len(regions) == 1
        assert regions[0].chain_id == "A"
        assert regions[0].start == 1
        assert regions[0].end == 100

    def test_parse_multiple_regions(self):
        """Test parsing multiple regions."""
        regions = parse_regions("A:1-100,A:150-250,B:1-80")
        assert len(regions) == 3

        assert regions[0].chain_id == "A"
        assert regions[0].start == 1
        assert regions[0].end == 100

        assert regions[1].chain_id == "A"
        assert regions[1].start == 150
        assert regions[1].end == 250

        assert regions[2].chain_id == "B"
        assert regions[2].start == 1
        assert regions[2].end == 80

    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace."""
        regions = parse_regions(" A:1-100 , B:50-150 ")
        assert len(regions) == 2
        assert regions[0].chain_id == "A"
        assert regions[1].chain_id == "B"

    def test_parse_lowercase_chain(self):
        """Test parsing with lowercase chain ID."""
        regions = parse_regions("a:1-100")
        assert len(regions) == 1
        assert regions[0].chain_id == "A"  # Should be converted to uppercase

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        with pytest.raises(ValueError, match="No valid regions found"):
            parse_regions("")

    def test_parse_invalid_format_no_colon(self):
        """Test parsing invalid format without colon."""
        with pytest.raises(ValueError, match="Failed to parse region"):
            parse_regions("A1-100")

    def test_parse_invalid_format_no_dash(self):
        """Test parsing invalid format without dash."""
        with pytest.raises(ValueError, match="Failed to parse region"):
            parse_regions("A:1100")

    def test_parse_invalid_numbers(self):
        """Test parsing with invalid numbers."""
        with pytest.raises(ValueError, match="Failed to parse region"):
            parse_regions("A:abc-100")

    def test_parse_negative_numbers(self):
        """Test parsing with negative numbers."""
        with pytest.raises(ValueError, match="Failed to parse region"):
            parse_regions("A:-1-100")

    def test_parse_start_greater_than_end(self):
        """Test parsing where start > end."""
        with pytest.raises(ValueError, match="Failed to parse region"):
            parse_regions("A:100-50")


class TestSplitConfig:
    """Test SplitConfig validation."""

    def test_valid_config(self, tmp_path):
        """Test valid configuration."""
        structure_file = tmp_path / "test.cif"
        structure_file.touch()

        config = SplitConfig(structure_file=structure_file, regions="A:1-100,B:1-50")

        assert config.structure_file == structure_file
        assert config.regions == "A:1-100,B:1-50"
        assert config.alignment_mode == "family-identity"
        assert config.gap_x == 0.0
        assert config.gap_y == 0.0

    def test_invalid_regions_empty(self, tmp_path):
        """Test invalid empty regions."""
        structure_file = tmp_path / "test.cif"
        structure_file.touch()

        with pytest.raises(ValueError, match="Regions cannot be empty"):
            SplitConfig(structure_file=structure_file, regions="")

    def test_invalid_regions_format(self, tmp_path):
        """Test invalid region format."""
        structure_file = tmp_path / "test.cif"
        structure_file.touch()

        with pytest.raises(ValueError, match="Invalid region format"):
            SplitConfig(structure_file=structure_file, regions="A1-100")

    def test_invalid_alignment_mode(self, tmp_path):
        """Test invalid alignment mode."""
        structure_file = tmp_path / "test.cif"
        structure_file.touch()

        with pytest.raises(ValueError, match="Invalid alignment mode"):
            SplitConfig(
                structure_file=structure_file,
                regions="A:1-100",
                alignment_mode="invalid",
            )

    def test_gap_parameters(self, tmp_path):
        """Test gap parameters."""
        structure_file = tmp_path / "test.cif"
        structure_file.touch()

        config = SplitConfig(
            structure_file=structure_file, regions="A:1-100", gap_x=50.0, gap_y=100.0
        )

        assert config.gap_x == 50.0
        assert config.gap_y == 100.0


class TestSplitCommand:
    """Test split command functionality."""

    @pytest.fixture
    def mock_structure_file(self, tmp_path):
        """Create a mock structure file."""
        structure_file = tmp_path / "test.cif"
        structure_file.write_text("# Mock CIF file")
        return structure_file

    @pytest.fixture
    def mock_output_file(self, tmp_path):
        """Create mock output file path."""
        return tmp_path / "output.svg"

    def test_split_command_missing_file(self, tmp_path):
        """Test split command with missing structure file."""
        missing_file = tmp_path / "missing.cif"
        output_file = tmp_path / "output.svg"

        # With @error_handler decorator, exceptions are caught and return code 1
        result = split_command(
            structure_file=missing_file, regions="A:1-100", output=output_file
        )
        assert result == 1

    def test_split_command_pdb_without_dssp(self, tmp_path):
        """Test split command with PDB file but no DSSP."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("# Mock PDB file")
        output_file = tmp_path / "output.svg"

        with patch("flatprot.cli.split.validate_structure_file"):
            # With @error_handler decorator, exceptions are caught and return code 1
            result = split_command(
                structure_file=pdb_file, regions="A:1-100", output=output_file
            )
            assert result == 1

    @patch("flatprot.cli.split.validate_structure_file")
    @patch("flatprot.cli.split.GemmiStructureParser")
    @patch("flatprot.cli.split.extract_structure_regions")
    @patch("flatprot.cli.split.ensure_database_available")
    @patch("flatprot.cli.split.align_regions_batch")
    @patch("flatprot.cli.split.apply_domain_transformations_masked")
    @patch("flatprot.cli.split.project_structure_orthographically")
    @patch("flatprot.cli.split.create_domain_aware_scene")
    @patch("flatprot.cli.split.SVGRenderer")
    def test_split_command_success(
        self,
        mock_renderer,
        mock_scene,
        mock_project,
        mock_transform,
        mock_align,
        mock_database,
        mock_extract,
        mock_parser,
        mock_validate,
        mock_structure_file,
        mock_output_file,
    ):
        """Test successful split command execution."""
        # Setup mocks
        mock_structure = Mock()
        mock_structure.id = "test"
        mock_structure.__len__ = Mock(return_value=100)
        mock_parser.return_value.parse_structure.return_value = mock_structure

        mock_region_files = [
            (Path("/tmp/region1.cif"), ResidueRange("A", 1, 100)),
            (Path("/tmp/region2.cif"), ResidueRange("A", 150, 250)),
        ]
        mock_extract.return_value = mock_region_files

        mock_database.return_value = Path("/tmp/db")

        # Create proper domain transformation mocks with transformation matrices
        mock_rotation = np.eye(3)
        mock_translation = np.zeros(3)
        mock_transform_matrix = TransformationMatrix(
            rotation=mock_rotation, translation=mock_translation
        )

        mock_dt1 = Mock()
        mock_dt1.domain_id = "A:1-100"
        mock_dt1.scop_id = "test_scop_1"
        mock_dt1.transformation_matrix = mock_transform_matrix
        mock_dt1.domain_range = ResidueRange("A", 1, 100)
        mock_dt1.alignment_probability = 0.85

        mock_dt2 = Mock()
        mock_dt2.domain_id = "A:150-250"
        mock_dt2.scop_id = "test_scop_2"
        mock_dt2.transformation_matrix = mock_transform_matrix
        mock_dt2.domain_range = ResidueRange("A", 150, 250)
        mock_dt2.alignment_probability = 0.92

        mock_domain_transformations = [mock_dt1, mock_dt2]
        mock_align.return_value = mock_domain_transformations

        mock_transformed_structure = Mock()
        mock_transform.return_value = mock_transformed_structure

        mock_projected_structure = Mock()
        mock_project.return_value = mock_projected_structure

        mock_scene_obj = Mock()
        mock_scene.return_value = mock_scene_obj

        mock_renderer_instance = Mock()
        mock_renderer.return_value = mock_renderer_instance

        # Execute command
        split_command(
            structure_file=mock_structure_file,
            regions="A:1-100,A:150-250",
            output=mock_output_file,
            alignment_mode="family-identity",
        )

        # Verify calls
        mock_validate.assert_called_once_with(mock_structure_file)
        mock_parser.assert_called_once()
        mock_extract.assert_called_once()
        mock_database.assert_called_once()
        mock_align.assert_called_once()
        # Verify domain transformation is called with original structure (no global inertia)
        # The actual domain transformations will be created from alignment results, so just check it was called
        mock_transform.assert_called_once()
        call_args = mock_transform.call_args
        assert call_args[0][0] == mock_structure  # First arg should be the structure
        assert (
            len(call_args[0][1]) == 2
        )  # Second arg should be list of 2 domain transformations
        mock_project.assert_called_once()
        mock_scene.assert_called_once()
        mock_renderer.assert_called_once()
        mock_renderer_instance.save_svg.assert_called_once_with(mock_output_file)

    @patch("flatprot.cli.split.validate_structure_file")
    @patch("flatprot.cli.split.GemmiStructureParser")
    def test_split_command_parser_failure(
        self, mock_parser, mock_validate, mock_structure_file, mock_output_file
    ):
        """Test split command with parser failure."""
        mock_parser.return_value.parse_structure.return_value = None

        # With @error_handler decorator, exceptions are caught and return code 1
        result = split_command(
            structure_file=mock_structure_file,
            regions="A:1-100",
            output=mock_output_file,
        )
        assert result == 1

    @patch("flatprot.cli.split.validate_structure_file")
    @patch("flatprot.cli.split.GemmiStructureParser")
    @patch("flatprot.cli.split.extract_structure_regions")
    @patch("flatprot.cli.split.ensure_database_available")
    @patch("flatprot.cli.split.align_regions_batch")
    def test_split_command_no_alignments(
        self,
        mock_align,
        mock_database,
        mock_extract,
        mock_parser,
        mock_validate,
        mock_structure_file,
        mock_output_file,
    ):
        """Test split command with no successful alignments."""
        # Setup mocks
        mock_structure = Mock()
        mock_structure.id = "test"
        mock_structure.__len__ = Mock(return_value=100)
        mock_parser.return_value.parse_structure.return_value = mock_structure

        mock_region_files = [(Path("/tmp/region1.cif"), ResidueRange("A", 1, 100))]
        mock_extract.return_value = mock_region_files

        mock_database.return_value = Path("/tmp/db")
        mock_align.return_value = []  # No successful alignments

        # With @error_handler decorator, exceptions are caught and return code 1
        result = split_command(
            structure_file=mock_structure_file,
            regions="A:1-100",
            output=mock_output_file,
        )
        assert result == 1

    @patch("flatprot.cli.split.validate_structure_file")
    @patch("flatprot.cli.split.GemmiStructureParser")
    @patch("flatprot.cli.split.extract_structure_regions")
    @patch("flatprot.cli.split.ensure_database_available")
    @patch("flatprot.cli.split.align_regions_batch")
    @patch("flatprot.cli.split.calculate_individual_inertia_transformations")
    @patch("flatprot.cli.split.apply_domain_transformations_masked")
    @patch("flatprot.cli.split.project_structure_orthographically")
    @patch("flatprot.cli.split.create_domain_aware_scene")
    @patch("flatprot.cli.split.SVGRenderer")
    def test_split_command_inertia_mode(
        self,
        mock_renderer,
        mock_scene,
        mock_project,
        mock_transform,
        mock_calc_inertia,
        mock_align,
        mock_database,
        mock_extract,
        mock_parser,
        mock_validate,
        mock_structure_file,
        mock_output_file,
    ):
        """Test split command with inertia alignment mode."""
        # Setup mocks
        mock_structure = Mock()
        mock_structure.id = "test"
        mock_structure.__len__ = Mock(return_value=100)
        mock_parser.return_value.parse_structure.return_value = mock_structure

        mock_region_files = [(Path("/tmp/region1.cif"), ResidueRange("A", 1, 100))]
        mock_extract.return_value = mock_region_files

        mock_database.return_value = Path("/tmp/db")

        # In inertia mode, alignment is not used, calc_inertia is used instead
        mock_domain_transformations = [Mock()]
        mock_calc_inertia.return_value = mock_domain_transformations
        mock_transform.return_value = mock_structure

        mock_projected_structure = Mock()
        mock_project.return_value = mock_projected_structure

        mock_scene_obj = Mock()
        mock_scene.return_value = mock_scene_obj

        mock_renderer_instance = Mock()
        mock_renderer.return_value = mock_renderer_instance

        # Execute command with inertia mode
        split_command(
            structure_file=mock_structure_file,
            regions="A:1-100",
            output=mock_output_file,
            alignment_mode="inertia",
        )

        # In inertia mode, individual domain inertia transformations are calculated internally
        mock_calc_inertia.assert_called_once()
        # Alignment should not be called in inertia mode
        mock_align.assert_not_called()

    @patch("flatprot.cli.split.validate_structure_file")
    @patch("flatprot.cli.split.GemmiStructureParser")
    @patch("flatprot.cli.split.extract_structure_regions")
    @patch("flatprot.cli.split.ensure_database_available")
    @patch("flatprot.cli.split.align_regions_batch")
    @patch("flatprot.cli.split.apply_domain_transformations_masked")
    @patch("flatprot.cli.split.project_structure_orthographically")
    @patch("flatprot.cli.split.create_domain_aware_scene")
    @patch("flatprot.cli.split.SVGRenderer")
    def test_split_command_canvas_size_adjustment(
        self,
        mock_renderer,
        mock_scene,
        mock_project,
        mock_transform,
        mock_align,
        mock_database,
        mock_extract,
        mock_parser,
        mock_validate,
        mock_structure_file,
        mock_output_file,
    ):
        """Test that canvas size is adjusted based on layout."""
        # Setup mocks
        mock_structure = Mock()
        mock_structure.id = "test"
        mock_structure.__len__ = Mock(return_value=100)
        mock_parser.return_value.parse_structure.return_value = mock_structure

        mock_region_files = [
            (Path("/tmp/region1.cif"), ResidueRange("A", 1, 100)),
            (Path("/tmp/region2.cif"), ResidueRange("A", 150, 250)),
        ]
        mock_extract.return_value = mock_region_files

        mock_database.return_value = Path("/tmp/db")

        # Create proper domain transformation mocks with transformation matrices
        mock_rotation = np.eye(3)
        mock_translation = np.zeros(3)
        mock_transform_matrix = TransformationMatrix(
            rotation=mock_rotation, translation=mock_translation
        )

        mock_dt1 = Mock()
        mock_dt1.domain_id = "A:1-100"
        mock_dt1.scop_id = "test_scop_1"
        mock_dt1.transformation_matrix = mock_transform_matrix
        mock_dt1.domain_range = ResidueRange("A", 1, 100)
        mock_dt1.alignment_probability = 0.85

        mock_dt2 = Mock()
        mock_dt2.domain_id = "A:150-250"
        mock_dt2.scop_id = "test_scop_2"
        mock_dt2.transformation_matrix = mock_transform_matrix
        mock_dt2.domain_range = ResidueRange("A", 150, 250)
        mock_dt2.alignment_probability = 0.92

        mock_domain_transformations = [mock_dt1, mock_dt2]
        mock_align.return_value = mock_domain_transformations

        mock_transformed_structure = Mock()
        mock_transform.return_value = mock_transformed_structure

        mock_projected_structure = Mock()
        mock_project.return_value = mock_projected_structure

        mock_scene_obj = Mock()
        mock_scene.return_value = mock_scene_obj

        mock_renderer_instance = Mock()
        mock_renderer.return_value = mock_renderer_instance

        # Test gap-based canvas size adjustment
        split_command(
            structure_file=mock_structure_file,
            regions="A:1-100,A:150-250",
            output=mock_output_file,
            gap_x=150.0,
            gap_y=100.0,
            canvas_width=1000,
            canvas_height=800,
        )

        # Should adjust canvas size for progressive gaps
        # With 2 domains: max translation = (2-1) * gap = 1 * gap
        # Canvas expansion: gap * 1.2 (20% padding)
        expected_width = 1000 + int(150.0 * 1.2)  # base width + gap expansion
        expected_height = 800 + int(100.0 * 1.2)  # base height + gap expansion
        mock_renderer.assert_called_with(
            mock_scene.return_value, expected_width, expected_height
        )
