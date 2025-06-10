"""Tests for overlay utility functions."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import drawsvg as draw

from flatprot.utils.overlay_utils import (
    OverlayConfig,
    create_overlay,
    generate_aligned_drawing,
    cluster_and_select_representatives,
    calculate_opacity,
    combine_drawings,
    load_styles,
)
from flatprot.core import Structure


class TestOverlayConfig:
    """Test OverlayConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OverlayConfig()

        assert config.alignment_mode == "family-identity"
        assert config.target_family is None
        assert config.min_probability == 0.5
        assert config.clustering_enabled is None  # Auto-decision
        assert config.clustering_auto_threshold == 100
        assert config.clustering_min_seq_id == 0.5
        assert config.clustering_coverage == 0.9
        assert config.opacity_scaling is True
        assert config.canvas_width == 1000
        assert config.canvas_height == 1000
        assert config.style_file is None
        assert config.output_format == "png"
        assert config.dpi == 300
        assert config.quiet is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OverlayConfig(
            alignment_mode="inertia",
            target_family="3000114",
            min_probability=0.7,
            clustering_enabled=False,
            clustering_auto_threshold=50,
            clustering_min_seq_id=0.8,
            clustering_coverage=0.95,
            canvas_width=1500,
            output_format="svg",
        )

        assert config.alignment_mode == "inertia"
        assert config.target_family == "3000114"
        assert config.min_probability == 0.7
        assert config.clustering_enabled is False
        assert config.clustering_auto_threshold == 50
        assert config.clustering_min_seq_id == 0.8
        assert config.clustering_coverage == 0.95
        assert config.canvas_width == 1500
        assert config.output_format == "svg"

    def test_clustering_auto_config(self):
        """Test auto-clustering configuration."""
        config = OverlayConfig(
            clustering_enabled=None,  # Auto-decision
            clustering_auto_threshold=25,
            clustering_min_seq_id=0.3,
            clustering_coverage=0.7,
        )

        assert config.clustering_enabled is None
        assert config.clustering_auto_threshold == 25
        assert config.clustering_min_seq_id == 0.3
        assert config.clustering_coverage == 0.7


class TestCalculateOpacity:
    """Test opacity calculation function."""

    def test_empty_clusters(self):
        """Test with empty cluster counts."""
        result = calculate_opacity({})
        assert result == {}

    def test_single_cluster(self):
        """Test with single cluster."""
        cluster_counts = {"cluster1": 5}
        result = calculate_opacity(cluster_counts)

        assert len(result) == 1
        assert (
            result["cluster1"] == 0.3
        )  # Single cluster gets max opacity (updated range)

    def test_multiple_clusters(self):
        """Test with multiple clusters of different sizes."""
        cluster_counts = {"small": 2, "medium": 5, "large": 10}
        result = calculate_opacity(cluster_counts)

        assert len(result) == 3
        assert result["small"] < result["medium"] < result["large"]
        assert result["small"] >= 0.05  # Min opacity (updated range)
        assert result["large"] <= 1.0  # Max opacity

    def test_custom_opacity_range(self):
        """Test with custom opacity range."""
        cluster_counts = {"a": 1, "b": 10}
        result = calculate_opacity(cluster_counts, min_opacity=0.2, max_opacity=0.8)

        assert result["a"] == 0.2
        assert result["b"] == 0.8


class TestCombineDrawings:
    """Test drawing combination function."""

    def test_combine_empty_list(self):
        """Test combining empty list of drawings."""
        config = OverlayConfig(canvas_width=500, canvas_height=400)

        result = combine_drawings([], config)

        assert isinstance(result, draw.Drawing)
        assert result.width == 500
        assert result.height == 400

    def test_combine_single_drawing(self):
        """Test combining single drawing."""
        drawing = draw.Drawing(100, 100)
        drawing.append(draw.Circle(50, 50, 20))

        config = OverlayConfig(canvas_width=200, canvas_height=200)
        drawings_with_opacity = [(drawing, 0.5, "test")]

        result = combine_drawings(drawings_with_opacity, config)

        assert isinstance(result, draw.Drawing)
        assert result.width == 200
        assert result.height == 200
        assert len(result.elements) == 1  # One group added

    @patch("drawsvg.Group")
    def test_combine_multiple_drawings(self, mock_group):
        """Test combining multiple drawings."""
        # Create mock drawings
        drawing1 = Mock()
        drawing1.elements = [Mock(), Mock()]

        drawing2 = Mock()
        drawing2.elements = [Mock()]

        config = OverlayConfig()
        drawings_with_opacity = [(drawing1, 0.8, "struct1"), (drawing2, 0.6, "struct2")]

        result = combine_drawings(drawings_with_opacity, config)

        assert isinstance(result, draw.Drawing)
        # Should have called Group constructor twice (once per drawing)
        assert mock_group.call_count == 2


class TestLoadStyles:
    """Test style loading function."""

    @patch("flatprot.utils.overlay_utils.StyleParser")
    def test_load_styles_success(self, mock_parser_class):
        """Test successful style loading."""
        mock_parser = Mock()
        mock_styles = {"helix": Mock(), "sheet": Mock()}
        mock_parser.get_element_styles.return_value = mock_styles
        mock_parser_class.return_value = mock_parser

        style_file = Path("test_style.toml")
        result = load_styles(style_file)

        mock_parser_class.assert_called_once_with(style_file)
        mock_parser.get_element_styles.assert_called_once()
        assert result == mock_styles

    @patch("flatprot.utils.overlay_utils.StyleParser")
    @patch("flatprot.utils.overlay_utils.logger")
    def test_load_styles_failure(self, mock_logger, mock_parser_class):
        """Test style loading with error."""
        mock_parser_class.side_effect = Exception("Parse error")

        style_file = Path("invalid_style.toml")
        result = load_styles(style_file)

        assert result is None
        mock_logger.warning.assert_called_once()


class TestGenerateAlignedDrawing:
    """Test drawing generation function."""

    @patch("flatprot.utils.overlay_utils.validate_structure_file")
    @patch("flatprot.utils.overlay_utils.GemmiStructureParser")
    @patch("flatprot.utils.overlay_utils.transform_structure_with_inertia")
    @patch("flatprot.utils.overlay_utils.project_structure_orthographically")
    @patch("flatprot.utils.overlay_utils.create_scene_from_structure")
    @patch("flatprot.utils.overlay_utils.SVGRenderer")
    @patch("flatprot.utils.overlay_utils.load_styles")
    def test_generate_aligned_drawing_inertia_mode(
        self,
        mock_load_styles,
        mock_renderer_class,
        mock_create_scene,
        mock_project,
        mock_transform,
        mock_parser_class,
        mock_validate,
    ):
        """Test drawing generation with inertia alignment mode."""
        # Setup mocks
        mock_structure = Mock(spec=Structure)
        mock_parser = Mock()
        mock_parser.parse_structure.return_value = mock_structure
        mock_parser_class.return_value = mock_parser

        mock_transformed = Mock()
        mock_transform.return_value = mock_transformed

        mock_projected = Mock()
        mock_project.return_value = mock_projected

        mock_scene = Mock()
        mock_create_scene.return_value = mock_scene

        mock_drawing = Mock(spec=draw.Drawing)
        mock_renderer = Mock()
        mock_renderer.render.return_value = mock_drawing
        mock_renderer_class.return_value = mock_renderer

        mock_load_styles.return_value = None

        # Test
        config = OverlayConfig(alignment_mode="inertia")
        file_path = Path("test.cif")

        result = generate_aligned_drawing(file_path, config)

        # Assertions
        mock_validate.assert_called_once_with(file_path)
        mock_parser_class.assert_called_once()
        mock_parser.parse_structure.assert_called_once_with(file_path)
        mock_transform.assert_called_once_with(mock_structure)
        mock_project.assert_called_once_with(
            mock_transformed, 1000, 1000, disable_scaling=True
        )
        mock_create_scene.assert_called_once_with(mock_projected, None)
        mock_renderer_class.assert_called_once_with(
            mock_scene, 1000, 1000, background_color=None
        )
        mock_renderer.render.assert_called_once()

        assert result == mock_drawing


class TestClusterAndSelectRepresentatives:
    """Test clustering functionality."""

    @patch("subprocess.run")
    @patch("tempfile.TemporaryDirectory")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.write_bytes")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.read_bytes")
    @patch("builtins.open")
    def test_clustering_success(
        self,
        mock_open,
        mock_read_bytes,
        mock_mkdir,
        mock_write_bytes,
        mock_exists,
        mock_temp_dir,
        mock_subprocess,
    ):
        """Test successful clustering."""
        # Setup temporary directory mock
        temp_path = Path("/tmp/test")
        mock_temp_dir.return_value.__enter__.return_value = str(temp_path)

        # Mock subprocess success
        mock_subprocess.return_value = Mock()

        # Create test files
        test_files = [Path("struct1.cif"), Path("struct2.cif")]

        # Mock file operations
        mock_read_bytes.return_value = b"mock file content"
        mock_exists.return_value = True

        # Mock cluster file content
        cluster_content = "struct1\tstruct1\nstruct1\tstruct2\n"
        mock_open.return_value.__enter__.return_value.read.return_value = (
            cluster_content
        )

        config = OverlayConfig(quiet=True)
        result = cluster_and_select_representatives(test_files, config)

        assert len(result) <= len(test_files)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    @patch("subprocess.run")
    @patch("tempfile.TemporaryDirectory")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_bytes")
    @patch("pathlib.Path.read_bytes")
    @patch("flatprot.utils.overlay_utils.logger")
    def test_clustering_failure_fallback(
        self,
        mock_logger,
        mock_read_bytes,
        mock_write_bytes,
        mock_mkdir,
        mock_temp_dir,
        mock_subprocess,
    ):
        """Test clustering failure falls back to all structures."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "foldseek")
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        mock_read_bytes.return_value = b"mock file content"

        test_files = [Path("struct1.cif"), Path("struct2.cif")]
        config = OverlayConfig(quiet=True)

        result = cluster_and_select_representatives(test_files, config)

        assert len(result) == len(test_files)
        assert all(opacity == 0.1 for _, opacity in result)  # Updated default opacity


# Removed mock_open_cluster_file helper function as it's no longer needed


class TestCreateOverlay:
    """Test main overlay creation function."""

    @patch("flatprot.utils.overlay_utils.generate_aligned_drawing")
    @patch("flatprot.utils.overlay_utils.combine_drawings")
    @patch("flatprot.utils.overlay_utils.logger")
    def test_create_overlay_basic(self, mock_logger, mock_combine, mock_generate):
        """Test basic overlay creation."""
        # Setup mocks
        mock_drawing = Mock(spec=draw.Drawing)
        mock_generate.return_value = mock_drawing

        mock_combined = Mock(spec=draw.Drawing)
        mock_combined.save_png = Mock()
        mock_combine.return_value = mock_combined

        # Test data
        input_files = [Path("struct1.cif"), Path("struct2.cif")]
        output_path = Path("overlay.png")
        config = OverlayConfig(clustering_enabled=False, quiet=True)

        result = create_overlay(input_files, output_path, config)

        # Assertions
        assert mock_generate.call_count == len(input_files)
        mock_combine.assert_called_once()
        mock_combined.save_png.assert_called_once_with(str(output_path), dpi=300)
        assert result == output_path

    @patch("flatprot.utils.overlay_utils.generate_aligned_drawing")
    @patch("flatprot.utils.overlay_utils.combine_drawings")
    @patch("flatprot.utils.overlay_utils.logger")
    def test_create_overlay_svg_output(self, mock_logger, mock_combine, mock_generate):
        """Test overlay creation with SVG output."""
        mock_drawing = Mock(spec=draw.Drawing)
        mock_generate.return_value = mock_drawing

        mock_combined = Mock(spec=draw.Drawing)
        mock_combined.save_svg = Mock()
        mock_combine.return_value = mock_combined

        input_files = [Path("struct1.cif")]
        output_path = Path("overlay.svg")
        config = OverlayConfig(
            output_format="svg", clustering_enabled=False, quiet=True
        )

        result = create_overlay(input_files, output_path, config)

        mock_combined.save_svg.assert_called_once_with(str(output_path))
        assert result == output_path

    @patch("flatprot.utils.overlay_utils.logger")
    def test_create_overlay_no_valid_drawings(self, mock_logger):
        """Test overlay creation when no drawings can be generated."""
        with patch(
            "flatprot.utils.overlay_utils.generate_aligned_drawing",
            side_effect=Exception("Generation failed"),
        ):
            input_files = [Path("struct1.cif")]
            output_path = Path("overlay.png")
            config = OverlayConfig(clustering_enabled=False, quiet=True)

            with pytest.raises(RuntimeError, match="No drawings could be generated"):
                create_overlay(input_files, output_path, config)
