"""Integration tests for the overlay command."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from flatprot.cli.overlay import overlay, resolve_input_files
from flatprot.cli.utils import CommonParameters
from flatprot.utils.overlay_utils import OverlayConfig


class TestResolveInputFiles:
    """Test file pattern resolution."""

    def test_resolve_single_glob_pattern(self):
        """Test resolving single glob pattern."""
        with patch("glob.glob", return_value=["file1.cif", "file2.cif"]):
            result = resolve_input_files(["*.cif"])

            assert len(result) == 2
            assert all(isinstance(f, Path) for f in result)
            assert str(result[0]).endswith("file1.cif")
            assert str(result[1]).endswith("file2.cif")

    def test_resolve_multiple_patterns(self):
        """Test resolving multiple patterns."""

        def mock_glob(pattern):
            if pattern == "folder1/*.cif":
                return ["folder1/a.cif", "folder1/b.cif"]
            elif pattern == "folder2/*.cif":
                return ["folder2/c.cif"]
            return []

        with patch("glob.glob", side_effect=mock_glob):
            result = resolve_input_files(["folder1/*.cif", "folder2/*.cif"])

            assert len(result) == 3
            assert all(isinstance(f, Path) for f in result)
            # Results should be sorted
            assert str(result[0]).endswith("a.cif")
            assert str(result[1]).endswith("b.cif")
            assert str(result[2]).endswith("c.cif")

    def test_resolve_duplicates_removed(self):
        """Test that duplicate files are removed."""

        def mock_glob(pattern):
            if pattern == "*.cif":
                return ["file1.cif", "file2.cif"]
            elif pattern == "file*.cif":
                return ["file1.cif", "file2.cif"]  # Same files
            return []

        with patch("glob.glob", side_effect=mock_glob):
            result = resolve_input_files(["*.cif", "file*.cif"])

            # Should deduplicate
            assert len(result) == 2

    def test_resolve_no_matches(self):
        """Test error when pattern matches no files."""
        with patch("glob.glob", return_value=[]):
            with pytest.raises(FileNotFoundError, match="No files found matching"):
                resolve_input_files(["nonexistent/*.cif"])

    def test_resolve_empty_patterns(self):
        """Test error with empty pattern list."""
        with pytest.raises(ValueError, match="No input files or patterns provided"):
            resolve_input_files([])


class TestOverlayCommand:
    """Integration tests for overlay command."""

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_overlay_basic_success(self, mock_resolve, mock_create):
        """Test basic successful overlay creation."""
        # Setup mocks
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files

        output_path = Path("overlay.png")
        mock_create.return_value = output_path

        # Test
        overlay(
            file_patterns=["*.cif"],
            output="overlay.png",
            common=CommonParameters(quiet=True),
        )

        # Verify calls
        mock_resolve.assert_called_once_with(["*.cif"])
        mock_create.assert_called_once()

        # Check config passed to create_overlay
        args, kwargs = mock_create.call_args
        assert args[0] == test_files  # input_files
        assert args[1] == Path("overlay.png")  # output_path
        config = args[2]
        assert isinstance(config, OverlayConfig)
        assert config.output_format == "png"

    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_overlay_insufficient_files(self, mock_resolve):
        """Test error with insufficient input files."""
        mock_resolve.return_value = [Path("single.cif")]

        with pytest.raises(SystemExit):
            overlay(file_patterns=["single.cif"], common=CommonParameters(quiet=True))

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_overlay_different_output_formats(self, mock_resolve, mock_create):
        """Test overlay with different output formats."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("test.svg")

        # Test SVG output (no Cairo needed)
        overlay(
            file_patterns=["*.cif"],
            output="test.svg",
            common=CommonParameters(quiet=True),
        )

        config = mock_create.call_args[0][2]
        assert config.output_format == "svg"

    def test_overlay_unsupported_format(self):
        """Test error with unsupported output format."""
        with patch("flatprot.cli.overlay.resolve_input_files") as mock_resolve:
            mock_resolve.return_value = [Path("file1.cif"), Path("file2.cif")]

            with pytest.raises(SystemExit):
                overlay(
                    file_patterns=["*.cif"],
                    output="test.xyz",  # Unsupported format
                    common=CommonParameters(quiet=True),
                )

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_overlay_custom_config(self, mock_resolve, mock_create):
        """Test overlay with custom configuration."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("custom.png")

        overlay(
            file_patterns=["*.cif"],
            output="custom.png",
            family="3000114",
            alignment_mode="inertia",
            canvas_width=1500,
            canvas_height=1200,
            min_probability=0.7,
            dpi=600,
            clustering=False,
            common=CommonParameters(quiet=True),
        )

        config = mock_create.call_args[0][2]
        assert config.target_family == "3000114"
        assert config.alignment_mode == "inertia"
        assert config.canvas_width == 1500
        assert config.canvas_height == 1200
        assert config.min_probability == 0.7
        assert config.dpi == 600
        assert config.clustering_enabled is False


class TestCairoAvailability:
    """Test Cairo availability detection."""

    @patch("flatprot.utils.overlay_utils.combine_drawings")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_cairo_check_png_output(self, mock_resolve, mock_combine_drawings):
        """Test Cairo availability check for PNG output."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files

        # Mock a drawing object that will fail when save_png is called
        mock_drawing = MagicMock()
        mock_drawing.save_png.side_effect = RuntimeError("Cairo not available")
        mock_combine_drawings.return_value = mock_drawing

        with pytest.raises(SystemExit):
            overlay(
                file_patterns=["*.cif"],
                output="test.png",
                common=CommonParameters(quiet=True),
            )

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_cairo_check_svg_output_no_cairo_needed(self, mock_resolve, mock_create):
        """Test SVG output works without Cairo."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("test.svg")

        # This should not check Cairo for SVG output
        overlay(
            file_patterns=["*.cif"],
            output="test.svg",
            common=CommonParameters(quiet=True),
        )

        # Should succeed without Cairo check
        mock_create.assert_called_once()


class TestOverlayErrorHandling:
    """Test error handling in overlay command."""

    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_file_resolution_error(self, mock_resolve):
        """Test handling of file resolution errors."""
        mock_resolve.side_effect = FileNotFoundError("No files found")

        with pytest.raises(SystemExit):
            overlay(
                file_patterns=["nonexistent/*.cif"], common=CommonParameters(quiet=True)
            )

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_overlay_creation_error(self, mock_resolve, mock_create):
        """Test handling of overlay creation errors."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.side_effect = RuntimeError("Creation failed")

        with pytest.raises(SystemExit):
            overlay(
                file_patterns=["*.cif"],
                output="test.png",
                common=CommonParameters(quiet=True),
            )

    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_keyboard_interrupt(self, mock_resolve):
        """Test handling of keyboard interrupt."""
        mock_resolve.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit):
            overlay(file_patterns=["*.cif"], common=CommonParameters(quiet=True))


class TestOverlayCommandLineInterface:
    """Test command line interface aspects."""

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_style_file_passed_to_config(self, mock_resolve, mock_create):
        """Test that style file is passed to config."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("test.png")

        style_path = Path("custom.toml")

        overlay(
            file_patterns=["*.cif"],
            style=style_path,
            common=CommonParameters(quiet=True),
        )

        config = mock_create.call_args[0][2]
        assert config.style_file == style_path

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    @patch("flatprot.cli.overlay.set_logging_level")
    def test_verbose_logging(self, mock_set_logging, mock_resolve, mock_create):
        """Test verbose logging configuration."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("test.png")

        # Test with verbose=True
        overlay(file_patterns=["*.cif"], common=CommonParameters(verbose=True))

        mock_set_logging.assert_called_with(CommonParameters(verbose=True))

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    @patch("flatprot.cli.overlay.set_logging_level")
    def test_quiet_logging(self, mock_set_logging, mock_resolve, mock_create):
        """Test quiet logging configuration."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("test.png")

        # Test with quiet=True
        overlay(file_patterns=["*.cif"], common=CommonParameters(quiet=True))

        mock_set_logging.assert_called_with(CommonParameters(quiet=True))


class TestRealFileSystemOperations:
    """Test with real file system operations (using temporary files)."""

    def test_resolve_input_files_with_temp_files(self):
        """Test file resolution with actual temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            file1 = temp_path / "test1.cif"
            file2 = temp_path / "test2.cif"
            file1.write_text("mock cif content")
            file2.write_text("mock cif content")

            # Test resolution
            pattern = str(temp_path / "*.cif")
            result = resolve_input_files([pattern])

            assert len(result) == 2
            assert all(f.exists() for f in result)
            assert all(f.suffix == ".cif" for f in result)
