"""CLI tests for the overlay command."""

import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from flatprot.cli.overlay import overlay, resolve_input_files
from flatprot.cli.utils import CommonParameters


class TestOverlayCommandParameters:
    """Test various command parameters."""

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_all_parameters_passed_correctly(self, mock_resolve, mock_create):
        """Test that all CLI parameters are passed correctly."""
        test_files = [Path("file1.cif"), Path("file2.cif")]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("output.svg")

        overlay(
            file_patterns=["file1.cif", "file2.cif"],
            output="output.svg",
            family="3000114",
            alignment_mode="inertia",
            canvas_width=1500,
            canvas_height=1200,
            min_probability=0.7,
            dpi=600,
            clustering=False,
            clustering_auto_threshold=50,
            clustering_min_seq_id=0.8,
            clustering_coverage=0.95,
            disable_scaling=True,
            common=CommonParameters(quiet=True),
        )

        # Verify create_overlay was called with correct config
        mock_create.assert_called_once()
        config = mock_create.call_args[0][2]

        assert config.target_family == "3000114"
        assert config.alignment_mode == "inertia"
        assert config.canvas_width == 1500
        assert config.canvas_height == 1200
        assert config.min_probability == 0.7
        assert config.dpi == 600
        assert config.clustering_enabled is False
        assert config.clustering_auto_threshold == 50
        assert config.clustering_min_seq_id == 0.8
        assert config.clustering_coverage == 0.95
        assert config.disable_scaling is True
        assert config.quiet is True

    @patch("flatprot.cli.overlay.create_overlay")
    @patch("flatprot.cli.overlay.resolve_input_files")
    def test_auto_clustering_parameters(self, mock_resolve, mock_create):
        """Test that auto-clustering parameters work correctly."""
        test_files = [Path(f"file{i}.cif") for i in range(5)]
        mock_resolve.return_value = test_files
        mock_create.return_value = Path("output.svg")

        # Test with auto-clustering (clustering=None)
        overlay(
            file_patterns=["*.cif"],
            output="output.svg",
            clustering=None,  # Auto-decide
            clustering_auto_threshold=100,
            common=CommonParameters(quiet=True),
        )

        # Verify create_overlay was called with correct config
        mock_create.assert_called_once()
        config = mock_create.call_args[0][2]

        assert config.clustering_enabled is None  # Should be None for auto-decision
        assert config.clustering_auto_threshold == 100


class TestOverlayWithRealFileSystem:
    """Test overlay command with real file system operations."""

    @patch("flatprot.cli.overlay.create_overlay")
    def test_file_pattern_resolution_with_temp_files(self, mock_create):
        """Test file pattern resolution with actual files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock CIF files
            for i in range(3):
                cif_file = temp_path / f"test{i}.cif"
                cif_file.write_text(f"mock cif content {i}")

            # Test with actual pattern
            pattern = str(temp_path / "*.cif")
            mock_create.return_value = Path("output.svg")

            overlay(
                file_patterns=[pattern],
                output="output.svg",
                common=CommonParameters(quiet=True),
            )

            # Should find files and attempt to create overlay
            mock_create.assert_called_once()
            # Verify correct number of files found
            files_arg = mock_create.call_args[0][0]
            assert len(files_arg) == 3

    def test_helpful_error_for_insufficient_files(self):
        """Test helpful error message for insufficient files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create only one file
            cif_file = temp_path / "single.cif"
            cif_file.write_text("mock content")

            with pytest.raises(SystemExit):
                overlay(
                    file_patterns=[str(cif_file)], common=CommonParameters(quiet=True)
                )


class TestOverlayFileResolution:
    """Test file resolution functionality."""

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

    def test_helpful_error_for_no_files_found(self):
        """Test helpful error message when no files are found."""
        with pytest.raises(FileNotFoundError, match="No files found matching"):
            resolve_input_files(["nonexistent/*.cif"])


if __name__ == "__main__":
    pytest.main([__file__])
