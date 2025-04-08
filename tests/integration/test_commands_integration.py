# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the FlatProt CLI commands."""

import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Generator, Any
import pytest
from pytest_mock import MockerFixture

from flatprot.cli.align import project_structure_svg, print_success_summary
from flatprot.core.error import FlatProtError


@pytest.fixture
def temp_structure_file() -> Generator[str, None, None]:
    """Create a temporary structure file with minimal valid PDB content.

    Yields:
        Path to the temporary structure file
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(
            b"ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N\n"
        )
        f.write(
            b"HETATM    2  CA  ALA A   1      13.000  13.000  13.000  1.00 20.00           C\n"
        )
        f.write(b"END\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_matrix_file() -> Generator[str, None, None]:
    """Create a temporary valid transformation matrix file.

    Yields:
        Path to the temporary matrix file
    """
    # Create a valid 4x4 transformation matrix and save it
    matrix = np.eye(4)  # Identity matrix
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        temp_path = f.name

    # Save a proper numpy array to the file
    np.save(temp_path, matrix)

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_style_file() -> Generator[str, None, None]:
    """Create a temporary valid style file.

    Yields:
        Path to the temporary style file
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b"[canvas]\nwidth = 800\nheight = 600\n")
        f.write(b"[helix]\nstroke_color = '#FF0000'\n")
        f.write(b"[point_annotation]\nstroke_color = '#0000FF'\nstroke_width = 2.0\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_annotations_file() -> Generator[str, None, None]:
    """Create a temporary valid annotations file.

    Yields:
        Path to the temporary annotations file
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b"[[point_annotation]]\nchain = 'A'\nresidue = 1\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_output_file() -> Generator[str, None, None]:
    """Create a temporary output file path.

    Yields:
        Path to the temporary output file
    """
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        temp_path = f.name

    os.unlink(temp_path)  # Delete it so the main function can create it
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_structure_obj(mocker: MockerFixture) -> Any:
    """Create a mock structure object with all necessary properties for transformation.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock structure object
    """
    # Create a mock chain with coordinates
    mock_chain = mocker.MagicMock()
    mock_chain.id = "A"
    mock_chain.num_residues = 10

    # Add coordinates property
    mock_coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mock_chain.coordinates = mock_coordinates

    # Add secondary structure
    mock_helix = mocker.MagicMock()
    mock_helix.start = 0
    mock_helix.end = 9
    mock_chain.secondary_structure = [mock_helix]

    # Create the structure
    mock_structure = mocker.MagicMock()
    mock_structure.__iter__.return_value = [mock_chain]
    mock_structure.chains = {"A": mock_chain}

    # Add center_of_mass property needed for transformation
    mock_structure.center_of_mass = np.array([0.0, 0.0, 0.0])

    return mock_structure


def test_integration_main_minimal(
    temp_structure_file: str,
    temp_output_file: str,
    mock_structure_obj: Any,
    mocker: MockerFixture,
) -> None:
    """Test the main function with minimal arguments.

    Args:
        temp_structure_file: Path to a temporary valid structure file
        temp_output_file: Path to a temporary output file
        mock_structure_obj: Mock structure object
        mocker: Pytest mocker fixture
    """
    # Mock all the necessary components properly
    mock_parser = mocker.patch("flatprot.cli.commands.GemmiStructureParser")
    mock_parser.return_value.parse_structure.return_value = mock_structure_obj

    # Mock coordinate manager
    mock_coord_manager = mocker.patch("flatprot.cli.commands.create_coordinate_manager")
    mock_coord_manager.return_value = mocker.MagicMock()

    # Mock projection
    mock_apply_projection = mocker.patch("flatprot.cli.commands.apply_projection")
    mock_apply_projection.return_value = mocker.MagicMock()

    # Mock SVG generation
    mock_generate_svg = mocker.patch("flatprot.cli.commands.generate_svg")
    mock_generate_svg.return_value = "<svg></svg>"

    # Mock save SVG
    mock_save_svg = mocker.patch("flatprot.cli.commands.save_svg")

    # Mock logger instead of console output
    mocker.patch("flatprot.cli.commands.logger")

    # Mock file validation
    mocker.patch("flatprot.cli.commands.validate_structure_file")
    mocker.patch("flatprot.cli.commands.validate_optional_files")

    # Mock the CIF file check to return True to bypass DSSP requirement
    mocker.patch(
        "flatprot.cli.commands.Path.suffix",
        new_callable=mocker.PropertyMock,
        return_value=".cif",
    )

    # Run the main function
    result = project_structure_svg(
        structure=Path(temp_structure_file), output=Path(temp_output_file)
    )

    # Verify the result
    assert result == 0

    # Verify our mocks were called appropriately
    mock_parser.return_value.parse_structure.assert_called_once()
    mock_coord_manager.assert_called_once()
    mock_apply_projection.assert_called_once()
    mock_generate_svg.assert_called_once()
    mock_save_svg.assert_called_once()


def test_integration_main_all_options(
    temp_structure_file: str,
    temp_output_file: str,
    temp_matrix_file: str,
    temp_style_file: str,
    temp_annotations_file: str,
    mock_structure_obj: Any,
    mocker: MockerFixture,
) -> None:
    """Test the main function with all available options.

    Args:
        temp_structure_file: Path to a temporary valid structure file
        temp_output_file: Path to a temporary output file
        temp_matrix_file: Path to a temporary matrix file
        temp_style_file: Path to a temporary style file
        temp_annotations_file: Path to a temporary annotations file
        mock_structure_obj: Mock structure object
        mocker: Pytest mocker fixture
    """
    # Mock all the necessary components properly
    mock_parser = mocker.patch("flatprot.cli.commands.GemmiStructureParser")
    mock_parser.return_value.parse_structure.return_value = mock_structure_obj

    # Mock coordinate manager
    mock_coord_manager = mocker.patch("flatprot.cli.commands.create_coordinate_manager")
    mock_coord_manager.return_value = mocker.MagicMock()

    # Mock projection
    mock_apply_projection = mocker.patch("flatprot.cli.commands.apply_projection")
    mock_apply_projection.return_value = mocker.MagicMock()

    # Mock SVG generation
    mock_generate_svg = mocker.patch("flatprot.cli.commands.generate_svg")
    mock_generate_svg.return_value = "<svg></svg>"

    # Mock save SVG
    mock_save_svg = mocker.patch("flatprot.cli.commands.save_svg")

    # Mock logger instead of console output
    mocker.patch("flatprot.cli.commands.logger")

    # Mock file validation
    mocker.patch("flatprot.cli.commands.validate_structure_file")
    mocker.patch("flatprot.cli.commands.validate_optional_files")

    # Mock the CIF file check to return True to bypass DSSP requirement
    mocker.patch(
        "flatprot.cli.commands.Path.suffix",
        new_callable=mocker.PropertyMock,
        return_value=".cif",
    )

    # Run the main function with all options
    result = project_structure_svg(
        structure=Path(temp_structure_file),
        output=Path(temp_output_file),
        matrix=Path(temp_matrix_file),
        style=Path(temp_style_file),
        annotations=Path(temp_annotations_file),
    )

    # Verify the result
    assert result == 0

    # Verify our mocks were called appropriately
    mock_parser.return_value.parse_structure.assert_called_once()
    mock_coord_manager.assert_called_once()
    mock_apply_projection.assert_called_once()
    mock_generate_svg.assert_called_once()
    mock_save_svg.assert_called_once()


def test_integration_main_stdout_output(
    temp_structure_file: str,
    mock_structure_obj: Any,
    mocker: MockerFixture,
) -> None:
    """Test the main function with output to stdout.

    Args:
        temp_structure_file: Path to a temporary valid structure file
        mock_structure_obj: Mock structure object
        mocker: Pytest mocker fixture
    """
    mocker.patch("flatprot.cli.commands.logger.info")

    # Mock all the necessary components properly
    mock_parser = mocker.patch("flatprot.cli.commands.GemmiStructureParser")
    mock_parser.return_value.parse_structure.return_value = mock_structure_obj

    # Mock coordinate manager
    mock_coord_manager = mocker.patch("flatprot.cli.commands.create_coordinate_manager")
    mock_coord_manager.return_value = mocker.MagicMock()

    # Mock projection
    mock_apply_projection = mocker.patch("flatprot.cli.commands.apply_projection")
    mock_apply_projection.return_value = mocker.MagicMock()

    # Mock SVG generation
    mock_generate_svg = mocker.patch("flatprot.cli.commands.generate_svg")
    mock_generate_svg.return_value = "<svg></svg>"

    # Mock print function instead of logger.info since commands.py uses print() for stdout
    mock_print = mocker.patch("flatprot.cli.commands.print")

    # Mock file validation
    mocker.patch("flatprot.cli.commands.validate_structure_file")
    mocker.patch("flatprot.cli.commands.validate_optional_files")

    # Mock the CIF file check to return True to bypass DSSP requirement
    mocker.patch(
        "flatprot.cli.commands.Path.suffix",
        new_callable=mocker.PropertyMock,
        return_value=".cif",
    )

    # Run the main function without output parameter (should print to stdout)
    result = project_structure_svg(structure=Path(temp_structure_file), output=None)

    # Verify the result
    assert result == 0

    # Verify that print was called with the SVG content
    mock_print.assert_called_once_with("<svg></svg>")


def test_integration_main_flatprot_error(
    temp_structure_file: str,
    mocker: MockerFixture,
) -> None:
    """Test the main function with a FlatProtError."""
    # Mock file validation to avoid file access
    mocker.patch("flatprot.cli.commands.validate_structure_file")

    # Mock the CIF file check to return True to bypass DSSP requirement
    mocker.patch(
        "flatprot.cli.commands.Path.suffix",
        new_callable=mocker.PropertyMock,
        return_value=".cif",
    )

    # Mock to raise a FlatProtError
    error_message = "Test error message"
    mock_error = FlatProtError(error_message)
    mock_parser = mocker.patch("flatprot.cli.commands.GemmiStructureParser")
    mock_parser.return_value.parse_structure.side_effect = mock_error

    # Mock logger error
    mock_logger_error = mocker.patch("flatprot.cli.commands.logger.error")
    mocker.patch("flatprot.cli.commands.logger.info")
    # Run the main function
    result = project_structure_svg(structure=Path(temp_structure_file))

    # Verify the result
    assert result == 1

    # Verify that logger.error was called with the error message
    mock_logger_error.assert_called_once_with(error_message)


def test_integration_main_unexpected_error(
    temp_structure_file: str,
    mocker: MockerFixture,
) -> None:
    """Test the main function with an unexpected error.

    Args:
        temp_structure_file: Path to a temporary valid structure file
        mocker: Pytest mocker fixture
    """

    # Mock file validation to avoid file access
    mocker.patch("flatprot.cli.commands.validate_structure_file")

    # Mock to raise an unexpected error
    error_message = "Unexpected test error"
    mock_parser = mocker.patch("flatprot.cli.commands.GemmiStructureParser")
    mock_parser.return_value.parse_structure.side_effect = RuntimeError(error_message)

    # Mock logger error
    mock_logger_error = mocker.patch("flatprot.cli.commands.logger.error")
    mocker.patch("flatprot.cli.commands.logger.info")

    # Mock the CIF file check to return True to bypass DSSP requirement
    mocker.patch(
        "flatprot.cli.commands.Path.suffix",
        new_callable=mocker.PropertyMock,
        return_value=".cif",
    )

    # Run the main function
    result = project_structure_svg(structure=Path(temp_structure_file))

    # Verify the result
    assert result == 1

    # Verify that logger.error was called with the error message
    # In commands.py, it uses logger.error(f"Unexpected error: {str(e)}")
    mock_logger_error.assert_called_once_with(f"Unexpected error: {error_message}")


def test_print_success_summary(mocker: MockerFixture) -> None:
    """Test the print_success_summary function.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock logger info instead of console output
    mock_logger_info = mocker.patch("flatprot.cli.commands.logger.info")

    # Test with all parameters
    structure_path = Path("structure.pdb")
    output_path = Path("output.svg")
    matrix_path = Path("matrix.npy")
    style_path = Path("style.toml")
    annotations_path = Path("annotations.toml")

    print_success_summary(
        structure_path=structure_path,
        output_path=output_path,
        matrix_path=matrix_path,
        style_path=style_path,
        annotations_path=annotations_path,
        dssp_path=None,
    )

    # Verify logger.info was called with expected messages
    assert mock_logger_info.call_count >= 1  # At least one call to logger.info


def test_print_success_summary_minimal(mocker: MockerFixture) -> None:
    """Test the print_success_summary function with minimal parameters.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock logger info instead of console output
    mock_logger_info = mocker.patch("flatprot.cli.commands.logger.info")

    # Test with minimal parameters
    structure_path = Path("structure.pdb")

    print_success_summary(
        structure_path=structure_path,
        output_path=None,
        matrix_path=None,
        style_path=None,
        annotations_path=None,
        dssp_path=None,
    )

    # Verify logger.info was called with expected messages
    assert mock_logger_info.call_count >= 1  # At least one call to logger.info

    # Verify logger.info was called with expected messages
    assert mock_logger_info.call_count >= 1  # At least one call to logger.info
