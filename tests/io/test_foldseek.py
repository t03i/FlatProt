import pytest
import numpy as np
from pathlib import Path
import tempfile

from flatprot.alignment.foldseek import FoldseekAligner, _parse_foldseek_vector


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_foldseek_executable(temp_dir):
    executable = temp_dir / "mock_foldseek"
    # Create mock executable that writes dummy results with correct format
    with open(executable, "w") as f:
        f.write(
            """#!/bin/bash
echo -e "query\\ttarget\\t1\\t100\\t1\\t100\\tSEQ\\t0.9\\t0.8\\t(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)\\t(0.0,0.0,0.0)\\t(1.0,1.0,1.0)" > $4
"""
        )
    executable.chmod(0o755)
    return str(executable)


class TestFoldseekAligner:
    def test_parse_foldseek_vector(self):
        vector_str = "(1.0, 2.0, 3.0)"
        result = _parse_foldseek_vector(vector_str)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_align_structure(self, temp_dir, mock_foldseek_executable):
        aligner = FoldseekAligner(
            foldseek_executable=mock_foldseek_executable,
            database_path=temp_dir / "mock_db",
        )

        # Create dummy structure file
        structure_file = temp_dir / "test.pdb"
        structure_file.touch()

        result = aligner.align_structure(structure_file)

        assert result is not None
        assert result.db_id == "target"
        assert result.probability == 0.9
        np.testing.assert_array_equal(result.aligned_region, np.array([1, 100]))
        np.testing.assert_almost_equal(result.rotation_matrix.rotation, np.eye(3))
        np.testing.assert_array_equal(result.rotation_matrix.translation, np.zeros(3))
