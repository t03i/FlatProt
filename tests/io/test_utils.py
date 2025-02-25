import pytest
import numpy as np

from flatprot.alignment.utils import AlignmentResult, alignment_to_db_rotation
from flatprot.alignment.db import AlignmentDatabase, AlignmentDBEntry
from flatprot.transformation import TransformationMatrix


@pytest.fixture
def sample_alignment():
    return AlignmentResult(
        db_id="test_entry",
        probability=0.9,
        aligned_region=np.array([1, 100]),
        alignment_scores=np.ones(100),
        rotation_matrix=TransformationMatrix(
            rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            translation=np.array([0.0, 1.0, 0.0]),
        ),
    )


@pytest.fixture
def mock_db(tmp_path):
    db_path = tmp_path / "test.h5"
    db = AlignmentDatabase(db_path)
    db.open()

    entry = AlignmentDBEntry(
        rotation_matrix=TransformationMatrix(
            rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            translation=np.array([0.0, 1.0, 0.0]),
        ),
        entry_id="test_entry",
        structure_name="test_structure",
    )
    db.add_entry(entry)
    db.close()
    return db


def test_alignment_to_db_rotation(sample_alignment, mock_db):
    result = alignment_to_db_rotation(sample_alignment, mock_db)

    # The combined rotation should be the composition of both rotations
    # In this case, both rotations are the same 90-degree rotation,
    # so the result should be a 180-degree rotation
    expected_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    np.testing.assert_array_almost_equal(result.rotation, expected_rotation)
    # No need to test translation as it's not part of the combined_rotation operation


def test_alignment_to_db_rotation_invalid_id(sample_alignment, mock_db):
    invalid_alignment = sample_alignment._replace(db_id="nonexistent")
    with pytest.raises(ValueError):
        alignment_to_db_rotation(invalid_alignment, mock_db)
