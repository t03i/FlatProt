import pytest
import numpy as np

from flatprot.alignment.utils import AlignmentResult, get_aligned_rotation_database
from flatprot.alignment.db import AlignmentDatabase, AlignmentDBEntry
from flatprot.transformation import TransformationMatrix
from flatprot.alignment.errors import DatabaseEntryNotFoundError


@pytest.fixture
def sample_alignment():
    """Create a sample alignment result for testing.
    NOTE: The rotation_matrix here represents the Query -> Target transform,
    as returned by the modified FoldseekAligner.
    """
    # Original T->Q: R=[[0,-1,0],[1,0,0],[0,0,1]], t=[0,1,0]
    # Inverse Q->T: R_inv=R.T=[[0,1,0],[-1,0,0],[0,0,1]], t_inv=-R_inv@t=[-1,0,0]
    query_to_target_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    query_to_target_translation = np.array([-1.0, 0.0, 0.0])

    return AlignmentResult(
        db_id="test_entry",
        probability=0.9,
        aligned_region=np.array([1, 100]),
        alignment_scores=np.ones(100),
        rotation_matrix=TransformationMatrix(
            rotation=query_to_target_rotation,
            translation=query_to_target_translation,
        ),
    )


@pytest.fixture
def mock_db(tmp_path):
    """Create a mock database with a test entry."""
    db_path = tmp_path / "test.h5"
    db = AlignmentDatabase(db_path)

    # Create and add test entry
    entry = AlignmentDBEntry(
        rotation_matrix=TransformationMatrix(
            rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            translation=np.array([0.0, 1.0, 0.0]),
        ),
        entry_id="test_entry",
        structure_name="test_structure",
    )

    with db:
        db.add_entry(entry)
        yield db


def test_alignment_to_db_rotation(sample_alignment, mock_db):
    """Test combining alignment rotation with database rotation."""
    result, _ = get_aligned_rotation_database(sample_alignment, mock_db, lambda x: x)

    # With the corrected logic (inverse alignment then db rotation):
    # Expected Rotation = R_db @ R_align_inv = (90_deg) @ (-90_deg) = Identity
    # Expected Translation = R_db @ t_align_inv + t_db = (90_deg @ [-1, 0, 0]) + [0, 1, 0] = [0, -1, 0] + [0, 1, 0] = [0, 0, 0]
    expected_rotation = np.identity(3)
    expected_translation = np.array([0.0, 0.0, 0.0])

    np.testing.assert_array_almost_equal(result.rotation, expected_rotation)
    np.testing.assert_array_almost_equal(result.translation, expected_translation)


def test_alignment_to_db_rotation_invalid_id(sample_alignment, mock_db):
    """Test handling of invalid database IDs."""
    invalid_alignment = sample_alignment._replace(db_id="nonexistent")
    with pytest.raises(DatabaseEntryNotFoundError) as excinfo:
        get_aligned_rotation_database(invalid_alignment, mock_db)
    assert "Database entry nonexistent not found" in str(excinfo.value)
