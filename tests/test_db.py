import pytest
import numpy as np
from pathlib import Path
import tempfile

from flatprot.alignment.db import AlignmentDatabase, AlignmentDBEntry
from flatprot.transformation import TransformationMatrix


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        yield Path(f.name)


@pytest.fixture
def sample_entry():
    return AlignmentDBEntry(
        rotation_matrix=TransformationMatrix(
            rotation=np.eye(3), translation=np.zeros(3)
        ),
        entry_id="test_entry",
        structure_name="test_structure",
    )


class TestAlignmentDatabase:
    def test_context_manager(self, temp_db_path):
        with AlignmentDatabase(temp_db_path) as db:
            assert db._file is not None
        assert db._file is None

    def test_add_entry(self, temp_db_path, sample_entry):
        with AlignmentDatabase(temp_db_path) as db:
            db.add_entry(sample_entry)
            assert db.contains_entry_id(sample_entry.entry_id)
            assert db.contains_structure_name(sample_entry.structure_name)

    def test_get_by_entry_id(self, temp_db_path, sample_entry):
        with AlignmentDatabase(temp_db_path) as db:
            db.add_entry(sample_entry)
            retrieved = db.get_by_entry_id(sample_entry.entry_id)
            assert retrieved == sample_entry

    def test_get_by_structure_name(self, temp_db_path, sample_entry):
        with AlignmentDatabase(temp_db_path) as db:
            db.add_entry(sample_entry)
            retrieved: AlignmentDBEntry | None = db.get_by_structure_name(
                sample_entry.structure_name
            )
            assert retrieved == sample_entry

    def test_update_entry(self, temp_db_path, sample_entry):
        with AlignmentDatabase(temp_db_path) as db:
            db.add_entry(sample_entry)

            updated_entry = AlignmentDBEntry(
                rotation_matrix=sample_entry.rotation_matrix,
                entry_id=sample_entry.entry_id,
                structure_name="updated_name",
            )
            db.update(updated_entry)

            retrieved = db.get_by_entry_id(sample_entry.entry_id)
            assert retrieved.structure_name == "updated_name"

    def test_duplicate_entry_error(self, temp_db_path, sample_entry):
        with AlignmentDatabase(temp_db_path) as db:
            db.add_entry(sample_entry)
            with pytest.raises(ValueError):
                db.add_entry(sample_entry)
