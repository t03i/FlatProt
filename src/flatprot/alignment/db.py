# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

import h5py
import numpy as np

from flatprot.transformation import TransformationMatrix


@dataclass
class AlignmentDBEntry:
    """Stores alignment data with its rotation matrix."""

    rotation_matrix: TransformationMatrix
    entry_id: str
    structure_name: str
    metadata: Optional[Dict[str, float | str]] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlignmentDBEntry):
            return False
        return (
            np.allclose(
                self.rotation_matrix.to_array(), other.rotation_matrix.to_array()
            )
            and self.entry_id == other.entry_id
            and self.structure_name == other.structure_name
        )


class AlignmentDatabase:
    """Handles alignment database using HDF5 storage with memory-mapped arrays."""

    def __init__(self, path: Path):
        self.path = path
        self._file: Optional[h5py.File] = None
        self._structure_name_index: dict[str, str] = {}  # structure_name -> entry_id

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        """Opens the database file in read/write mode."""
        self._file = h5py.File(self.path, "a")
        self._load_index()

    def close(self) -> None:
        """Closes the database file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _load_index(self) -> None:
        """Loads structure name index from HDF5 file."""
        if "index" not in self._file:
            return

        names = self._file["index/structure_names"][:]
        ids = self._file["index/entry_ids"][:]
        self._structure_name_index = dict(zip(names, ids))

    def _save_index(self) -> None:
        """Saves structure name index to HDF5 file."""
        if "index" in self._file:
            del self._file["index"]

        index = self._file.create_group("index")
        names = list(self._structure_name_index.keys())
        ids = list(self._structure_name_index.values())
        index.create_dataset("structure_names", data=names)
        index.create_dataset("entry_ids", data=ids)

    def contains_entry_id(self, entry_id: str) -> bool:
        """Checks if an entry_id exists in the database. O(1) lookup."""
        if self._file is None:
            raise RuntimeError("Database not opened")
        return entry_id in self._file

    def contains_structure_name(self, structure_name: str) -> bool:
        """Checks if a structure_name exists in the database. O(1) lookup."""
        return structure_name in self._structure_name_index

    def get_by_entry_id(
        self, entry_id: str, default: Optional[AlignmentDBEntry] = None
    ) -> Optional[AlignmentDBEntry]:
        """Returns alignment entry for given entry_id or default if not found. O(1) lookup."""
        if self._file is None:
            raise RuntimeError("Database not opened")
        if entry_id not in self._file:
            return default

        entry_group = self._file[entry_id]
        metadata = {k: v for k, v in entry_group.attrs.items() if k != "structure_name"}
        return AlignmentDBEntry(
            rotation_matrix=TransformationMatrix.from_array(entry_group["rotation"][:]),
            entry_id=entry_id,
            structure_name=entry_group.attrs["structure_name"],
            metadata=metadata if metadata else None,
        )

    def get_by_structure_name(
        self, structure_name: str, default: Optional[AlignmentDBEntry] = None
    ) -> Optional[AlignmentDBEntry]:
        """Returns alignment entry for given structure_name or default if not found. O(1) lookup."""
        if structure_name not in self._structure_name_index:
            return default
        entry_id = self._structure_name_index[structure_name]
        return self.get_by_entry_id(entry_id)

    def add_entry(self, entry: AlignmentDBEntry) -> None:
        """Adds a new entry to the database."""
        if self._file is None:
            raise RuntimeError("Database not opened")

        if entry.entry_id in self._file:
            raise ValueError(f"Entry ID {entry.entry_id} already exists")

        if entry.structure_name in self._structure_name_index:
            raise ValueError(f"Structure name {entry.structure_name} already exists")

        # Create entry group and save data
        entry_group = self._file.create_group(entry.entry_id)
        entry_group.create_dataset("rotation", data=entry.rotation_matrix.to_array())
        entry_group.attrs["structure_name"] = entry.structure_name

        # Store metadata if available
        if entry.metadata:
            for key, value in entry.metadata.items():
                entry_group.attrs[key] = value

        # Update index
        self._structure_name_index[entry.structure_name] = entry.entry_id
        self._save_index()

    def update(self, entry: AlignmentDBEntry) -> None:
        """Updates an existing entry in the database."""
        if self._file is None:
            raise RuntimeError("Database not opened")

        if entry.entry_id not in self._file:
            raise KeyError(f"Entry ID {entry.entry_id} not found in database")

        old_entry = self.get_by_entry_id(entry.entry_id)

        # Check structure name conflicts
        if (
            entry.structure_name != old_entry.structure_name
            and entry.structure_name in self._structure_name_index
        ):
            raise ValueError(f"Structure name {entry.structure_name} already exists")

        # Update index if structure name changed
        if entry.structure_name != old_entry.structure_name:
            del self._structure_name_index[old_entry.structure_name]
            self._structure_name_index[entry.structure_name] = entry.entry_id

        # Update entry data
        del self._file[entry.entry_id]
        entry_group = self._file.create_group(entry.entry_id)
        entry_group.create_dataset("rotation", data=entry.rotation_matrix.to_array())
        entry_group.attrs["structure_name"] = entry.structure_name

        # Store metadata if available
        if entry.metadata:
            for key, value in entry.metadata.items():
                entry_group.attrs[key] = value

        # Save index
        self._save_index()
