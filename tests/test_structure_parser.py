# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pathlib import Path

from flatprot.io.structure_bio_adapter import BiopythonStructureParser
from flatprot.structure.components import Protein


class TestStructureParser:
    def test_parse_structure(self, tmp_path):
        parser = BiopythonStructureParser()

        # Use the actual PDB file
        test_file = Path("tests/data/None-Nana_c1_1-Naja_naja.pdb")

        # Test parsing
        result = parser.parse_structure(test_file)

        # Assertions
        assert isinstance(result, dict)
        assert "A" in result  # Chain ID from the PDB
        assert isinstance(result["A"], Protein)

        protein = result["A"]
        # The PDB file has 72 residues in chain A
        assert len(protein.residues) == 72

        # Test some specific residues from the file
        assert protein.residues[0].name == "ALA"  # First residue
        assert protein.residues[5].name == "CYS"  # CYS at position 6
        assert protein.residues[71].name == "ARG"  # Last residue

        # Test coordinate access for a specific atom
        # Testing coordinates for CYS 6 SG atom (line 52 in PDB)
        expected_coords = np.array([[6.691, 2.671, 3.234]])
        np.testing.assert_array_almost_equal(
            protein.get_coordinates_for_index(49),  # Atom index 49 from PDB
            expected_coords,
        )
