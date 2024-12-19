# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pathlib import Path

from flatprot.io.structure_gemmi_adapter import GemmiStructureParser
from flatprot.structure.components import Chain, Structure
from flatprot.structure.secondary import SecondaryStructureType
from flatprot.io.dssp import parse_dssp


class TestStructureParser:
    def test_parse_annotated_structure(self, tmp_path):
        parser = GemmiStructureParser()

        # Use the actual PDB file
        test_file = Path("tests/data/test.cif")

        # Test parsing
        result = parser.parse_structure(test_file)

        # Assertions
        assert isinstance(result, Structure)
        assert "A" in result
        assert isinstance(result["A"], Chain)

        protein = result["A"]
        # The PDB file has 72 residues in chain A
        assert len(protein.residues) == 72
        # Test some specific residues from the file
        assert protein.residues[0].name == "LEU"  # First residue
        assert protein.residues[5].name == "CYS"  # CYS at position 6
        assert protein.residues[71].name == "ARG"  # Last residue

        # Test coordinate access for a specific atom
        # Testing coordinates for CYS 6 SG atom (line 52 in PDB)
        expected_coords = np.array([[-6.903, -7.615, 4.269]])
        np.testing.assert_array_almost_equal(
            protein.get_coordinates_for_index(49),  # Atom index 49 from PDB
            expected_coords,
        )

        # New tests for secondary structure elements
        ss_elements = result["A"].secondary_structure
        # Test beta sheets
        # Sheet A: residues 2-5 and 14-17
        assert any(
            ss.type == SecondaryStructureType.SHEET and ss.start == 1 and ss.end == 4
            for ss in ss_elements
        )
        assert any(
            ss.type == SecondaryStructureType.SHEET and ss.start == 13 and ss.end == 16
            for ss in ss_elements
        )

        # Sheet B: residues 6-6, 24-30, 38-44, and 60-65
        assert any(
            ss.type == SecondaryStructureType.SHEET and ss.start == 5 and ss.end == 5
            for ss in ss_elements
        )

        # Test helices
        # Alpha helix: residues 46-54
        assert any(
            ss.type == SecondaryStructureType.HELIX and ss.start == 45 and ss.end == 53
            for ss in ss_elements
        )

        # Left-handed helix: residues 18-19
        assert any(
            ss.type == SecondaryStructureType.HELIX and ss.start == 17 and ss.end == 18
            for ss in ss_elements
        )

        # Count total secondary structure elements
        sheet_count = sum(
            1 for ss in ss_elements if ss.type == SecondaryStructureType.SHEET
        )
        helix_count = sum(
            1 for ss in ss_elements if ss.type == SecondaryStructureType.HELIX
        )

        # Based on the CIF file structure
        assert sheet_count == 6  # Total number of strands in sheets A and B
        assert helix_count == 2  # One alpha helix and one left-handed helix

    def test_parse_structure_with_dssp(self, tmp_path):
        parser = GemmiStructureParser()

        # Use the test PDB and DSSP files
        pdb_file = Path("tests/data/test.pdb")
        dssp_file = Path("tests/data/test.dssp")

        # Test parsing with DSSP
        result = parser.parse_structure(pdb_file, dssp_file)

        # Basic structure assertions
        assert isinstance(result, Structure)
        assert "A" in result
        assert isinstance(result["A"], Chain)

        protein = result["A"]
        # Verify residue count (adjust number based on your test file)
        assert len(protein.residues) == 72  # Adjust if different

        # Test some specific residues
        assert protein.residues[0].name == "LEU"  # First residue
        assert protein.residues[5].name == "CYS"  # Verify specific residue
        assert protein.residues[-1].name == "ARG"  # Last residue

        # Test coordinate access for a specific atom
        expected_coords = np.array([[-6.903, -7.615, 4.269]])  # Adjust coordinates
        np.testing.assert_array_almost_equal(
            protein.get_coordinates_for_index(49),  # Adjust index as needed
            expected_coords,
        )

        # Test secondary structure elements from DSSP
        ss_elements = result["A"].secondary_structure

        # Test beta sheets (adjust indices based on DSSP file)
        assert any(
            ss.type == SecondaryStructureType.SHEET and ss.start == 1 and ss.end == 4
            for ss in ss_elements
        )

        # Test helices (adjust indices based on DSSP file)
        assert any(
            ss.type == SecondaryStructureType.HELIX and ss.start == 45 and ss.end == 53
            for ss in ss_elements
        )

        # Count secondary structure elements
        sheet_count = sum(
            1 for ss in ss_elements if ss.type == SecondaryStructureType.SHEET
        )
        helix_count = sum(
            1 for ss in ss_elements if ss.type == SecondaryStructureType.HELIX
        )

        # Adjust these numbers based on your DSSP file
        assert sheet_count == 6  # Adjust based on DSSP content
        assert helix_count == 2  # Adjust based on DSSP content

        # Additional DSSP-specific tests
        # Test accessibility values if available
        # Test hydrogen bonding if available
        # Test phi/psi angles if available

    def test_dssp_parser(self):
        # Test parsing of raw DSSP file
        dssp_file = Path("tests/data/test.dssp")
        segments = parse_dssp(dssp_file)

        # Verify the segments are correctly parsed
        # Each segment should be a tuple of (SecondaryStructureType, start, end)
        assert isinstance(segments, list)
        assert all(isinstance(s, tuple) and len(s) == 3 for s in segments)
        assert all(isinstance(s[0], SecondaryStructureType) for s in segments)

        # Test specific segments from the DSSP file
        # Sheet segments (based on your test.dssp file content)
        sheet_segments = [s for s in segments if s[0] == SecondaryStructureType.SHEET]
        assert (
            len(sheet_segments) == 6
        )  # Should match the number in test_parse_structure_with_dssp

        # Helix segments
        helix_segments = [s for s in segments if s[0] == SecondaryStructureType.HELIX]
        assert (
            len(helix_segments) == 2
        )  # Should match the number in test_parse_structure_with_dssp

        # Test specific segment positions
        # These numbers should match your test.dssp file content
        assert any(
            seg[0] == SecondaryStructureType.SHEET and seg[1] == 2 and seg[2] == 5
            for seg in segments
        )
        assert any(
            seg[0] == SecondaryStructureType.HELIX and seg[1] == 46 and seg[2] == 54
            for seg in segments
        )
