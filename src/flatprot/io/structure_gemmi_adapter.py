# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional

import numpy as np
import gemmi

from flatprot.core import ResidueType, SecondaryStructureType, Structure, Chain
from .dssp import parse_dssp

from .structure import StructureParser


class GemmiStructureParser(StructureParser):
    def parse_structure(
        self, structure_file: Path, secondary_structure_file: Optional[Path] = None
    ) -> Structure:
        """Main entry point for structure parsing"""
        # 1. Parse structure
        structure = self._parse_structure_file(structure_file)

        # 2. Process each chain
        chains = []
        for chain in structure[0]:
            # Extract basic chain data
            chain_data = self._parse_chain_data(chain)
            # Get secondary structure
            ss_regions = self._get_secondary_structure(
                structure, secondary_structure_file
            )

            chain_obj = Chain(chain.name, **chain_data)
            for region in ss_regions:
                chain_obj.add_secondary_structure(region[0], region[1], region[2])
            chains.append(chain_obj)

        # Assign structure ID from filename stem
        structure_id = structure_file.stem
        return Structure(chains, id=structure_id)

    def _parse_structure_file(self, structure_file: Path) -> gemmi.Structure:
        """Parse structure from file using gemmi"""
        structure = gemmi.read_structure(str(structure_file))
        return structure

    def _parse_chain_data(self, chain: gemmi.Chain) -> dict:
        """Extract basic chain data using gemmi"""
        residue_indices = []
        residue_names = []
        coordinates = []

        def get_ca_coordinates(residue: gemmi.Residue) -> np.ndarray:
            for atom in residue:
                if atom.name == "CA":
                    return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            return None

        for residue in chain:
            coordinate = get_ca_coordinates(residue)
            if coordinate is not None:
                coordinates.append(coordinate)
                residue_indices.append(residue.seqid.num)
                residue = gemmi.find_tabulated_residue(residue.name).one_letter_code
                residue = "X" if not residue.isupper() else residue
                residue_names.append(ResidueType(residue))
        assert len(residue_indices) == len(coordinates)
        assert len(residue_indices) == len(residue_names)
        return {
            "index": residue_indices,
            "residues": residue_names,
            "coordinates": np.array(coordinates, dtype=np.float32),
        }

    def _get_secondary_structure(
        self,
        structure: gemmi.Structure,
        secondary_structure_file: Optional[Path] = None,
    ) -> list[tuple[SecondaryStructureType, int, int]]:
        if secondary_structure_file is not None:
            return parse_dssp(secondary_structure_file)
        else:
            return self._get_secondary_structure_cif(structure)

    def _get_secondary_structure_cif(
        self, structure: gemmi.Structure
    ) -> list[tuple[SecondaryStructureType, int, int]]:
        """Get secondary structure from gemmi structure"""
        ss_regions = []

        # Extract helices and sheets from gemmi structure
        for helix in structure.helices:
            start = helix.start.res_id.seqid.num
            end = helix.end.res_id.seqid.num
            ss_regions.append((SecondaryStructureType.HELIX, start, end))

        for sheet in structure.sheets:
            for strand in sheet.strands:
                start = strand.start.res_id.seqid.num
                end = strand.end.res_id.seqid.num
                ss_regions.append((SecondaryStructureType.SHEET, start, end))

        return ss_regions

    def save_structure(
        self, structure: Structure, output_file: Path, separate_chains=False
    ) -> None:
        """Save structure using gemmi"""
        gemmi_structure = gemmi.Structure()
        model = gemmi.Model("1")

        for chain_id, chain_data in structure.items():
            chain = gemmi.Chain(chain_id)

            for idx, (residue_idx, residue, coord) in enumerate(
                zip(chain_data.index, chain_data.residues, chain_data.coordinates)
            ):
                gemmi_res = gemmi.Residue()
                gemmi_res.name = residue.name
                gemmi_res.seqid = gemmi.SeqId(residue_idx)

                ca = gemmi.Atom()
                ca.name = "CA"
                ca.pos = gemmi.Position(*coord)
                ca.element = gemmi.Element("C")
                gemmi_res.add_atom(ca)

                chain.add_residue(gemmi_res)

            model.add_chain(chain)

        gemmi_structure.add_model(model)
        gemmi_structure.write_pdb(str(output_file))
