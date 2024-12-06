# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
import numpy as np
import os

from flatprot.structure.components import Protein, Helix, Sheet, Loop, Residue
from flatprot.structure.base import SubStructureComponent
from .structure import StructureParser


def find_ss_regions(
    ss_list: list[str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Find continuous regions of secondary structure elements."""
    helices = []
    sheets = []
    loops = []

    current_type = None
    start_idx = 0

    for i, (ss, res_idx) in enumerate(ss_list):
        if ss in ("H", "G", "I"):  # Helix types
            ss_type = "helix"
        elif ss in ("E", "B"):  # Sheet types
            ss_type = "sheet"
        else:
            ss_type = "loop"

        if current_type != ss_type:
            if current_type:
                end_idx = i - 1
                if current_type == "helix":
                    helices.append((start_idx, end_idx))
                elif current_type == "sheet":
                    sheets.append((start_idx, end_idx))
                else:
                    loops.append((start_idx, end_idx))
            current_type = ss_type
            start_idx = res_idx

    # Handle last element
    end_idx = len(ss_list) - 1
    if current_type == "helix":
        helices.append((start_idx, end_idx))
    elif current_type == "sheet":
        sheets.append((start_idx, end_idx))
    else:
        loops.append((start_idx, end_idx))

    return helices, sheets, loops


class BiopythonStructureParser(StructureParser):
    def _parse_chain_data(
        self, chain: Chain
    ) -> tuple[list[int], list[str], np.ndarray]:
        """Extract basic chain data: residue numbers, names, and coordinates."""
        residue_indices = []
        residue_names = []
        coordinates = []

        for residue in chain:
            if "CA" in residue:
                residue_indices.append(residue.id[1])
                residue_names.append(residue.resname)
                coordinates.append(residue["CA"].get_coord())

        return (residue_indices, residue_names, np.array(coordinates, dtype=np.float32))

    def _get_secondary_structure(
        self, structure: Structure, chain_id: str
    ) -> list[str]:
        """Get secondary structure assignment for a chain."""
        # Use DSSP to get secondary structure
        dssp_dict = dssp_dict_from_pdb_file(structure)[0]
        ss_list = []

        for residue in structure[0][chain_id]:
            if residue.id[1] in dssp_dict:
                ss = dssp_dict[residue.id[1]][2]  # Get SS assignment
                ss_list.append(ss)
            else:
                ss_list.append("-")  # No SS assignment

        return ss_list

    def _chain_to_protein(self, chain: Chain, structure: Structure) -> Protein:
        """Convert a Bio.PDB Chain to our Protein model."""
        # Parse basic chain data
        indices, names, coords = self._parse_chain_data(chain)

        # Convert to numpy arrays
        index = np.array(indices, dtype=np.int32)
        coordinates = np.array(coords, dtype=np.float32)

        # Create residue objects
        residues = [Residue(idx, name) for idx, name in zip(indices, names)]

        # Get secondary structure
        ss_list = self._get_secondary_structure(structure, chain.id)

        # Find secondary structure regions
        helices, sheets, loops = find_ss_regions(ss_list, indices)

        def _create_ss_component(
            ss_region, component_class: type[SubStructureComponent], parent: Protein
        ):
            return component_class(
                parent=parent,
                start_idx=ss_region.start,
                end_idx=ss_region.end,
            )

        # Create protein with all components
        protein = Protein(
            residues=residues,
            index=index,
            coordinates=coordinates,
        )

        # Create secondary structure components
        ss_components = []
        ss_components.extend(_create_ss_component(h, Helix) for h in helices)
        ss_components.extend(_create_ss_component(s, Sheet) for s in sheets)
        ss_components.extend(_create_ss_component(l, Loop) for l in loops)

        return protein

    def _structure_to_proteins(self, structure: Structure) -> dict[str, Protein]:
        """Convert a Bio.PDB Structure to a dictionary of Protein models."""
        proteins = {}
        for model in structure:
            for chain in model:
                proteins[chain.id] = self._chain_to_protein(chain, structure)
        return proteins

    def parse_structure(self, structure_file: str) -> dict[str, Protein]:
        """Parse structure from various file formats (PDB, mmCIF, BCIF).

        Args:
            structure_file: Path to structure file (.pdb, .cif, .bcif)

        Returns:
            Dictionary mapping chain IDs to Protein objects
        """
        # Determine file format from extension
        _, ext = os.path.splitext(structure_file)
        ext = ext.lower()

        if ext == ".pdb":
            parser = PDBParser(QUIET=True)
        elif ext in (".cif", ".mmcif", ".bcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Supported formats: .pdb, .cif, .mmcif, .bcif"
            )

        structure = parser.get_structure("protein", structure_file)
        return self._structure_to_proteins(structure)
