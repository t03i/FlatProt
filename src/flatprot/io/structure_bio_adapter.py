# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Atom import Atom
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1

from flatprot.structure.components import Protein, Helix, Sheet, Loop
from flatprot.structure.residue import Residue
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
        self, structure: Structure, chain_id: str, structure_file: Path
    ) -> list[str]:
        """Get secondary structure assignment from mmCIF annotation or DSSP file."""
        if structure_file.suffix.lower() in (".cif", ".mmcif", ".bcif"):
            ss_list = self._get_ss_from_mmcif(structure, chain_id)
            if ss_list is not None:
                return ss_list

        # Try DSSP file
        ss_list = self._get_ss_from_dssp(structure, chain_id, structure_file)
        if ss_list is not None:
            return ss_list

        # Default: return all coil
        return ["-"] * len(list(structure[0][chain_id]))

    def _get_ss_from_mmcif_file(
        self, structure: Structure, chain_id: str
    ) -> list[str] | None:
        """Extract secondary structure information from mmCIF file."""
        try:
            mmcif_dict = structure.header["_struct_conf"]
            if not mmcif_dict:
                return None

            ss_list = ["-"] * len(list(structure[0][chain_id]))

            for conf in mmcif_dict:
                if conf["pdbx_beg_auth_asym_id"] == chain_id:
                    start = int(conf["beg_auth_seq_id"])
                    end = int(conf["end_auth_seq_id"])
                    ss_type = "H" if conf["conf_type_id"].startswith("HELX") else "E"

                    self._update_ss_list(
                        structure, chain_id, ss_list, start, end, ss_type
                    )

            return ss_list
        except (KeyError, AttributeError):
            return None

    def _get_ss_from_dssp_file(
        self, structure: Structure, chain_id: str, structure_file: Path
    ) -> list[str] | None:
        """Extract secondary structure information from DSSP file."""
        dssp_file = structure_file.with_suffix(".dssp")
        if not dssp_file.exists():
            return None

        try:
            dssp_dict = dssp_dict_from_pdb_file(str(structure_file), str(dssp_file))
            ss_list = []

            for residue in structure[0][chain_id]:
                key = (chain_id, residue.id)
                ss = dssp_dict.get(key, [None, "-"])[1]
                ss_list.append(ss)

            return ss_list
        except Exception:
            return None

    def _update_ss_list(
        self,
        structure: Structure,
        chain_id: str,
        ss_list: list[str],
        start: int,
        end: int,
        ss_type: str,
    ) -> None:
        """Update the secondary structure list for a specific range."""
        for i in range(start, end + 1):
            idx = next(
                (
                    idx
                    for idx, res in enumerate(structure[0][chain_id])
                    if res.id[1] == i
                ),
                None,
            )
            if idx is not None:
                ss_list[idx] = ss_type

    def _chain_to_protein(
        self, chain: Chain, structure: Structure, structure_file: Path
    ) -> Protein:
        """Convert a Bio.PDB Chain to our Protein model."""
        # Parse basic chain data
        indices, names, coords = self._parse_chain_data(chain)

        # Convert to numpy arrays
        index = np.array(indices, dtype=np.int32)
        coordinates = np.array(coords, dtype=np.float32)

        # Create residue objects - convert 3-letter codes to Residue enum
        residues: list[Residue] = []
        for resname in names:
            # Convert 3-letter code to 1-letter code and create Residue enum
            # Default to XAA (X) if residue is unknown
            try:
                one_letter = protein_letters_3to1[resname]
                residues.append(Residue(one_letter))
            except KeyError:
                residues.append(Residue.XAA)

        # Get secondary structure
        ss_list = self._get_secondary_structure(structure, chain.id, structure_file)

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

    def _structure_to_proteins(
        self, structure: Structure, structure_file: Path
    ) -> dict[str, Protein]:
        """Convert a Bio.PDB Structure to a dictionary of Protein models."""
        proteins = {}
        for model in structure:
            for chain in model:
                proteins[chain.id] = self._chain_to_protein(
                    chain, structure, structure_file
                )
        return proteins

    def parse_structure(self, structure_file: Path) -> dict[str, Protein]:
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
        return self._structure_to_proteins(structure, structure_file)

    def _protein_to_chain(self, protein: Protein, chain_id: str) -> Chain:
        """Convert a Protein model to a Bio.PDB Chain."""
        chain = Chain(chain_id)

        for idx, (residue_idx, residue, coord) in enumerate(
            zip(protein.index, protein.residues, protein.coordinates)
        ):
            # Create Bio.PDB Residue
            bio_residue = BioResidue(
                (" ", residue_idx, " "),  # id
                residue.name,  # resname
                "",  # segid
            )

            # Create CA atom
            ca_atom = Atom(
                name="CA",
                coord=coord,
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=" CA ",
                serial_number=idx + 1,
                element="C",
            )

            bio_residue.add(ca_atom)
            chain.add(bio_residue)

        return chain

    def save_structure(self, structure: dict[str, Protein], output_file: Path) -> None:
        """Save structure to PDB file.

        Args:
            structure: Dictionary mapping chain IDs to Protein objects
            output_file: Path where to save the PDB file
        """
        # Create new Bio.PDB Structure
        bio_structure = Structure("protein")
        model = Model(0)
        bio_structure.add(model)

        # Convert each protein to a chain
        for chain_id, protein in structure.items():
            chain = self._protein_to_chain(protein, chain_id)
            model.add(chain)

        # Write structure to file
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(str(output_file))
