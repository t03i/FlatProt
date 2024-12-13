# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional
from collections import namedtuple

import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import (
    MMCIFParser,
    PDBParser,
    Structure as BioStructure,
    DSSP,
    Chain as BioChain,
    Model,
    Atom,
    Residue as BioResidue,
    MMCIFIO,
)

from flatprot.structure.residue import Residue
from flatprot.structure.components import Structure
from flatprot.structure.secondary import SecondaryStructureType

from .structure import StructureParser

ChainData = namedtuple("ChainData", ["residue_indices", "residue_names", "coordinates"])


class BiopythonStructureParser(StructureParser):
    def parse_structure(
        self, structure_file: Path, secondary_structure_file: Optional[Path] = None
    ) -> Structure:
        """Main entry point for structure parsing"""

        # 1. Parse structure
        bioStructure = self._parse_structure_file(structure_file)

        # 2. Process each chain
        chains = []
        for chain in bioStructure.get_chains():
            # Extract basic chain data
            chain_data = self._parse_chain_data(chain)

            # Get secondary structure (following priority)
            ss_regions = self._get_secondary_structure(
                chain, structure_file, secondary_structure_file
            )

            for region in ss_regions:
                chain.add_secondary_structure(region[0], region[1], region[2])

            # Convert to protein model
            chains.append(self._create_chain(chain_data))

        return Structure(chains)

    def _parse_structure_file(self, structure_file: Path) -> Structure:
        """Parse structure from file based on extension"""
        ext = structure_file.suffix.lower()
        if ext == ".pdb":
            parser = PDBParser(QUIET=True)
        elif ext in (".cif", ".mmcif", ".bcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return parser.get_structure("protein", structure_file)

    def _parse_chain_data(self, chain: Chain) -> ChainData:
        """Extract basic chain data: residue numbers, names, and coordinates."""
        residue_indices = []
        residue_names = []
        coordinates = []

        for i, residue in enumerate(chain):
            if "CA" in residue:
                residue_indices.append(residue.id[1])
                residue_names.append(residue.resname)
                coordinates.append(residue["CA"].get_coord())

        return ChainData(
            residue_indices,
            residue_names,
            np.array(coordinates, dtype=np.float32),
        )

    def _create_chain(self, chain_data: ChainData) -> Chain:
        """Convert chain data to Protein model"""
        residues = []
        for resname in chain_data.residue_names:
            try:
                one_letter = protein_letters_3to1[resname]
                residues.append(Residue(one_letter))
            except KeyError:
                residues.append(Residue.XAA)

        chain = Chain(
            chain_id=chain_data.chain_id,
            residues=residues,
            index=np.array(chain_data.residue_indices, dtype=np.int32),
            coordinates=chain_data.coordinates,
        )

        return chain

    def _get_secondary_structure(
        self, chain: Chain, structure_file: Path, ss_file: Optional[Path]
    ) -> list[tuple[SecondaryStructureType, int, int]]:
        """Get secondary structure following priority order"""

        # 1. Try secondary structure file if provided
        if ss_file:
            ss_list = self._get_ss_from_dssp_file(ss_file, chain)

        # 2. Try mmCIF annotation
        if structure_file.suffix.lower() in (".cif", ".mmcif", ".bcif"):
            ss_list = self._get_ss_from_mmcif(chain)

        if ss_list:
            return ss_list

        raise ValueError("No secondary structure found")

    def _get_ss_from_mmcif_file(
        self, structure: Structure, chain_id: str
    ) -> list[tuple[SecondaryStructureType, int, int]]:
        """Extract secondary structure information from mmCIF file.

        Args:
            structure: Bio.PDB Structure object
            chain_id: Chain identifier to extract SS for

        Returns:
            List of secondary structure assignments ('H' for helix, 'S' for sheet, '-' for coil)

        Raises:
            ValueError: If no secondary structure information can be parsed
        """
        try:
            # Initialize all positions as coil
            ss_regions = []

            # Parse struct_conf table
            struct_conf = structure.header.get("_struct_conf", [])
            if not struct_conf:
                raise ValueError("No secondary structure found in mmCIF file")

            for conf in struct_conf:
                # Check if this entry belongs to our chain
                if conf["beg_label_asym_id"] != chain_id:
                    continue

                # Get start and end positions
                start = int(conf["beg_label_seq_id"])
                end = int(conf["end_label_seq_id"])

                # Determine SS type
                conf_type = conf["conf_type_id"]
                if conf_type.startswith("HELX"):
                    ss_type = SecondaryStructureType.HELIX
                elif conf_type.startswith("STRN"):
                    ss_type = SecondaryStructureType.SHEET
                else:  # TURN or other types remain as coil
                    ss_type = SecondaryStructureType.COIL

                ss_regions.append((ss_type, start, end))

            return ss_regions

        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"Failed to parse secondary structure from mmCIF: {str(e)}"
            )

    def _get_ss_from_dssp_file(
        self, structure: Structure, secondary_structure_file: Path, chain_id: str
    ) -> list[tuple[SecondaryStructureType, int, int]]:
        """Extract secondary structure information from DSSP file.

        Args:
            structure: Bio.PDB Structure object
            secondary_structure_file: Path to DSSP file
            chain_id: Chain identifier to extract SS for

        Returns:
            List of tuples containing (SS_type, start_position, end_position)

        Raises:
            ValueError: If DSSP file cannot be parsed
        """
        try:
            dssp_dict = DSSP.make_dssp_dict(str(secondary_structure_file))
            ss_regions = []
            current_ss = None
            start_pos = None

            for residue in structure[0][chain_id]:
                key = (chain_id, residue.id)
                ss = dssp_dict.get(key, [None, "-"])[1]

                # Convert DSSP code to SecondaryStructureType
                if ss in ("H", "G", "I"):  # Various helix types
                    ss_type = SecondaryStructureType.HELIX
                elif ss in ("B", "E"):  # Various sheet types
                    ss_type = SecondaryStructureType.SHEET
                else:
                    ss_type = SecondaryStructureType.COIL

                # Start new region if SS type changes
                if ss_type != current_ss:
                    # Save previous region if exists
                    if current_ss is not None and start_pos is not None:
                        ss_regions.append((current_ss, start_pos, residue.id[1] - 1))
                    # Start new region
                    current_ss = ss_type
                    start_pos = residue.id[1]

            # Add final region
            if current_ss is not None and start_pos is not None:
                ss_regions.append((current_ss, start_pos, residue.id[1]))

            return ss_regions

        except Exception as e:
            raise ValueError(f"Failed to parse DSSP file: {str(e)}")

    def _structure_to_chains(self, structure: Structure) -> list[BioChain]:
        """Convert a Protein model to a Bio.PDB Chain."""
        chains = []
        for chain_id, chain in structure.items():
            bioChain = BioChain(chain_id)

            for idx, (residue_idx, residue, coord) in enumerate(
                zip(chain.index, chain.residues, chain.coordinates)
            ):
                # Create Bio.PDB Residue
                bioResidue = BioResidue(
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

                bioResidue.add(ca_atom)
                bioChain.add(bioResidue)

            chains.append(bioChain)

        return chains

    def _save_structure_with_chains(
        self, chains: list[BioChain], output_file: Path, protein_name: str = "protein"
    ) -> None:
        bioStructure = BioStructure(protein_name)
        model = Model(0)
        bioStructure.add(model)

        for chain in chains:
            model.add(chain)

        io = MMCIFIO()
        io.set_structure(bioStructure)
        io.save(str(output_file))

    def save_structure(
        self, structure: Structure, output_file: Path, separate_chains=False
    ) -> None:
        """Save structure to PDB file.

        Args:
            structure: Dictionary mapping chain IDs to Protein objects
            output_file: Path where to save the PDB file
        """
        # Create new Bio.PDB Structure
        chains = [
            self._protein_to_chain(protein, chain_id)
            for chain_id, protein in structure.items()
        ]

        if not separate_chains:
            self._save_structure_with_chains(
                chains, output_file=output_file.with_suffix(".cif")
            )
        else:
            for chain in chains:
                self._save_structure_with_chains(
                    [chain],
                    output_file=output_file.with_suffix(f".{chain.id}.cif"),
                    protein_name=chain.id,
                )
