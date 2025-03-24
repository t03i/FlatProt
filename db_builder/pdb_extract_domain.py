# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""
Extract domain from PDB structure based on chain and residue range.

This script extracts a specific domain from a PDB structure file based on
the provided chain ID and residue range (start-end). It uses the gemmi
library for efficient structure manipulation.

The extracted domain is saved as a new PDB file with naming convention:
{pdb_id}_{chain}_{start}_{end}.pdb
"""

import logging
import sys
from pathlib import Path

import gemmi
from snakemake.script import snakemake

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def extract_domain(
    pdb_file: Path, chain: str, start_res: int, end_res: int, output_file: Path
) -> bool:
    """
    Extract a domain from a PDB file using gemmi.

    Args:
        pdb_file: Path to the input PDB file
        chain: Chain identifier
        start_res: Start residue number
        end_res: End residue number
        output_file: Path to save the extracted domain

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Parse PDB file with gemmi
        structure = gemmi.read_structure(str(pdb_file))

        # Create a new structure for the domain
        domain = gemmi.Structure()
        domain.name = f"{pdb_file.stem}_{chain}_{start_res}_{end_res}"

        # Create a new model
        model = gemmi.Model("1")

        # Find and copy the specified chain
        found_chain = False
        chain_len = 0
        extracted_residues = 0

        for original_chain in structure[0]:
            if original_chain.name == chain:
                found_chain = True
                chain_len = len(original_chain)

                new_chain = gemmi.Chain(chain)

                # Copy residues in the specified range
                for residue in original_chain:
                    seq_id = residue.seqid.num
                    if start_res <= seq_id <= end_res:
                        new_chain.add_residue(residue)
                        extracted_residues += 1

                if extracted_residues > 0:
                    model.add_chain(new_chain)

        # Check if extraction was successful
        if not found_chain:
            logger.warning(f"Chain {chain} not found in {pdb_file}")
            return False

        if extracted_residues == 0:
            logger.warning(
                f"No residues in range {start_res}-{end_res} found in chain {chain} of {pdb_file}"
            )
            return False

        # Only consider it a success if we extracted more than 10% of the chain
        if chain_len > 0 and extracted_residues < 0.1 * chain_len:
            logger.warning(
                f"Extracted only {extracted_residues}/{chain_len} residues from {pdb_file}, "
                f"which is less than 10% of the chain"
            )

        # Add model to structure and save
        domain.add_model(model)
        domain.write_pdb(str(output_file))

        logger.info(
            f"Extracted domain from {pdb_file}: chain {chain}, residues {start_res}-{end_res} "
            f"({extracted_residues} residues)"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to extract domain from {pdb_file}: {e}")
        return False


# Snakemake integration
if __name__ == "__main__":
    # Get parameters from Snakemake
    pdb_file = Path(snakemake.input.pdb_file)
    output_file = Path(snakemake.output.domain_file)
    chain = snakemake.params.chain
    start_res = int(snakemake.params.start)
    end_res = int(snakemake.params.end)

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract the domain
    extract_domain(pdb_file, chain, start_res, end_res, output_file)
