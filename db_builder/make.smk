from pathlib import Path
import os
import json
import random

import polars as pl



# Default values if not in config
OUTPUT_DIR = "data/alignment_db"
WORK_DIR = "out/alignment_pipeline"

RETRY_COUNT = 3
TEST_MODE = True
NUM_FAMILIES = 5
RANDOM_SEED = 42
CONCURRENT_DOWNLOADS = 5
FOLDSEEK_PATH = "foldseek"

## Step outputs
SCOP_FILE = f"{WORK_DIR}/scop-cla-latest.txt"
SCOP_INFO = f"{OUTPUT_DIR}/superfamilies.tsv"
PDB_FILES = f"{WORK_DIR}/pdbs"
DOMAIN_FILES = f"{WORK_DIR}/domains"
REPRESENTATIVE_DOMAINS = f"{WORK_DIR}/representative_domains"
TMP_FOLDSEEK_DBS = f"{WORK_DIR}/foldseek"
MATRICES = f"{WORK_DIR}/matrices"
ALIGNMENT_DB = f"{OUTPUT_DIR}/alignment_database.h5"
FOLDSEEK_DB = f"{OUTPUT_DIR}/foldseek"
DATABASE_INFO = f"{OUTPUT_DIR}/database_info.json"
REPORT_DIR = f"{OUTPUT_DIR}/reports"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(PDB_FILES, exist_ok=True)
os.makedirs(DOMAIN_FILES, exist_ok=True)
os.makedirs(REPRESENTATIVE_DOMAINS, exist_ok=True)
os.makedirs(TMP_FOLDSEEK_DBS, exist_ok=True)
os.makedirs(MATRICES, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
# Final output files - defines the complete workflow end result
rule all:
    input:
        FOLDSEEK_DB,
        ALIGNMENT_DB,
        DATABASE_INFO


# -------------------------------------------------------------------------------------- #
#                                Process SCOP information                                #
# -------------------------------------------------------------------------------------- #

rule download_scop:
    output:
        scop_file = f"{SCOP_FILE}"
    shell:
        """
        wget -O {output.scop_file} https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt
        """

checkpoint parse_scop:
    input:
        scop_file = f"{SCOP_FILE}"
    output:
        superfamilies = f"{SCOP_INFO}",
        scop_report = report(f"{REPORT_DIR}/scop.rst", category="SCOP Analysis")
    script:
        "scop_parse.py"

# -------------------------------------------------------------------------------------- #
#                           Download PDBs and extract domains                            #
# -------------------------------------------------------------------------------------- #

TIMEOUT = 20  # Default timeout in seconds

rule download_pdb:
    output:
        pdb_file = f"{PDB_FILES}/{{pdb_id}}.pdb",
        # Add a status file to indicate success/failure
        status = f"{PDB_FILES}/{{pdb_id}}.status"
    params:
        pdb_id = "{pdb_id}",
        pdb_dir = f"{PDB_FILES}",
        retry_count = RETRY_COUNT,
        timeout = TIMEOUT
    resources:
        network_connections = 1,
    script:
        "pdb_download.py"

rule extract_domain:
    input:
        pdb_file = f"{PDB_FILES}/{{pdb_id}}.pdb"
    output:
        domain_file = f"{DOMAIN_FILES}/{{sf_id}}/{{pdb_id}}_{{chain}}_{{start}}_{{end}}.cif"
    params:
        sf_id = "{sf_id}",
        chain = "{chain}",
        start = "{start}",
        end = "{end}"
    script:
        "pdb_extract_domain.py"

# Add a rule to generate domain extraction requests from SCOP data
rule generate_domain_extraction_requests:
    input:
        superfamilies = SCOP_INFO
    output:
        sf_domains = f"{WORK_DIR}/sf_domains/{{sf_id}}.tsv"
    params:
        sf_id = "{sf_id}"
    run:
        df = pl.read_csv(input.superfamilies, separator="\t")
        sf_df = df.filter(pl.col("sf_id") == params.sf_id)
        sf_df.write_csv(output.sf_domains, separator="\t")



# -------------------------------------------------------------------------------------- #
#                          Foldseek Alignment for all families                           #
# -------------------------------------------------------------------------------------- #


rule extract_all_domains_for_superfamily:
    input:
        sf_domains = f"{WORK_DIR}/sf_domains/{{sf_id}}.tsv",
        # Use dynamic input based on checkpoint data
        domain_files = lambda wildcards: get_domains_for_superfamily(wildcards)
    output:
        flag = f"{WORK_DIR}/domains_extracted/{{sf_id}}.flag"
    shell:
        "touch {output.flag}"

# Update create_domain_db to depend on domain extraction completion
rule create_domain_db:
    input:
        extraction_complete = f"{WORK_DIR}/domains_extracted/{{sf_id}}.flag"
    output:
        domain_db = directory(f"{TMP_FOLDSEEK_DBS}/{{sf_id}}/db")
    params:
        sf_id = "{sf_id}"
        domain_dir = f"{DOMAIN_FILES}/{{sf_id}}/"
    shell:
        "foldseek createdb {params.domain_dir} {output.domain_db}"

# Update get_domain_alignment to track alignment progress
# cf. https://github.com/steineggerlab/foldseek/issues/33#issuecomment-1495652159 for all v. all

rule get_domain_alignment:
    input:
        domain_db = f"{TMP_FOLDSEEK_DBS}/{{sf_id}}/db/",
    output:
        alignment = f"{MATRICES}/{{sf_id}}.m8",
    params:
        sf_id = "{sf_id}",
    resources:
        cpus = 1
    shell:
        """
        mkdir -p {MATRICES}/{params.sf_id}
        foldseek easy-search \
            {input.domain_db} \
            {input.domain_db} \
            {output.alignment} \
            /tmp/{params.sf_id} \
            --format-output query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore,a \
            -e inf \
            --exhaustive-search 1 \
            --tm-score-threshold 0.0 \
            --alignment-type 2
        rm -rf /tmp/{params.sf_id}
        """

# Connect representative selection to domain alignments
rule get_representative_domain:
    input:
        alignment_file = f"{MATRICES}/{{sf_id}}.m8",
    output:
        representative_domain = f"{REPRESENTATIVE_DOMAINS}/{{sf_id}}.pdb"
    params:
        sf_id = "{sf_id}",
        domain_dir = f"{DOMAIN_FILES}/{{sf_id}}"
    script:
        "representative_domain.py"


# -------------------------------------------------------------------------------------- #
#                            Aggregate final output database                             #
# -------------------------------------------------------------------------------------- #



rule aggregate_representatives:
    input:
        superfamilies = SCOP_INFO,
        representative_domains = expand(
            f"{REPRESENTATIVE_DOMAINS}/{{sf_id}}.pdb",
            sf_id=get_all_superfamily_ids()
        )
    shell:
        "touch {REPRESENTATIVE_DOMAINS}/.completed"


rule create_alignment_database:
    input:
        f"{REPRESENTATIVE_DOMAINS}/.completed",
    output:
        database = f"{ALIGNMENT_DB}",
        database_info = f"{DATABASE_INFO}"
    script:
        "create_alignment_db.py"


rule create_alignment_foldseek_db:
    input:
        f"{REPRESENTATIVE_DOMAINS}/.completed",
    output:
        directory(FOLDSEEK_DB)
    shell:
        """
        {FOLDSEEK_PATH} createdb {REPRESENTATIVE_DOMAINS} {output[0]}
        """



# -------------------------------------------------------------------------------------- #
#                               Helper Functions                                         #
# -------------------------------------------------------------------------------------- #

def get_all_superfamily_ids():
    """Get list of superfamilies to process based on TEST_MODE setting."""
    try:
        df = pl.read_csv(SCOP_INFO, separator="\t")
        if TEST_MODE:
            return df["sf_id"].unique().shuffle(seed=RANDOM_SEED).to_list()[:NUM_FAMILIES]
        return df["sf_id"].unique().to_list()
    except FileNotFoundError:
        # Return empty list if file doesn't exist yet (during DAG construction)
        return []


def get_domains_for_superfamily(wildcards):
    """Get domains for a superfamily based on checkpoint output and successful downloads."""
    checkpoint_output = checkpoints.parse_scop.get().output[0]
    df = pl.read_csv(checkpoint_output, separator="\t")

    # Filter for this superfamily
    sf_df = df.filter(pl.col("sf_id") == wildcards.sf_id)

    # Return list of domain files, but only for successfully downloaded PDBs
    domain_files = []
    for row in sf_df.iter_rows(named=True):
        pdb_id = row["pdb_id"]
        chain = row["chain"]
        start = row["start"]
        end = row["end"]

        # Check if the PDB was successfully downloaded
        status_file = f"{PDB_FILES}/{pdb_id}.status"
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = f.read().strip()

            # Only include domain if download was successful
            if status == "success":
                domain_file = f"{DOMAIN_FILES}/{wildcards.sf_id}/{pdb_id}_{chain}_{start}_{end}.pdb"
                domain_files.append(domain_file)

    return domain_files
