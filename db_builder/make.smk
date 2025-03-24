from pathlib import Path
import os
import json
import random
import tempfile

import polars as pl



# Default values if not in config
OUTPUT_DIR = "data/alignment_db"
WORK_DIR = "tmp/alignment_pipeline"

RETRY_COUNT = 3
TEST_MODE = True
NUM_FAMILIES = 5
RANDOM_SEED = 42
CONCURRENT_DOWNLOADS = 5
FOLDSEEK_PATH = "foldseek"

## Step outputs
SCOP_FILE = f"{WORK_DIR}/scop-cla-latest.txt"
SUPERFAMILIES = f"{WORK_DIR}/superfamilies.tsv"
PDB_FILES = f"{WORK_DIR}/pdbs"
DOMAIN_FILES = f"{WORK_DIR}/domains"
REPRESENTATIVE_DOMAINS = f"{WORK_DIR}/representative_domains"
MATRICES = f"{WORK_DIR}/matrices"
ALIGNMENT_DB = f"{OUTPUT_DIR}/alignments.h5"
FOLDSEEK_DB = f"{OUTPUT_DIR}/foldseek/db"
DATABASE_INFO = f"{OUTPUT_DIR}/database_info.json"
REPORT_DIR = f"{WORK_DIR}/reports"
TMP_DIR = Path(tempfile.mkdtemp())
TMP_FOLDSEEK_DB = f"{TMP_DIR}/foldseek/{{sf_id}}/db"
PDB_FLAG = f"{TMP_DIR}/download_all_pdbs.flag"
REPRESENTATIVE_FLAG = f"{TMP_DIR}/representatives.flag"
DOMAIN_FLAG = f"{TMP_DIR}/{{sf_id}}-extracted.flag"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(Path(FOLDSEEK_DB).parent, exist_ok=True)
os.makedirs(PDB_FILES, exist_ok=True)
os.makedirs(DOMAIN_FILES, exist_ok=True)
os.makedirs(REPRESENTATIVE_DOMAINS, exist_ok=True)
os.makedirs(MATRICES, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------------------------------------------------------------------- #
#                               Helper Functions                                         #
# -------------------------------------------------------------------------------------- #

def get_all_superfamily_ids():
    """Get list of superfamilies to process based on TEST_MODE setting."""
    try:
        df = pl.read_csv(SUPERFAMILIES, separator="\t")
        if TEST_MODE:
            return df["sf_id"].unique().shuffle(seed=RANDOM_SEED).to_list()[:NUM_FAMILIES]
        return df["sf_id"].unique().to_list()
    except FileNotFoundError:
        # Return empty list if file doesn't exist yet (during DAG construction)
        return []

def get_all_structure_ids():
    try:
        ids = get_all_superfamily_ids()
        df = pl.read_csv(SUPERFAMILIES, separator="\t")
        return df.filter(pl.col("sf_id").is_in(ids))["pdb_id"].unique().to_list()
    except FileNotFoundError:
        # Return empty list if file doesn't exist yet (during DAG construction)
        return []

def get_domains_for_superfamily(wildcards):
    """Get domains for a superfamily based on checkpoint output and successful downloads."""
    checkpoint_output = checkpoints.parse_scop.get().output[0]
    df = pl.read_csv(checkpoint_output, separator="\t")

    # Filter for this superfamily
    sf_df = df.filter(pl.col("sf_id") == int(wildcards.sf_id))

    # Return list of domain files, but only for successfully downloaded PDBs
    domain_files = []
    for row in sf_df.iter_rows(named=True):
        pdb_id = row["pdb_id"]
        chain = row["chain"]
        start = row["start_res"]
        end = row["end_res"]

        # Check if the PDB was successfully downloaded
        status_file = f"{PDB_FILES}/{pdb_id}.status"
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_data = json.load(f)

            # Only include domain if download was successful
            if status_data.get("success", False):
                domain_file = f"{DOMAIN_FILES}/{wildcards.sf_id}/{pdb_id}_{chain}_{start}_{end}.cif"
                domain_files.append(domain_file)


    return domain_files

# -------------------------------------------------------------------------------------- #
#                                      Target rule                                       #
# -------------------------------------------------------------------------------------- #

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
        superfamilies = f"{SUPERFAMILIES}",
        report = report(f"{REPORT_DIR}/scop.rst", category="SCOP Analysis")
    script:
        "scop_parse.py"

# -------------------------------------------------------------------------------------- #
#                           Download PDBs and extract domains                            #
# -------------------------------------------------------------------------------------- #

TIMEOUT = 20  # Default timeout in seconds

rule download_pdb:
    output:
        struct_file = f"{PDB_FILES}/{{pdb_id}}.struct",  # Generic extension to handle both formats
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

rule download_all_pdbs:
    input:
        superfamilies = SUPERFAMILIES,
        structure_ids = [f"{PDB_FILES}/{pdb_id}.struct" for pdb_id in get_all_structure_ids()]
    output:
        flag = temp(f"{PDB_FLAG}"),  # Temporary flag only valid for this run
        report = report(f"{REPORT_DIR}/pdb_download_report.rst", category="PDB Download Report")
    params:
        pdb_dir = f"{PDB_FILES}"
    script:
        "pdb_download_report.py"


rule extract_domain:
    input:
        struct_file = f"{PDB_FILES}/{{pdb_id}}.struct"
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
        superfamilies = SUPERFAMILIES,
        flag = f"{PDB_FLAG}"
    output:
        sf_domains = f"{DOMAIN_FILES}/{{sf_id}}/domains.tsv"
    params:
        sf_id = "{sf_id}"
    run:
        df = pl.read_csv(input.superfamilies, separator="\t")
        sf_df = df.filter(pl.col("sf_id") == int(params.sf_id))
        sf_df.write_csv(output.sf_domains, separator="\t")



# -------------------------------------------------------------------------------------- #
#                          Foldseek Alignment for all families                           #
# -------------------------------------------------------------------------------------- #

# Fix wildcard constraints by using raw strings or proper escaping
wildcard_constraints:
    sf_id=r"\d+",  # Using raw string for digit pattern
    pdb_id="[^/]+",
    chain="[^/]+",
    start=r"\d+",
    end=r"\d+"

rule extract_all_domains_for_superfamily:
    input:
        sf_domains = f"{DOMAIN_FILES}/{{sf_id}}/domains.tsv",
        domain_files = lambda wildcards: get_domains_for_superfamily(wildcards)
    output:
        flag = f"{DOMAIN_FLAG}"
    shell:
        "touch {output.flag}"

# Update create_domain_db to depend on domain extraction completion
rule create_domain_db:
    input:
        extraction_complete = f"{DOMAIN_FLAG}"
    output:
        domain_db = f"{TMP_FOLDSEEK_DB}"
    params:
        sf_id = "{sf_id}",
        domain_dir = f"{DOMAIN_FILES}/{{sf_id}}/"
    resources:
        cpus = 1
    shell:
        """
        mkdir -p $(dirname {output.domain_db}) && \
        foldseek createdb {params.domain_dir} {output.domain_db} --threads 1 -v 1
        """

# Update get_domain_alignment to track alignment progress
# cf. https://github.com/steineggerlab/foldseek/issues/33#issuecomment-1495652159 for all v. all
rule get_domain_alignment:
    input:
        domain_db = f"{TMP_FOLDSEEK_DB}",
    output:
        alignment = f"{MATRICES}/{{sf_id}}.m8",
        tmp_dir = temp(directory(f"{WORK_DIR}/{{sf_id}}"))
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
            {output.tmp_dir} \
            --format-output query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore \
            -e inf \
            --exhaustive-search 1 \
            --tmscore-threshold 0.0 \
            --alignment-type 2 \
            --threads 1 \
            -v 1
        """

# Connect representative selection to domain alignments
rule get_representative_domain:
    input:
        alignment_file = f"{MATRICES}/{{sf_id}}.m8",
    output:
        representative_domain = f"{REPRESENTATIVE_DOMAINS}/{{sf_id}}.cif"
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
        superfamilies = SUPERFAMILIES,
        representative_domains = expand(
            f"{REPRESENTATIVE_DOMAINS}/{{sf_id}}.cif",
            sf_id=get_all_superfamily_ids()
        )
    output:
        flag = f"{REPRESENTATIVE_FLAG}"
    shell:
        "touch {output.flag}"


rule create_alignment_database:
    input:
        flag = f"{REPRESENTATIVE_FLAG}"
    params:
        representative_domains = [f for f in Path(REPRESENTATIVE_DOMAINS).glob("*.cif")],
    output:
        database = f"{ALIGNMENT_DB}",
        database_info = f"{DATABASE_INFO}"
    script:
        "create_alignment_db.py"


rule create_alignment_foldseek_db:
    input:
        flag = f"{REPRESENTATIVE_FLAG}"
    output:
        FOLDSEEK_DB
    shell:
        """
        {FOLDSEEK_PATH} createdb {REPRESENTATIVE_DOMAINS} {output} --threads 1 -v 1
        """
