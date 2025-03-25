from pathlib import Path
import os
import json
import random
import tempfile

import polars as pl



# Default values if not in config
OUTPUT_DIR = "out/alignment_db"
WORK_DIR = "tmp/alignment_pipeline"

RETRY_COUNT = 3
TEST_MODE = True
NUM_FAMILIES = 40
RANDOM_SEED = 42
CONCURRENT_DOWNLOADS = 5
FOLDSEEK_PATH = "foldseek"

## Step outputs
SCOP_FILE = f"{WORK_DIR}/scop-cla-latest.txt"
SUPERFAMILIES = f"{WORK_DIR}/superfamilies.tsv"
PDB_FILES = f"{WORK_DIR}/pdbs"
DOMAIN_FILES = f"{WORK_DIR}/domains"
DOMAIN_FLAG = f"{DOMAIN_FILES}/{{sf_id}}/extracted.flag"
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

# Add log directory
LOG_DIR = f"{WORK_DIR}/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(Path(FOLDSEEK_DB).parent, exist_ok=True)
os.makedirs(PDB_FILES, exist_ok=True)
os.makedirs(DOMAIN_FILES, exist_ok=True)
os.makedirs(REPRESENTATIVE_DOMAINS, exist_ok=True)
os.makedirs(MATRICES, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

wildcard_constraints:
    sf_id=r"\d+",
    pdb_id="[^/]+",
    chain="[^/]+",
    start=r"\d+",
    end=r"\d+"

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
    """Get domains for a superfamily based on successful downloads."""
    # We only need to wait for the checkpoint to complete, no need to use its output
    checkpoints.download_all_pdbs.get()

    df = pl.read_csv(SUPERFAMILIES, separator="\t")

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

def generate_domain_tsv(sf_id, input_file, output_file):
    """Generate a TSV file for a superfamily."""
    df = pl.read_csv(input_file, separator="\t")
    sf_df = df.filter(pl.col("sf_id") == int(sf_id))
    sf_df.write_csv(output_file, separator="\t")

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
    log:
        f"{LOG_DIR}/download_scop.log"
    shell:
        """
        wget -O {output.scop_file} https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt 2>&1 | tee {log}
        """

rule parse_scop:
    input:
        scop_file = f"{SCOP_FILE}"
    output:
        superfamilies = f"{SUPERFAMILIES}",
        report = report(f"{REPORT_DIR}/scop.rst", category="SCOP Analysis")
    log:
        f"{LOG_DIR}/parse_scop.log"
    script:
        "scop_parse.py"

# -------------------------------------------------------------------------------------- #
#                           Download PDBs and extract domains                            #
# -------------------------------------------------------------------------------------- #

TIMEOUT = 20  # Default timeout in seconds

rule download_pdb:
    output:
        struct_file = f"{PDB_FILES}/{{pdb_id}}.struct",
        status = f"{PDB_FILES}/{{pdb_id}}.status"
    params:
        pdb_id = "{pdb_id}",
        retry_count = RETRY_COUNT,
        timeout = TIMEOUT
    resources:
        network_connections = 1
    log:
        f"{LOG_DIR}/download_pdb_{{pdb_id}}.log"
    script:
        "pdb_download.py"

checkpoint download_all_pdbs:
    input:
        superfamilies = SUPERFAMILIES,
        structure_ids = [f"{PDB_FILES}/{pdb_id}.struct" for pdb_id in get_all_structure_ids()]
    output:
        flag = temp(f"{PDB_FLAG}"),
        report = report(f"{REPORT_DIR}/pdb_download_report.rst", category="PDB Download Report")
    params:
        pdb_dir = f"{PDB_FILES}"
    log:
        f"{LOG_DIR}/download_all_pdbs.log"
    script:
        "pdb_report.py"


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
    log:
        f"{LOG_DIR}/extract_domain_{{sf_id}}_{{pdb_id}}_{{chain}}_{{start}}_{{end}}.log"
    script:
        "domain_extract.py"

# Add a rule to generate domain extraction requests from SCOP data
rule generate_domain_extraction_requests:
    input:
        superfamilies = SUPERFAMILIES,
        flag = f"{PDB_FLAG}"
    output:
        sf_domains = f"{DOMAIN_FILES}/{{sf_id}}/domains.tsv"
    params:
        sf_id = "{sf_id}"
    log:
        f"{LOG_DIR}/generate_domain_extraction_requests_{{sf_id}}.log"
    run:
        with open(log[0], "w") as f:
            f.write(f"Generating domain extraction requests for superfamily {params.sf_id}\n")
            generate_domain_tsv(params.sf_id, input.superfamilies, output.sf_domains)
            f.write("Domain extraction requests generated successfully\n")

rule extract_all_domains_for_superfamily:
    input:
        sf_domains = f"{DOMAIN_FILES}/{{sf_id}}/domains.tsv",
        domain_files = lambda wildcards: get_domains_for_superfamily(wildcards)
    output:
        flag = temp(f"{DOMAIN_FLAG}"),
    log:
        f"{LOG_DIR}/extract_all_domains_{{sf_id}}.log"
    shell:
        """
        echo "Completed domain extraction for superfamily {wildcards.sf_id}" > {log}
        touch {output.flag}
        """


rule domain_extraction_report:
    input:
        superfamilies = SUPERFAMILIES,
        flags = expand(f"{DOMAIN_FLAG}", sf_id=get_all_superfamily_ids()),
        pdb_dir = PDB_FILES,
        domain_dir = DOMAIN_FILES
    output:
        report = report(f"{REPORT_DIR}/domain_extraction_report.rst", category="Domain Extraction Report")
    log:
        f"{LOG_DIR}/domain_extraction_report.log"
    script:
        "domain_report.py"

# -------------------------------------------------------------------------------------- #
#                          Foldseek Alignment for all families                           #
# -------------------------------------------------------------------------------------- #

# Update create_domain_db to depend on domain extraction completion
rule create_domain_db:
    input:
        extraction_complete = f"{DOMAIN_FLAG}",
    output:
        domain_db = temp(f"{TMP_FOLDSEEK_DB}")
    params:
        sf_id = "{sf_id}",
        domain_dir = str(Path(f"{DOMAIN_FLAG}").parent)
    resources:
        cpus = 4
    log:
        f"{LOG_DIR}/create_domain_db_{{sf_id}}.log"
    shell:
        """
        mkdir -p $(dirname {output.domain_db}) && \
        foldseek createdb {params.domain_dir} {output.domain_db} --threads 4 -v 3  2>&1 > {log}
        """

# Update get_domain_alignment to track alignment progress
# cf. https://github.com/steineggerlab/foldseek/issues/33#issuecomment-1495652159 for all v. all
rule get_domain_alignment:
    input:
        domain_db = f"{TMP_FOLDSEEK_DB}"
    output:
        alignment = f"{MATRICES}/{{sf_id}}.m8",
        tmp_dir = temp(directory(f"{WORK_DIR}/{{sf_id}}"))
    params:
        sf_id = "{sf_id}"
    resources:
        cpus = 4
    log:
        f"{LOG_DIR}/get_domain_alignment_{{sf_id}}.log"
    shell:
        """
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
            --threads 4 \
            -v 3 2>&1 > {log}
        """

# Connect representative selection to domain alignments
rule get_representative_domain:
    input:
        alignment_file = f"{MATRICES}/{{sf_id}}.m8",
        domain_flag = f"{DOMAIN_FLAG}"
    output:
        representative_domain = f"{REPRESENTATIVE_DOMAINS}/{{sf_id}}.cif"
    params:
        domain_dir = str(Path(f"{DOMAIN_FLAG}").parent),
        sf_id = "{sf_id}"
    log:
        f"{LOG_DIR}/get_representative_domain_{{sf_id}}.log"
    script:
        "domain_select.py"


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
        flag = temp(f"{REPRESENTATIVE_FLAG}")
    log:
        f"{LOG_DIR}/aggregate_representatives.log"
    shell:
        """
        echo "Aggregating representatives completed" > {log}
        touch {output.flag}
        """

rule create_alignment_database:
    input:
        flag = f"{REPRESENTATIVE_FLAG}",
        representative_domains = [f for f in Path(REPRESENTATIVE_DOMAINS).glob("*.cif")],
        scop_file = SCOP_FILE
    output:
        database = f"{ALIGNMENT_DB}",
        database_info = f"{DATABASE_INFO}"
    log:
        f"{LOG_DIR}/create_alignment_database.log"
    params:
        test_mode = TEST_MODE,
        num_families = NUM_FAMILIES
    script:
        "db_create.py"


rule create_alignment_foldseek_db:
    input:
        flag = f"{REPRESENTATIVE_FLAG}",
        representative_domains_folder = REPRESENTATIVE_DOMAINS,
        scop_file = SCOP_FILE
    output:
        FOLDSEEK_DB
    resources:
        cpus = 4
    params:
        test_mode = TEST_MODE,
        num_families = NUM_FAMILIES
    log:
        f"{LOG_DIR}/create_alignment_foldseek_db.log"
    shell:
        """
        {FOLDSEEK_PATH} createdb {input.representative_domains_folder} {output} --threads 4 -v 3 2>&1 > {log}
        """
