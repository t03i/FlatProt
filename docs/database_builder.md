# FlatProt Database Builder

The database builder is a collection of tools for creating and managing FlatProt's structural alignment databases. It processes SCOP classifications and representative domain structures to create optimized databases for structural alignment.

## Overview

The database building process consists of several automated steps:

1. SCOP classification downloading and parsing
2. PDB structure downloading with retry mechanism
3. Domain extraction and validation
4. All-vs-all domain alignment using Foldseek
5. Representative domain selection
6. Final database creation (HDF5 and Foldseek formats)

## Usage

The entire database building process is managed through a Snakemake workflow. To build a new database:

```bash
snakemake -s db_builder/snakefile --cores all
```

### Configuration

The build process can be customized through the following variables in the snakefile:

-   `OUTPUT_DIR`: Output path for the database (default: "out/alignment_db")
-   `WORK_DIR`: Working directory for intermediate files (default: "tmp/alignment_pipeline")
-   `TEST_MODE`: Enable test mode with limited superfamilies
-   `NUM_FAMILIES`: Number of families to process in test mode (default: 401)
-   `RETRY_COUNT`: Number of download retries (default: 3)
-   `CONCURRENT_DOWNLOADS`: Maximum parallel downloads (default: 5)
-   `FOLDSEEK_PATH`: Path to Foldseek executable (default: "foldseek")

### Database Structure

The pipeline produces two main outputs:

1. An HDF5 alignment database (`alignments.h5`) containing:

    - Domain structural information
    - Superfamily classifications
    - Alignment matrices

2. A Foldseek database for rapid structural searching

3. A JSON info file (`database_info.json`) with:
    - Database statistics
    - Creation timestamp
    - Configuration parameters

### Pipeline Steps

The workflow consists of the following main stages:

1. **SCOP Processing**

    - Downloads latest SCOP classification
    - Parses and validates superfamily information
    - Generates initial reports

2. **Structure Processing**

    - Downloads PDB structures with retry mechanism
    - Validates downloads
    - Extracts individual domains
    - Generates domain extraction reports

3. **Domain Analysis**

    - Creates Foldseek databases per superfamily
    - Performs all-vs-all structural alignments
    - Selects representative domains

4. **Database Creation**
    - Aggregates representative domains
    - Creates final HDF5 database
    - Builds Foldseek search database

### Running in Docker

```bash
docker run -it --rm -v $(pwd):/app ghcr.io/astral-sh/uv:python3.13-bookworm-slim
```

````bash
apt update && apt install -y git wget
git clone https://github.com/t03i/flatprot.git
cd flatprot

wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz; tar xvzf foldseek-linux-avx2.tar.gz
git checkout staging
uv sync --all-groups
uv run snakemake -s db_builder/snakefile --cores all --quiet --config foldseek_path=foldseek/foldseek
```
````
