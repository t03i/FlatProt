#!/bin/bash

# Benchmarking script for protein visualization tools
# Usage: ./benchmark_visualization_tools.sh <structure_files> <N_iterations> [output_file]

set -euo pipefail

# Default values
N_ITERATIONS=5
OUTPUT_FILE="benchmark_results.csv"
TEMP_DIR="./tmp/benchmark_$$"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <structure_files> <N_iterations> [output_file]"
    echo "Example: $0 'data/*/*.cif' 10 results.csv"
    exit 1
fi

STRUCTURE_PATTERN="$1"
N_ITERATIONS="$2"
if [ $# -ge 3 ]; then
    OUTPUT_FILE="$3"
fi

# Create temp directory
mkdir -p "$TEMP_DIR"

# Initialize CSV with headers
echo "tool,method,structure_file,iteration,execution_time_seconds,memory_mb,exit_code,error_message,family_available" > "$OUTPUT_FILE"

# Utility functions
log_result() {
    local tool="$1"
    local method="$2"
    local structure="$3"
    local iteration="$4"
    local exec_time="$5"
    local memory="$6"
    local exit_code="$7"
    local error_msg="$8"
    local family_available="$9"

    echo "$tool,$method,$structure,$iteration,$exec_time,$memory,$exit_code,\"$error_msg\",$family_available" >> "$OUTPUT_FILE"
}

measure_execution() {
    local cmd="$1"
    local start_time end_time exec_time memory exit_code error_msg

    start_time=$(date +%s.%N)
    memory_before=$(ps -o rss= -p $$ 2>/dev/null || echo "0")

    # Execute command and capture exit code
    if output=$(eval "$cmd" 2>&1); then
        exit_code=0
        error_msg=""
    else
        exit_code=$?
        error_msg="$output"
    fi

    end_time=$(date +%s.%N)
    memory_after=$(ps -o rss= -p $$ 2>/dev/null || echo "0")

    exec_time=$(echo "$end_time - $start_time" | bc -l)
    memory=$(echo "($memory_after - $memory_before) / 1024" | bc -l)

    echo "$exec_time $memory $exit_code $error_msg"
}

# Tool-specific functions

# FlatProt Single Structure
flatprot_single() {
    local structure="$1"
    local output_file="$TEMP_DIR/flatprot_single_$(basename "$structure").svg"

    if [[ "$structure" == *.pdb ]]; then
        # PDB requires DSSP file
        local dssp_file="$TEMP_DIR/$(basename "$structure" .pdb).dssp"
        if command -v mkdssp >/dev/null 2>&1; then
            mkdssp -i "$structure" -o "$dssp_file" >/dev/null 2>&1 || true
            flatprot project "$structure" --dssp "$dssp_file" -o "$output_file"
        else
            echo "mkdssp not available for PDB processing" >&2
            return 1
        fi
    else
        flatprot project "$structure" -o "$output_file"
    fi
}

# FlatProt Family/Overlay
flatprot_family() {
    local structures="$1"
    local output_file="$TEMP_DIR/flatprot_overlay_$(date +%s).png"

    flatprot overlay $structures -o "$output_file"
}

# ProLego Single (placeholder - adapt to actual ProLego API)
prolego_single() {
    local structure="$1"
    local output_file="$TEMP_DIR/prolego_single_$(basename "$structure").svg"

    # Placeholder command - replace with actual ProLego command
    if command -v prolego >/dev/null 2>&1; then
        prolego visualize "$structure" --output "$output_file"
    else
        echo "ProLego not available" >&2
        return 1
    fi
}

# ProLego Family (placeholder)
prolego_family() {
    local structures="$1"
    local output_file="$TEMP_DIR/prolego_family_$(date +%s).svg"

    # Check if ProLego supports family/overlay functionality
    if command -v prolego >/dev/null 2>&1; then
        if prolego --help | grep -q "overlay\|family\|multi"; then
            prolego overlay $structures --output "$output_file"
        else
            echo "ProLego family functionality not available" >&2
            return 2  # Special exit code for unsupported feature
        fi
    else
        echo "ProLego not available" >&2
        return 1
    fi
}

# SSDraw Single (placeholder)
ssdraw_single() {
    local structure="$1"
    local output_file="$TEMP_DIR/ssdraw_single_$(basename "$structure").svg"

    if command -v ssdraw >/dev/null 2>&1; then
        ssdraw "$structure" -o "$output_file"
    else
        echo "SSDraw not available" >&2
        return 1
    fi
}

# SSDraw Family (placeholder)
ssdraw_family() {
    local structures="$1"
    local output_file="$TEMP_DIR/ssdraw_family_$(date +%s).svg"

    # SSDraw likely doesn't support family functionality
    echo "SSDraw family functionality not available" >&2
    return 2
}

# PyMOL Single
pymol_single() {
    local structure="$1"
    local output_file="$TEMP_DIR/pymol_single_$(basename "$structure").png"
    local script_file="$TEMP_DIR/pymol_script_$(date +%s).pml"

    if command -v pymol >/dev/null 2>&1; then
        cat > "$script_file" << EOF
load $structure
as cartoon
color spectrum
bg_color white
png $output_file, dpi=300
quit
EOF
        pymol -c "$script_file"
    else
        echo "PyMOL not available" >&2
        return 1
    fi
}

# PyMOL Family (multi-structure alignment)
pymol_family() {
    local structures="$1"
    local output_file="$TEMP_DIR/pymol_family_$(date +%s).png"
    local script_file="$TEMP_DIR/pymol_script_$(date +%s).pml"

    if command -v pymol >/dev/null 2>&1; then
        cat > "$script_file" << EOF
# Load all structures
EOF
        local first_structure=""
        for struct in $structures; do
            echo "load $struct" >> "$script_file"
            if [ -z "$first_structure" ]; then
                first_structure=$(basename "$struct" | cut -d. -f1)
            fi
        done

        cat >> "$script_file" << EOF
# Align all to first structure
align_all $first_structure
as cartoon
color spectrum
bg_color white
png $output_file, dpi=300
quit
EOF
        pymol -c "$script_file"
    else
        echo "PyMOL not available" >&2
        return 1
    fi
}

# Main benchmarking loop
echo "Starting benchmark with $N_ITERATIONS iterations..."
echo "Structure pattern: $STRUCTURE_PATTERN"
echo "Output file: $OUTPUT_FILE"

# Get list of structure files
STRUCTURE_FILES=($(eval "ls $STRUCTURE_PATTERN" 2>/dev/null || echo ""))

if [ ${#STRUCTURE_FILES[@]} -eq 0 ]; then
    echo "Error: No structure files found matching pattern: $STRUCTURE_PATTERN"
    exit 1
fi

echo "Found ${#STRUCTURE_FILES[@]} structure files"

# Define tools and methods
TOOLS_METHODS=(
    "flatprot:single:flatprot_single"
    "flatprot:family:flatprot_family"
    "prolego:single:prolego_single"
    "prolego:family:prolego_family"
    "ssdraw:single:ssdraw_single"
    "ssdraw:family:ssdraw_family"
    "pymol:single:pymol_single"
    "pymol:family:pymol_family"
)

# Run benchmarks
for tool_method in "${TOOLS_METHODS[@]}"; do
    IFS=':' read -r tool method func <<< "$tool_method"

    echo "Benchmarking $tool ($method)..."

    for iteration in $(seq 1 $N_ITERATIONS); do
        echo "  Iteration $iteration/$N_ITERATIONS"

        if [ "$method" = "single" ]; then
            # Single structure benchmarks
            for structure in "${STRUCTURE_FILES[@]}"; do
                echo "    Processing $structure"

                result=$(measure_execution "$func '$structure'")
                read -r exec_time memory exit_code error_msg <<< "$result"

                family_available="N/A"
                log_result "$tool" "$method" "$structure" "$iteration" "$exec_time" "$memory" "$exit_code" "$error_msg" "$family_available"
            done
        else
            # Family/overlay benchmarks (use all structures)
            echo "    Processing family with ${#STRUCTURE_FILES[@]} structures"

            # Convert array to space-separated string
            structures_str="${STRUCTURE_FILES[*]}"

            result=$(measure_execution "$func '$structures_str'")
            read -r exec_time memory exit_code error_msg <<< "$result"

            # Determine family availability based on exit code
            if [ "$exit_code" -eq 2 ]; then
                family_available="false"
            elif [ "$exit_code" -eq 0 ]; then
                family_available="true"
            else
                family_available="error"
            fi

            log_result "$tool" "$method" "multiple_structures" "$iteration" "$exec_time" "$memory" "$exit_code" "$error_msg" "$family_available"
        fi
    done
done

# Cleanup
rm -rf "$TEMP_DIR"

echo "Benchmark completed. Results saved to: $OUTPUT_FILE"
echo "Summary:"
echo "- Total iterations per tool/method: $N_ITERATIONS"
echo "- Structure files processed: ${#STRUCTURE_FILES[@]}"
echo "- Tools tested: flatprot, prolego, ssdraw, pymol"

# Display quick summary statistics
if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$OUTPUT_FILE')
    print('\nQuick Summary:')
    print('=' * 50)

    # Success rates by tool and method
    success_rates = df.groupby(['tool', 'method'])['exit_code'].apply(lambda x: (x == 0).mean() * 100).round(2)
    print('Success Rates (%):', success_rates.to_dict())

    # Average execution times for successful runs
    successful = df[df['exit_code'] == 0]
    if not successful.empty:
        avg_times = successful.groupby(['tool', 'method'])['execution_time_seconds'].mean().round(3)
        print('Average Execution Times (seconds):', avg_times.to_dict())

    # Family availability
    family_support = df[df['method'] == 'family']['family_available'].value_counts()
    print('Family Support:', family_support.to_dict())

except Exception as e:
    print(f'Could not generate summary: {e}')
"
fi
