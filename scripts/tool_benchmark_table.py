#!/usr/bin/env -S uv --quiet run --script

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "polars"
# ]
# ///
"""
Benchmark Results Analysis Script.

This script analyzes benchmark results from CSV files, calculating averages
and standard errors for different tool-method combinations, and outputs
a formatted typst table.

For single and aligned methods: expects 10 entries per tool
For family and family_large methods: expects 10 entries per tool
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl


def calculate_stats(values: List[float]) -> Tuple[float, float]:
    """
    Calculate mean and standard error for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Tuple of (mean, standard_error)
    """
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std_dev = variance**0.5
    std_error = std_dev / (len(values) ** 0.5)

    return mean, std_error


def load_and_validate_data(csv_file: Path) -> pl.DataFrame:
    """
    Load CSV data and validate its structure.

    Args:
        csv_file: Path to the CSV file

    Returns:
        Polars DataFrame with the benchmark data

    Raises:
        ValueError: If the CSV structure is invalid
    """
    try:
        df = pl.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    required_columns = ["tool", "method", "execution_time_seconds", "memory_mb"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def analyze_benchmark_data(df: pl.DataFrame) -> Dict[str, Dict[str, Dict]]:
    """
    Analyze benchmark data and calculate statistics for each tool-method combination.

    Args:
        df: Polars DataFrame with benchmark data

    Returns:
        Dictionary mapping tool -> method -> statistics dict
    """
    results = {}

    # Get unique tool-method combinations
    tool_method_combinations = (
        df.select(["tool", "method"]).unique().sort(["tool", "method"])
    )

    for row in tool_method_combinations.iter_rows(named=True):
        tool = row["tool"]
        method = row["method"]

        # Filter data for this tool-method combination
        subset = df.filter((pl.col("tool") == tool) & (pl.col("method") == method))

        # Extract timing and memory data
        times = subset["execution_time_seconds"].to_list()
        memories = subset["memory_mb"].to_list()

        if len(times) == 0:
            print(f"Warning: No data found for {tool}-{method}")
            continue

        # Calculate statistics
        mean_time, std_err_time = calculate_stats(times)
        mean_memory, std_err_memory = calculate_stats(memories)

        if tool not in results:
            results[tool] = {}

        results[tool][method] = {
            "time_mean": mean_time,
            "time_std_err": std_err_time,
            "memory_mean": mean_memory,
            "memory_std_err": std_err_memory,
            "count": len(times),
        }

        print(
            f"{tool}-{method}: {len(times)} entries, "
            f"time: {mean_time:.4f}±{std_err_time:.4f}s, "
            f"memory: {mean_memory:.2f}±{std_err_memory:.2f}MB"
        )

    return results


def format_value_with_error(mean: float, std_err: float, precision: int = 3) -> str:
    """
    Format a value with its standard error for display using significant figures.

    Args:
        mean: Mean value
        std_err: Standard error
        precision: Maximum number of decimal places (used as fallback)

    Returns:
        Formatted string like "0.264±0.008" or "96±2" based on significance
    """
    if std_err == 0.0:
        # No error, format based on magnitude
        if mean >= 100:
            return f"{mean:.0f}"
        elif mean >= 10:
            return f"{mean:.1f}"
        elif mean >= 1:
            return f"{mean:.2f}"
        else:
            return f"{mean:.3f}"

    # Determine appropriate decimal places based on standard error
    if std_err >= 10:
        # Large errors, show no decimal places
        decimal_places = 0
    elif std_err >= 1:
        # Moderate errors, show 1 decimal place
        decimal_places = 1
    elif std_err >= 0.1:
        # Small errors, show 1-2 decimal places
        decimal_places = 1 if std_err >= 0.5 else 2
    elif std_err >= 0.01:
        # Very small errors, show 2-3 decimal places
        decimal_places = 2 if std_err >= 0.05 else 3
    else:
        # Tiny errors, show 3 decimal places max
        decimal_places = 3

    # Don't exceed the provided precision limit
    decimal_places = min(decimal_places, precision)

    return f"{mean:.{decimal_places}f}±{std_err:.{decimal_places}f}"


def generate_typst_table(
    results: Dict[str, Dict[str, Dict]], metric: str = "time"
) -> str:
    """
    Generate a typst table from the analysis results.

    Args:
        results: Analysis results dictionary
        metric: Either 'time' or 'memory' to specify which metric to display

    Returns:
        Formatted typst table string
    """
    # Tool name mapping from CSV to display names
    tool_name_mapping = {
        "ssdraw": "SSDraw",
        "pymol": "PyMol",
        "flatprot": "FlatProt",
        "chimerax": "ChimeraX",
        "proorigami": "Pro-origami",  # Assuming this is the CSV name
    }

    # Define custom tool order: Pro-origami, SSDraw, FlatProt (bold), ChimeraX, PyMol
    tool_order = ["proorigami", "ssdraw", "flatprot", "chimerax", "pymol"]

    # Filter and sort tools based on what's available in results and our custom order
    available_tools = [tool for tool in tool_order if tool in results]

    # Start building the table with 4 columns (removed Aligned column)
    table_lines = [
        "#table(",
        "  columns: 4,",
        "  stroke: none,",
        '  fill: (x, y) => if y == 0 { rgb("#f0f0f0") } else if calc.odd(y) { rgb("#f8f8f8") } else { white },',
        "  [*Tool*], [*Single*], [*Family*], [*Family Large*],",
    ]

    for i, tool in enumerate(available_tools):
        tool_data = results.get(tool, {})

        # Get display name and make FlatProt bold
        display_name = tool_name_mapping.get(tool, tool)
        if tool == "flatprot":
            tool_cell = f"  [*{display_name}*]"
        else:
            tool_cell = f"  [{display_name}]"

        row_parts = [tool_cell]

        # Single column
        if "single" in tool_data:
            stats = tool_data["single"]

            # Choose metric values and units
            if metric == "memory":
                mean_key, err_key = "memory_mean", "memory_std_err"
                unit = "MB"
                max_precision = 1  # Memory usually doesn't need more than 1 decimal
            else:
                mean_key, err_key = "time_mean", "time_std_err"
                unit = "s"
                max_precision = (
                    3  # Time can go up to 3 decimals for millisecond precision
                )

            # For flatprot single, we want to show aligned as an alternative
            if tool == "flatprot" and "aligned" in tool_data:
                aligned_stats = tool_data["aligned"]
                single_value = format_value_with_error(
                    stats[mean_key], stats[err_key], max_precision
                )
                aligned_value = format_value_with_error(
                    aligned_stats[mean_key], aligned_stats[err_key], max_precision
                )
                cell_content = f"[{single_value}{unit} \\ ({aligned_value}{unit})]"
            else:
                value_str = format_value_with_error(
                    stats[mean_key], stats[err_key], max_precision
                )
                cell_content = f"[{value_str}{unit}]"
        else:
            cell_content = "[--]"
        row_parts.append(cell_content)

        # Family column
        if "family" in tool_data:
            stats = tool_data["family"]
            if metric == "memory":
                mean_key, err_key = "memory_mean", "memory_std_err"
                unit = "MB"
                max_precision = 1
            else:
                mean_key, err_key = "time_mean", "time_std_err"
                unit = "s"
                max_precision = 3

            value_str = format_value_with_error(
                stats[mean_key], stats[err_key], max_precision
            )
            row_parts.append(f"[{value_str}{unit}]")
        else:
            row_parts.append("[--]")

        # Family Large column
        if "family_large" in tool_data:
            stats = tool_data["family_large"]
            if metric == "memory":
                mean_key, err_key = "memory_mean", "memory_std_err"
                unit = "MB"
                max_precision = 1
            else:
                mean_key, err_key = "time_mean", "time_std_err"
                unit = "s"
                max_precision = 3

            value_str = format_value_with_error(
                stats[mean_key], stats[err_key], max_precision
            )
            row_parts.append(f"[{value_str}{unit}]")
        else:
            row_parts.append("[--]")

        table_lines.append(", ".join(row_parts) + ",")

    table_lines.append(")")

    return "\n".join(table_lines)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate a typst table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.csv
  %(prog)s /path/to/benchmark_data.csv -o output_table.typ
        """,
    )

    parser.add_argument(
        "csv_file", type=Path, help="Path to the CSV file containing benchmark results"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark_results_table.typ"),
        help="Output file for the typst table (default: benchmark_results_table.typ)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "-m",
        "--metric",
        choices=["time", "memory", "both"],
        default="both",
        help="Which metric to generate table for (default: both)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main function to execute the benchmark analysis.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    # Check if file exists
    if not args.csv_file.exists():
        print(f"Error: CSV file '{args.csv_file}' not found.", file=sys.stderr)
        return 1

    try:
        # Load and validate data
        if args.verbose:
            print(f"Loading data from {args.csv_file}...")
        df = load_and_validate_data(args.csv_file)
        if args.verbose:
            print(f"Loaded {len(df)} rows of data")

        # Analyze the data
        if args.verbose:
            print("\nAnalyzing benchmark data...")
        results = analyze_benchmark_data(df)

        if not results:
            print("Error: No valid data found for analysis.", file=sys.stderr)
            return 1

        # Generate typst table(s)
        if args.verbose:
            print("\nGenerating typst table(s)...")

        output_content = []

        if args.metric in ["time", "both"]:
            if args.verbose:
                print("\nGenerating TIME table...")
            time_table = generate_typst_table(results, "time")
            output_content.append("// Execution Time Table")
            output_content.append(time_table)

            if args.verbose:
                print("\n" + "=" * 60)
                print("TIME TABLE OUTPUT:")
                print("=" * 60)
                print(time_table)
                print("=" * 60)

        if args.metric in ["memory", "both"]:
            if args.verbose:
                print("\nGenerating MEMORY table...")
            memory_table = generate_typst_table(results, "memory")
            output_content.append("// Memory Usage Table")
            output_content.append(memory_table)

            if args.verbose:
                print("\n" + "=" * 60)
                print("MEMORY TABLE OUTPUT:")
                print("=" * 60)
                print(memory_table)
                print("=" * 60)

        # Combine tables with spacing
        final_output = "\n\n".join(output_content)

        # Output the table(s)
        if not args.verbose:
            print(final_output)

        # Save to file
        with args.output.open("w") as f:
            f.write(final_output)

        if args.verbose:
            print(f"\nTable(s) saved to {args.output}")

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
