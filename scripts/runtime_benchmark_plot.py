#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "scipy",
#     "matplotlib",
#     "numpy",
#     "polars",
#     "cycler",
# ]
# ///

"""Generates a log-log plot of runtime vs protein length with regression lines."""

import argparse
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cycler import cycler
from scipy import stats  # For linear regression

# --- Matplotlib Styling ---


# --- colors.py content ---
@dataclass
class OKABE_ITO_COLORS:
    lightblue = "#56B4E9"
    yellow = "#F0E442"
    orange = "#E69F00"
    green = "#009E73"
    purple = "#CC79A7"
    red = "#D55E00"
    blue = "#0072B2"
    black = "#000000"


OKABE_ITO_PALETTE = [
    OKABE_ITO_COLORS.orange,
    OKABE_ITO_COLORS.lightblue,
    OKABE_ITO_COLORS.green,
    OKABE_ITO_COLORS.yellow,
    OKABE_ITO_COLORS.blue,
    OKABE_ITO_COLORS.red,
    OKABE_ITO_COLORS.purple,
    OKABE_ITO_COLORS.black,
]

GRAYS_PALETTE = ["#000", "#666", "#ccc"]

# --- __init__.py content (simplified) ---
FONT_FAMILY = [
    "TUM Neue Helvetica",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]


class MPLParamSet(dict):
    """A dictionary subclass for managing Matplotlib parameter sets."""

    def __add__(self, other):
        """Combine two parameter sets."""
        result = MPLParamSet(self)
        result.update(other)
        return result

    @contextmanager
    def rc_context(self):
        """Use this parameter set as a context manager."""
        with mpl.rc_context(self):
            yield

    def __enter__(self):
        """Enter the context manager."""
        self._context = mpl.rc_context(rc=self)
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        return self._context.__exit__(exc_type, exc_val, exc_tb)


# Standard parameter set
standard_params = MPLParamSet(
    {
        "font.family": ["sans-serif"],
        "font.sans-serif": FONT_FAMILY,
        "pdf.fonttype": 42,
        "font.size": 10,
        "text.color": ".15",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": ".8",
        "axes.labelcolor": ".15",
        "axes.prop_cycle": cycler(color=OKABE_ITO_PALETTE),
        "xtick.color": ".15",
        "ytick.color": ".15",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.solid_capstyle": "round",
        "image.cmap": "Greys",
    }
)

# Grid parameter set
grid_params = MPLParamSet(
    {
        "axes.grid": True,
        "grid.color": ".9",
        "grid.linestyle": "-",
        "axes.axisbelow": True,
    }
)

# Line parameter set
line_params = MPLParamSet(
    {
        "lines.linewidth": 1.5,  # Adjusted for regression line visibility
        "lines.markersize": 4,
    }
)

# Legend parameter set
legend_params = MPLParamSet(
    {
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
    }
)

# Tick parameter set
tick_params = MPLParamSet(
    {
        "xtick.major.size": 0.0,
        "xtick.minor.size": 0.0,
        "ytick.major.size": 0.0,
        "ytick.minor.size": 0.0,
    }
)

# Axes parameter set
axes_params = MPLParamSet(
    {
        "axes.linewidth": 1.0,
    }
)

# --- End Matplotlib Styling ---

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def plot_runtime_vs_length(results_file: Path, plot_output_file: Path):
    """
    Generates a log-log scatter plot of runtime vs protein length
    with regression lines from a results TSV file.
    """
    log.info(f"--- Starting Result Plotting from {results_file} ---")

    # --- Read and Prepare Data ---
    if not results_file.exists():
        log.error(f"Results file not found: {results_file}. Cannot generate plot.")
        return

    try:
        log.info(f"Loading benchmark results from: {results_file}")
        df = pl.read_csv(results_file, separator="\t", infer_schema_length=1000)

        # Filter for valid runs: runtime > 0, length > 0, and no error recorded
        if "error" in df.columns:
            df_valid = df.filter(
                (pl.col("runtime") > 0)
                & (pl.col("length") > 0)
                & (pl.col("error").is_null() | (pl.col("error") == ""))
            )
        else:
            log.warning(
                "Column 'error' not found in results. Filtering only on runtime and length."
            )
            df_valid = df.filter((pl.col("runtime") > 0) & (pl.col("length") > 0))

        num_valid_points = df_valid.height
        num_total_points = df.height
        log.info(
            f"Found {num_valid_points} valid data points out of {num_total_points} total rows for plotting."
        )

        if num_valid_points == 0:
            log.warning(
                "No valid data points found after filtering. Cannot generate plot."
            )
            return

        # --- Print overall length stats ---
        if df_valid.height > 0:
            min_overall_len = df_valid["length"].min()
            max_overall_len = df_valid["length"].max()
            log.info(
                f"Overall valid length range: {min_overall_len} - {max_overall_len} residues."
            )
        else:
            log.info("No valid data points to calculate length range.")

        # Separate data by method for plotting
        inertia_data = df_valid.filter(pl.col("method") == "inertia")
        family_data = df_valid.filter(pl.col("method") == "family")
        log.info(
            f"Inertia points: {inertia_data.height}, Family points: {family_data.height}"
        )

        # --- Print runtime stats per method ---
        if inertia_data.height > 0:
            min_inertia_time = inertia_data["runtime"].min()
            max_inertia_time = inertia_data["runtime"].max()
            log.info(
                f"Inertia runtime range: {min_inertia_time:.4f}s - {max_inertia_time:.4f}s."
            )
        else:
            log.info("No valid data points for Inertia runtime range.")

        if family_data.height > 0:
            min_family_time = family_data["runtime"].min()
            max_family_time = family_data["runtime"].max()
            log.info(
                f"Family runtime range: {min_family_time:.4f}s - {max_family_time:.4f}s."
            )
        else:
            log.info("No valid data points for Family runtime range.")

        # --- Calculate Regression Lines (on log-transformed data) ---
        regression_lines = {}
        for method, data, color in [
            ("inertia", inertia_data, OKABE_ITO_PALETTE[1]),  # lightblue
            ("family", family_data, OKABE_ITO_PALETTE[2]),  # green
        ]:
            if data.height > 1:  # Need at least 2 points for regression
                log_length = np.log10(data["length"].to_numpy())
                log_runtime = np.log10(data["runtime"].to_numpy())

                # Perform linear regression: log_runtime = slope * log_length + intercept
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_length, log_runtime
                )
                log.info(
                    f"Method '{method}': log10(Runtime) = {slope:.2f} * log10(Length) + {intercept:.2f} (R^2={r_value**2:.2f})"
                )

                # Generate points for the line plot
                min_log_len = log_length.min()
                max_log_len = log_length.max()
                x_line_log = np.linspace(min_log_len, max_log_len, 100)
                y_line_log = slope * x_line_log + intercept

                # Convert back to original scale
                x_line = 10**x_line_log
                y_line = 10**y_line_log
                regression_lines[method] = {
                    "x": x_line,
                    "y": y_line,
                    "color": color,
                    "slope": slope,
                }
            else:
                log.warning(
                    f"Not enough data points to calculate regression for '{method}'."
                )

        # --- Create Plot ---
        plot_params = (
            standard_params
            + grid_params
            + line_params
            + legend_params
            + tick_params
            + axes_params
        )
        with plot_params:
            fig, ax = plt.subplots(figsize=(5, 3.5))  # Adjusted size

            scatter_config = {
                "s": 15,
                "linewidths": 0.5,
                "alpha": 0.4,
                "edgecolors": "w",
            }

            # Plot inertia (Blue)
            ax.scatter(
                inertia_data["length"],
                inertia_data["runtime"],
                color=OKABE_ITO_PALETTE[1],  # lightblue
                marker="+",
                **scatter_config,
            )

            # Plot family (Green) - alignment-based
            ax.scatter(
                family_data["length"],
                family_data["runtime"],
                color=OKABE_ITO_PALETTE[2],  # green
                marker="+",
                **scatter_config,
            )

            # Plot Regression Lines
            for method, line_data in regression_lines.items():
                (line,) = ax.plot(
                    line_data["x"],
                    line_data["y"],
                    color=line_data["color"],
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.8,
                )

            # --- Configure Axes and Title ---
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(
                "Protein Length (Number of Residues)", loc="right", fontweight="bold"
            )
            ax.set_ylabel("Runtime (s)", loc="top", fontweight="bold")

            # --- Dynamic Ticks based on Data Range ---
            all_lengths = df_valid["length"]
            all_runtimes = df_valid["runtime"]

            min_len = all_lengths.min() if not all_lengths.is_empty() else 10
            max_len = all_lengths.max() if not all_lengths.is_empty() else 10000
            min_time = all_runtimes.min() if not all_runtimes.is_empty() else 0.01
            max_time = all_runtimes.max() if not all_runtimes.is_empty() else 100

            try:
                x_tick_vals = 10 ** np.arange(
                    np.floor(np.log10(min_len)), np.ceil(np.log10(max_len)) + 1
                )
                y_tick_vals = 10 ** np.arange(
                    np.floor(np.log10(min_time)), np.ceil(np.log10(max_time)) + 1
                )

                x_tick_labels = [
                    f"{t:.0f}" if t >= 1 else f"{t:.1g}" for t in x_tick_vals
                ]
                y_tick_labels = [
                    f"{t:.0f}" if t >= 1 else f"{t:.2g}" for t in y_tick_vals
                ]

                ax.set_xticks(x_tick_vals)
                ax.set_xticklabels(x_tick_labels, fontsize="small")
                ax.set_yticks(y_tick_vals)
                ax.set_yticklabels(y_tick_labels, fontsize="small")
            except ValueError:
                log.warning("Could not set dynamic log ticks, using default.")

            # Add text labels directly to the plot near the end of regression lines
            for method, line_data in regression_lines.items():
                x_pos = line_data["x"][-1] * 1.2  # Last x point, with slight offset
                y_pos = line_data["y"][-1]  # Last y point
                ax.text(
                    x_pos,
                    y_pos,
                    f"{method.capitalize()}",
                    color=line_data["color"],
                    ha="left",
                    va="center",
                    fontsize="small",
                    fontweight="bold",
                )

            # --- Save and Show Plot ---
            try:
                plot_output_file.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_output_file, dpi=300, bbox_inches="tight")
                log.info(f"Plot saved successfully to: {plot_output_file}")
            except Exception as save_err:
                log.error(
                    f"Failed to save plot to {plot_output_file}: {save_err}",
                    exc_info=True,
                )

            plt.show()

    except ImportError as import_err:
        log.error(
            f"Required library (matplotlib, numpy, scipy, polars) not found: {import_err}. Please install it."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred during plotting: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate runtime vs length plot with regression lines from benchmark results."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default="tmp/runtime/results/runtime_benchmark_results.tsv",
        help="Path to the input TSV file containing benchmark results.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="tmp/runtime/results/runtime_vs_length_regression_plot.png",
        help="Path to save the output plot PNG file.",
    )
    args = parser.parse_args()

    plot_runtime_vs_length(args.input, args.output)
