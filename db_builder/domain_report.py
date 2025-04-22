from pathlib import Path
from typing import Dict, List, Tuple
import json
import polars as pl
import gemmi
from snakemake.script import snakemake


def get_domain_length(domain_file: Path) -> int:
    """Get the number of residues in a domain structure using gemmi.

    Args:
        domain_file: Path to the domain CIF file

    Returns:
        Number of residues in the domain
    """
    try:
        structure = gemmi.read_structure(str(domain_file))
        return len(structure[0][0])  # First model, first chain
    except Exception:
        return 0


def analyze_extraction_results(
    superfamilies_file: Path, pdb_dir: Path, domain_dir: Path
) -> Tuple[Dict[str, float], List[Dict[str, str]], Dict[str, Dict]]:
    """Analyze domain extraction results and generate statistics.

    Args:
        superfamilies_file: Path to the superfamilies TSV file
        pdb_dir: Directory containing PDB files and status information
        domain_dir: Directory containing extracted domains

    Returns:
        Tuple containing:
        - Dictionary of overall statistics
        - List of failed extractions with error details
        - Dictionary of per-superfamily statistics
    """
    # Read superfamilies data
    df = pl.read_csv(superfamilies_file, separator="\t")

    # Analyze per superfamily
    sf_stats = {}
    total_residues = 0
    total_domains = 0

    for sf_dir in domain_dir.glob("*"):
        if sf_dir.is_dir():
            sf_id = sf_dir.name
            domain_files = list(sf_dir.glob("*.cif"))

            # Get domain lengths
            lengths = []
            for domain_file in domain_files:
                length = get_domain_length(domain_file)
                if length > 0:
                    lengths.append(length)

            if lengths:
                avg_length = sum(lengths) / len(lengths)
                total_residues += sum(lengths)
                total_domains += len(lengths)
            else:
                avg_length = 0

            sf_stats[sf_id] = {
                "domains": len(domain_files),
                "avg_length": avg_length,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
            }

    # Calculate overall statistics
    total_expected = df.shape[0]  # Total number of domains in superfamilies file
    total_extracted = total_domains

    # Collect failed extractions
    failed_extractions = []
    for pdb_status in pdb_dir.glob("*.status"):
        with open(pdb_status, "r") as f:
            status = json.load(f)
            if not status.get("success", False):
                failed_extractions.append(
                    {
                        "pdb_id": pdb_status.stem,
                        "error": status.get("error", "Unknown error"),
                    }
                )

    stats = {
        "total_expected_domains": total_expected,
        "total_extracted_domains": total_extracted,
        "extraction_success_rate": (total_extracted / total_expected) * 100
        if total_expected > 0
        else 0,
        "average_domain_length": total_residues / total_domains
        if total_domains > 0
        else 0,
        "failed_extractions": len(failed_extractions),
    }

    return stats, failed_extractions, sf_stats


def generate_report(superfamilies_file: Path, pdb_dir: Path, domain_dir: Path) -> str:
    """Generate a reStructuredText report for domain extraction results.

    Args:
        superfamilies_file: Path to the superfamilies TSV file
        pdb_dir: Directory containing PDB files and status information
        domain_dir: Directory containing extracted domains

    Returns:
        Report content in reStructuredText format
    """
    stats, failed_extractions, sf_stats = analyze_extraction_results(
        superfamilies_file, pdb_dir, domain_dir
    )

    report = [
        "Domain Extraction Report",
        "=====================",
        "",
        "Summary",
        "-------",
        f"* Expected domains: {stats['total_expected_domains']}",
        f"* Successfully extracted domains: {stats['total_extracted_domains']}",
        f"* Success rate: {stats['extraction_success_rate']:.2f}%",
        f"* Average domain length: {stats['average_domain_length']:.1f} residues",
        f"* Failed extractions: {stats['failed_extractions']}",
        "",
        "Superfamily Statistics",
        "--------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Superfamily ID",
        "     - Domains",
        "     - Avg Length",
        "     - Min Length",
        "     - Max Length",
    ]

    for sf_id, sf_data in sorted(sf_stats.items()):
        report.append(
            f"   * - {sf_id}\n"
            f"     - {sf_data['domains']}\n"
            f"     - {sf_data['avg_length']:.1f}\n"
            f"     - {sf_data['min_length']}\n"
            f"     - {sf_data['max_length']}"
        )

    if failed_extractions:
        report.extend(
            [
                "",
                "Failed Extractions",
                "-----------------",
                "",
                ".. list-table::",
                "   :header-rows: 1",
                "",
                "   * - PDB ID",
                "     - Error",
            ]
        )

        for failure in failed_extractions:
            report.append(f"   * - {failure['pdb_id']}\n     - {failure['error']}")

    return "\n".join(report)


if __name__ == "__main__":
    # Access Snakemake parameters
    superfamilies_file = Path(snakemake.input.superfamilies)
    pdb_dir = Path(snakemake.input.pdb_dir)
    domain_dir = Path(snakemake.input.domain_dir)
    output_file = Path(snakemake.output.report)

    # Generate and write report
    report_content = generate_report(superfamilies_file, pdb_dir, domain_dir)
    output_file.write_text(report_content)
