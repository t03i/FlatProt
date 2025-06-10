# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, NamedTuple, List, Dict
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import polars as pl

from flatprot.transformation import TransformationMatrix


class AlignmentResult(NamedTuple):
    """Results from a structural family alignment."""

    db_id: str
    probability: float
    aligned_region: np.ndarray
    alignment_scores: np.ndarray
    rotation_matrix: TransformationMatrix


class FoldseekAligner:
    """Handles structural family alignments using FoldSeek."""

    def __init__(
        self,
        foldseek_executable: str,
        database_path: Path,
    ):
        self.foldseek_executable = foldseek_executable
        self.database_path = database_path

    def align_structure(
        self,
        structure_path: Path,
        min_probability: float = 0.5,
        fixed_alignment_id: Optional[str] = None,
        tmp_dir: Optional[Path] = None,
    ) -> Optional[AlignmentResult]:
        """Aligns structure to family database and returns best match."""

        if tmp_dir is None:
            tmp_dir = tempfile.TemporaryDirectory(
                ignore_cleanup_errors=True, prefix="foldseek_"
            )
            tmp_dir_path = Path(tmp_dir.name)
        else:
            tmp_dir_path = tmp_dir

        # Run FoldSeek search
        result_file = tmp_dir_path / "foldseek_result.tsv"
        foldseek_tmp_dir = tmp_dir_path / "foldseek_tmp"

        self._run_foldseek_search(structure_path, result_file, foldseek_tmp_dir)

        if not result_file.exists():
            raise RuntimeError("FoldSeek result file not found")

        # Parse results
        results = pl.read_csv(
            result_file,
            separator="\t",
            has_header=False,
            new_columns=[
                "query",
                "target",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "tseq",
                "prob",
                "alntmscore",
                "u",
                "t",
                "lddtfull",
            ],
            schema_overrides={
                "query": pl.Utf8,
                "target": pl.Utf8,
            },
        )
        if len(results) == 0:
            return None

        # Get best match (or fixed family if specified)
        if fixed_alignment_id:
            match = results.filter(pl.col("target") == fixed_alignment_id)
        else:
            match = results.sort("prob", descending=True)
            if match[0, "prob"] < min_probability:
                return None

        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup()

        target_to_query_matrix = _parse_foldseek_vector(match[0, "u"]).reshape(3, 3)
        target_to_query_translation = _parse_foldseek_vector(match[0, "t"])

        # Manually calculate the inverse of the alignment transformation
        # Inverse rotation is the transpose of the rotation matrix
        R_align_inv = target_to_query_matrix.T
        # Inverse translation is -R_inv @ t
        t_align_inv = -R_align_inv @ target_to_query_translation
        query_to_target_transform = TransformationMatrix(
            rotation=R_align_inv, translation=t_align_inv
        )

        return AlignmentResult(
            db_id=match[0, "target"],
            probability=float(match[0, "prob"]),
            aligned_region=np.array((int(match[0, "qstart"]), int(match[0, "qend"]))),
            alignment_scores=_parse_foldseek_vector(match[0, "lddtfull"]),
            rotation_matrix=query_to_target_transform,
        )

    def align_structures_batch(
        self,
        structure_paths: List[Path],
        min_probability: float = 0.5,
        fixed_alignment_id: Optional[str] = None,
        tmp_dir: Optional[Path] = None,
    ) -> Dict[Path, Optional[AlignmentResult]]:
        """Batch align multiple structures to family database for better performance.

        Args:
            structure_paths: List of structure file paths to align
            min_probability: Minimum alignment probability threshold
            fixed_alignment_id: Optional fixed family ID for all alignments
            tmp_dir: Optional temporary directory path

        Returns:
            Dictionary mapping structure paths to their alignment results
        """
        if tmp_dir is None:
            tmp_dir = tempfile.TemporaryDirectory(
                ignore_cleanup_errors=True, prefix="foldseek_batch_"
            )
            tmp_dir_path = Path(tmp_dir.name)
        else:
            tmp_dir_path = tmp_dir

        # Create input directory with all structures
        structures_dir = tmp_dir_path / "batch_structures"
        structures_dir.mkdir(parents=True, exist_ok=True)

        # Copy all structures to batch directory
        structure_mapping = {}
        for i, structure_path in enumerate(structure_paths):
            # Use index to avoid naming conflicts
            batch_filename = (
                f"structure_{i}_{structure_path.stem}{structure_path.suffix}"
            )
            batch_file = structures_dir / batch_filename
            batch_file.write_bytes(structure_path.read_bytes())
            structure_mapping[batch_filename] = structure_path

        # Run batch foldseek search
        result_file = tmp_dir_path / "batch_foldseek_result.tsv"
        foldseek_tmp_dir = tmp_dir_path / "foldseek_tmp"

        self._run_foldseek_batch_search(structures_dir, result_file, foldseek_tmp_dir)

        if not result_file.exists():
            raise RuntimeError("FoldSeek batch result file not found")

        # Parse batch results
        try:
            results = pl.read_csv(
                result_file,
                separator="\t",
                has_header=False,
                new_columns=[
                    "query",
                    "target",
                    "qstart",
                    "qend",
                    "tstart",
                    "tend",
                    "tseq",
                    "prob",
                    "alntmscore",
                    "u",
                    "t",
                    "lddtfull",
                ],
                schema_overrides={
                    "query": pl.Utf8,
                    "target": pl.Utf8,
                },
            )
        except Exception:
            # Return empty results if parsing fails
            return {path: None for path in structure_paths}

        # Group results by query and process each structure
        alignment_results = {}
        for structure_path in structure_paths:
            # Find the batch filename for this structure
            batch_filename = None
            for batch_name, orig_path in structure_mapping.items():
                if orig_path == structure_path:
                    batch_filename = batch_name
                    break

            if batch_filename is None:
                alignment_results[structure_path] = None
                continue

            # Filter results for this specific query
            query_results = results.filter(
                pl.col("query").str.contains(Path(batch_filename).stem)
            )

            if len(query_results) == 0:
                alignment_results[structure_path] = None
                continue

            # Get best match (or fixed family if specified)
            if fixed_alignment_id:
                match = query_results.filter(pl.col("target") == fixed_alignment_id)
                if len(match) == 0:
                    alignment_results[structure_path] = None
                    continue
            else:
                match = query_results.sort("prob", descending=True)
                if match[0, "prob"] < min_probability:
                    alignment_results[structure_path] = None
                    continue

            # Create alignment result
            target_to_query_matrix = _parse_foldseek_vector(match[0, "u"]).reshape(3, 3)
            target_to_query_translation = _parse_foldseek_vector(match[0, "t"])

            # Calculate inverse transformation
            R_align_inv = target_to_query_matrix.T
            t_align_inv = -R_align_inv @ target_to_query_translation
            query_to_target_transform = TransformationMatrix(
                rotation=R_align_inv, translation=t_align_inv
            )

            alignment_results[structure_path] = AlignmentResult(
                db_id=match[0, "target"],
                probability=float(match[0, "prob"]),
                aligned_region=np.array(
                    (int(match[0, "qstart"]), int(match[0, "qend"]))
                ),
                alignment_scores=_parse_foldseek_vector(match[0, "lddtfull"]),
                rotation_matrix=query_to_target_transform,
            )

        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup()

        return alignment_results

    def _run_foldseek_batch_search(
        self, structures_dir: Path, output_file: Path, tmp_dir: Path
    ) -> None:
        """Runs FoldSeek batch search against family database."""
        cmd = [
            self.foldseek_executable,
            "easy-search",
            str(structures_dir),
            str(self.database_path),
            str(output_file),
            str(tmp_dir),
            "--format-output",
            "query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore,u,t,lddtfull",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _run_foldseek_search(
        self, structure_path: Path, output_file: Path, tmp_dir: Path
    ) -> None:
        """Runs FoldSeek search against family database."""
        cmd = [
            self.foldseek_executable,
            "easy-search",
            str(structure_path),
            str(self.database_path),
            str(output_file),
            str(tmp_dir),
            "--format-output",
            "query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore,u,t,lddtfull",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)


def _parse_foldseek_vector(vector_str: str) -> np.ndarray:
    """Parse vector string in format '(x, y, z)' or '(x, y, z, w, ...)' to numpy array."""
    # Remove parentheses and split by comma
    nums = vector_str.strip("()")
    # Convert to float array
    return np.fromstring(nums, sep=",")
