"""Parquet loading and immutable raw-data contract validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


EXPECTED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_count",
    "wap",
    "contract_con_id",
    "contract_expiry",
    "local_symbol",
    "root",
)

EXPECTED_ARROW_TYPES = {
    "timestamp": pa.timestamp("ns", tz="UTC"),
    "open": pa.float64(),
    "high": pa.float64(),
    "low": pa.float64(),
    "close": pa.float64(),
    "volume": pa.float64(),
    "bar_count": pa.int64(),
    "wap": pa.float64(),
    "contract_con_id": pa.int64(),
    "contract_expiry": pa.int64(),
    "local_symbol": pa.string(),
    "root": pa.string(),
}


class DataContractError(ValueError):
    """Raised when a dataset violates the documented raw-data contract."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise DataContractError(f"Configuration must be a mapping: {path}")
    return value


def base_config() -> dict[str, Any]:
    return load_yaml(project_root() / "configs" / "base.yaml")


def raw_data_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or base_config()
    configured = Path(cfg["paths"]["raw_parquet"])
    return configured if configured.is_absolute() else project_root() / configured


def _validate_arrow_schema(path: Path) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    if tuple(schema.names) != EXPECTED_COLUMNS:
        raise DataContractError(
            f"Unexpected columns: expected {EXPECTED_COLUMNS}, got {tuple(schema.names)}"
        )
    mismatches = {
        name: (EXPECTED_ARROW_TYPES[name], schema.field(name).type)
        for name in EXPECTED_COLUMNS
        if schema.field(name).type != EXPECTED_ARROW_TYPES[name]
    }
    if mismatches:
        raise DataContractError(f"Unexpected Arrow dtypes: {mismatches}")


def validate_raw_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_roots: Iterable[str] | None = None,
    expected_contracts: int | None = None,
) -> None:
    """Fail fast on violations that would invalidate all downstream research."""
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise DataContractError("Loaded columns do not match the raw-data contract")
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataContractError(f"Expected {expected_rows} rows, got {len(frame)}")
    if frame.isna().to_numpy().any():
        raise DataContractError("Raw data contains null values")
    if frame.duplicated(["root", "timestamp"]).any():
        raise DataContractError("Duplicate (root, timestamp) keys found")
    if not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        raise DataContractError("timestamp must be timezone-aware")
    if str(frame["timestamp"].dt.tz) != "UTC":
        raise DataContractError("timestamp timezone must be UTC")
    if expected_roots is not None and set(frame["root"].unique()) != set(expected_roots):
        raise DataContractError(
            f"Expected roots {sorted(expected_roots)}, got {sorted(frame['root'].unique())}"
        )
    if expected_contracts is not None and frame["local_symbol"].nunique() != expected_contracts:
        raise DataContractError(
            f"Expected {expected_contracts} contracts, got {frame['local_symbol'].nunique()}"
        )


def load(
    path: str | Path | None = None,
    *,
    roots: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load raw minute bars with predicate pushdown and strict validation.

    The source parquet remains read-only. Root filtering is pushed into PyArrow so
    callers do not pay the memory cost of loading unrelated instruments.
    """
    cfg = base_config()
    source = Path(path) if path is not None else raw_data_path(cfg)
    if not source.is_file():
        raise FileNotFoundError(source)

    _validate_arrow_schema(source)
    requested_roots = tuple(dict.fromkeys(roots or ()))
    known_roots = set(cfg["data_contract"]["expected_roots"])
    unknown = set(requested_roots) - known_roots
    if unknown:
        raise DataContractError(f"Unknown roots requested: {sorted(unknown)}")

    filters = [("root", "in", requested_roots)] if requested_roots else None
    frame = pd.read_parquet(source, engine="pyarrow", filters=filters)

    if requested_roots:
        validate_raw_frame(frame, expected_roots=requested_roots)
    else:
        contract = cfg["data_contract"]
        validate_raw_frame(
            frame,
            expected_rows=int(contract["expected_rows"]),
            expected_roots=contract["expected_roots"],
            expected_contracts=int(contract["expected_contracts"]),
        )
    return frame


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_hash(root: str | Path | None = None) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root or project_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(
    output_path: str | Path,
    *,
    inputs: Iterable[str | Path],
    configs: Iterable[str | Path],
) -> dict[str, Any]:
    """Write a deterministic provenance manifest beside a stage artifact."""
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "inputs": {str(Path(p).resolve()): sha256_file(p) for p in inputs},
        "configs": {str(Path(p).resolve()): sha256_file(p) for p in configs},
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest

