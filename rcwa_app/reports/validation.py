from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from typing_extensions import TypedDict


class Thresholds(TypedDict):
    pass_th: float
    warn_th: float


@dataclass(frozen=True)
class ValidationRecord:
    dataset_name: str
    theta_deg: float
    pol: str
    rmse: float
    thresholds: Thresholds
    timestamp: str  # ISO 8601


def make_record(
    dataset_name: str, theta_deg: float, pol: str, rmse: float, *, pass_th: float, warn_th: float
) -> ValidationRecord:
    ts = datetime.now(timezone.utc).isoformat()
    return ValidationRecord(
        dataset_name=dataset_name,
        theta_deg=theta_deg,
        pol=pol,
        rmse=rmse,
        thresholds={"pass_th": pass_th, "warn_th": warn_th},
        timestamp=ts,
    )


def to_json_bytes(rec: ValidationRecord) -> bytes:
    return json.dumps(asdict(rec), indent=2).encode("utf-8")
