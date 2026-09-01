"""Typed experiment configuration.

Every number that changes an experimental result lives in YAML and is loaded into
these dataclasses. Nothing downstream reads a literal, so a run is fully described
by its config file plus the git revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    smd_root: Path
    calibration_machines: tuple[str, ...]
    development_machines: tuple[str, ...]
    heldout_machines: tuple[str, ...]

    def machines_for(self, split: str) -> tuple[str, ...]:
        match split:
            case "calibration":
                return self.calibration_machines
            case "development":
                return self.development_machines
            case "heldout":
                return self.heldout_machines
            case _:
                raise ValueError(f"unknown split {split!r}")


@dataclass(frozen=True)
class WindowConfig:
    size: int
    stride: int
    reference_size: int
    policy: str  # fixed | sliding | reset_on_alarm


@dataclass(frozen=True)
class InjectionConfig:
    onset_fraction: float
    n_affected_features: int
    magnitudes: tuple[float, ...]
    gradual_ramp: int
    block_shuffle_length: int
    seeds: tuple[int, ...]
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    tolerance_windows: int
    target_false_alarms_per_stream: float
    target_false_alarms_per_window: float
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    window: WindowConfig
    injection: InjectionConfig
    evaluation: EvaluationConfig
    detectors: tuple[str, ...]
    artifacts_dir: Path
    extra: dict[str, Any] = field(default_factory=dict)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(path).resolve().parent.parent

    data_raw = raw["data"]
    data = DataConfig(
        smd_root=(root / data_raw["smd_root"]).resolve(),
        calibration_machines=_as_tuple(data_raw["calibration_machines"]),
        development_machines=_as_tuple(data_raw["development_machines"]),
        heldout_machines=_as_tuple(data_raw["heldout_machines"]),
    )
    window = WindowConfig(**raw["window"])
    inj_raw = dict(raw["injection"])
    inj_raw["magnitudes"] = _as_tuple(inj_raw["magnitudes"])
    inj_raw["seeds"] = _as_tuple(inj_raw["seeds"])
    inj_raw["scenarios"] = _as_tuple(inj_raw["scenarios"])
    injection = InjectionConfig(**inj_raw)
    evaluation = EvaluationConfig(**raw["evaluation"])

    return ExperimentConfig(
        data=data,
        window=window,
        injection=injection,
        evaluation=evaluation,
        detectors=_as_tuple(raw["detectors"]),
        artifacts_dir=(root / raw.get("artifacts_dir", "artifacts")).resolve(),
        extra=raw.get("extra", {}),
    )
