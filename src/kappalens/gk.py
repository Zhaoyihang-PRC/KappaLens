"""Group-resolved Green–Kubo conductivity from an extensive heat-current CSV.

Q is the *total* heat current, in eV Angstrom / ps. It is not a heat-flux
density. The source simulation must use a compatible per-atom energy/stress
definition for the chosen potential. All ordered group cross terms are kept;
for groups A and B, total conductivity includes AA, AB, BA, and BB.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import __version__
from .common import (AnalysisError, current_groups, directions, model_output,
                     resolve, sha256, write_csv, write_json)

EV_J = 1.602176634e-19
ANG_M = 1e-10
PS_S = 1e-12
KB_J = 1.380649e-23
AVOGADRO = 6.02214076e23


def _read_currents(path: Path, groups: list[str], unit: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict]:
    """Normalize LAMMPS metal/real currents to eV Angstrom / ps."""
    factors = {
        "ev_angstrom_per_ps": 1.0,
        "kcal_per_mol_angstrom_per_fs": (4184.0 / AVOGADRO / EV_J) * 1000.0,
    }
    if unit not in factors:
        raise AnalysisError(f"Unknown heat_current_unit '{unit}'. Use one of: {', '.join(factors)}")
    factor = factors[unit]
    if not path.is_file():
        raise AnalysisError(f"Heat-current CSV not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            present = set(reader.fieldnames or [])
            time_fields = [key for key in ("time_ps", "time_fs") if key in present]
            if len(time_fields) != 1:
                raise AnalysisError("Heat-current CSV needs exactly one time_ps or time_fs column.")
            time_field = time_fields[0]
            required = [time_field] + [f"Q_total_{a}" for a in "xyz"]
            required += [f"Q_{g}_{a}" for g in groups for a in "xyz"]
            missing = set(required) - set(reader.fieldnames or [])
            if missing:
                raise AnalysisError(f"Heat-current CSV lacks columns: {', '.join(sorted(missing))}")
            raw = [{key: float(row[key]) for key in required} for row in reader]
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"Cannot read numeric heat-current CSV {path}: {exc}") from exc
    if len(raw) < 16:
        raise AnalysisError("Heat-current CSV needs at least 16 time samples.")
    time = np.array([row[time_field] for row in raw]) * (0.001 if time_field == "time_fs" else 1.0)
    total = np.array([[row[f"Q_total_{a}"] for a in "xyz"] for row in raw]) * factor
    currents = {g: np.array([[row[f"Q_{g}_{a}"] for a in "xyz"] for row in raw]) * factor
                for g in groups}
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(total)) or any(
        not np.all(np.isfinite(x)) for x in currents.values()
    ):
        raise AnalysisError("Heat-current CSV contains non-finite values.")
    steps = np.diff(time)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-9):
        raise AnalysisError("time_ps must be strictly increasing with uniform spacing.")
    summed = sum(currents.values())
    if not np.allclose(total, summed, rtol=1e-5, atol=1e-7):
        raise AnalysisError("Q_total must equal the sum of all group currents at every time step.")
    return time, currents, total, {"source_time_column": time_field,
                                   "source_heat_current_unit": unit,
                                   "current_to_ev_angstrom_per_ps": factor}


def _cross_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased positive-lag <x(0)y(t)> with FFT and local mean removal."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = len(x)
    fft_size = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(np.conj(np.fft.rfft(x, fft_size)) * np.fft.rfft(y, fft_size), fft_size)
    return corr[:max_lag + 1] / np.arange(n, n - max_lag - 1, -1)


def _integral(corr: np.ndarray, dt_ps: float, temperature_k: float,
              volume_ang3: float) -> np.ndarray:
    """Trapezoidal Green–Kubo integral in W/(m K)."""
    factor = (EV_J * ANG_M / PS_S) ** 2 * PS_S / (KB_J * temperature_k**2 * volume_ang3 * ANG_M**3)
    return np.r_[0.0, np.cumsum((corr[:-1] + corr[1:]) * 0.5 * dt_ps)] * factor


def analyze_gk(config: dict, model: dict) -> Path:
    groups = current_groups(config, model)
    axes = directions(model)
    path = resolve(config, model.get("heat_current_csv"), f"{model['name']}.heat_current_csv")
    unit = model.get("heat_current_unit", config.get("heat_current_unit", "ev_angstrom_per_ps"))
    time, currents, total, unit_info = _read_currents(path, groups, unit)
    dt = float(time[1] - time[0])
    temperature = float(model.get("temperature_k", config.get("temperature_k", 300.0)))
    volume = float(model.get("volume_ang3", 0))
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(volume) or volume <= 0:
        raise AnalysisError(f"{model['name']}: positive temperature_k and volume_ang3 are required.")
    blocks = int(model.get("gk_blocks", config.get("gk_blocks", 4)))
    if blocks < 2:
        raise AnalysisError("gk_blocks must be at least 2 for uncertainty estimation.")
    block_n = len(time) // blocks
    max_lag_ps = float(model.get("max_lag_ps", config.get("max_lag_ps", 5.0)))
    max_lag = int(np.floor(max_lag_ps / dt))
    if max_lag < 1 or max_lag >= block_n // 2:
        raise AnalysisError(f"max_lag_ps must be >= {dt:g} ps and < half a block "
                            f"({block_n * dt / 2:g} ps). Add data or shorten lag.")
    window = model.get("plateau_ps", config.get("plateau_ps"))
    if not isinstance(window, list) or len(window) != 2:
        raise AnalysisError("Set plateau_ps to [start_ps, end_ps] after inspecting the integral curve.")
    start, end = map(float, window)
    if not (0 <= start < end <= max_lag * dt):
        raise AnalysisError("plateau_ps must lie inside [0, max_lag_ps].")
    lags = np.arange(max_lag + 1) * dt
    plateau_mask = (lags >= start) & (lags <= end)
    if plateau_mask.sum() < 2:
        raise AnalysisError("plateau_ps needs at least two sampled lag points.")

    input_hashes = {"heat_current_csv": sha256(path)}
    if model.get("groups_csv"):
        input_hashes["groups_csv"] = sha256(resolve(config, model["groups_csv"], "groups_csv"))
    summary = {"model": model["name"], "analysis_version": __version__,
               "method": "group_heat_current_green_kubo",
               "heat_current_csv": str(path), "temperature_k": temperature,
               **unit_info,
               "input_sha256": input_hashes,
               "heat_current_groups": groups,
               "volume_ang3": volume, "block_count": blocks, "block_samples": block_n,
               "ignored_trailing_samples": len(time) - blocks * block_n,
               "sampling_dt_ps": dt, "plateau_ps": [start, end], "directions": {}}
    rows = []
    for direction, unit in axes.items():
        projected = {g: currents[g] @ unit for g in groups}
        projected_total = total @ unit
        pairs = [(g, h) for g in groups for h in groups]
        labels = [f"{g}__{h}" for g, h in pairs]
        curves = {label: [] for label in labels + ["total"]}
        for ib in range(blocks):
            segment = slice(ib * block_n, (ib + 1) * block_n)
            for (g, h), label in zip(pairs, labels):
                corr = _cross_corr(projected[g][segment], projected[h][segment], max_lag)
                curves[label].append(_integral(corr, dt, temperature, volume))
            corr = _cross_corr(projected_total[segment], projected_total[segment], max_lag)
            curves["total"].append(_integral(corr, dt, temperature, volume))
        means = {label: np.mean(curves[label], axis=0) for label in curves}
        if not np.allclose(sum(means[label] for label in labels), means["total"], rtol=1e-5, atol=1e-8):
            raise AnalysisError(f"{direction}: grouped GK integrals do not reconstruct total.")
        plateau_by_block = {label: np.array([np.mean(curve[plateau_mask]) for curve in curves[label]])
                            for label in curves}
        result = {label: {"mean_w_mk": float(np.mean(value)),
                          "sem_w_mk": float(np.std(value, ddof=1) / np.sqrt(blocks))}
                  for label, value in plateau_by_block.items()}
        result["direction_cart_unit"] = unit.tolist()
        result["group_order"] = groups
        summary["directions"][direction] = result
        for i, lag in enumerate(lags):
            row = {"direction": direction, "lag_ps": float(lag)}
            for label in ["total"] + labels:
                row[f"kappa_{label}_w_mk"] = float(means[label][i])
            rows.append(row)
    out = model_output(config, model, "gk")
    write_csv(out / "integral_curves.csv", list(rows[0]), rows)
    write_json(out / "summary.json", summary)
    return out
