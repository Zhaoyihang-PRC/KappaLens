"""Dynasor NPZ or explicit CSV spectrum extraction and single-peak fitting.

A fitted linewidth describes a selected spectral feature. It is not by itself
an atomic group's thermal conductivity or a unique phonon lifetime.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import __version__
from .common import AnalysisError, model_output, resolve, sha256, write_csv, write_json


def _dho(omega: np.ndarray, amplitude: float, omega0: float, gamma: float,
         baseline: float) -> np.ndarray:
    """Damped oscillator spectral line, with omega and gamma in the same units."""
    return baseline + amplitude * gamma * omega**2 / ((omega**2 - omega0**2)**2 + (gamma * omega)**2)


def fit_peak(omega: np.ndarray, signal: np.ndarray, low: float, high: float) -> dict:
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise AnalysisError("scipy is required for spectral fitting.") from exc
    x = np.asarray(omega, dtype=float)
    y = np.asarray(signal, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= low) & (x <= high)
    x, y = x[mask], y[mask]
    if len(x) < 8 or not np.all(np.diff(x) > 0) or low < 0:
        raise AnalysisError("DSF fit window needs >=8 ascending nonnegative frequency points.")
    if np.any(y < -1e-10):
        raise AnalysisError("DSF fit expects a nonnegative auto spectrum; cross spectra need separate analysis.")
    baseline = max(0.0, float(np.quantile(y, 0.1)))
    peak = float(x[np.argmax(y)])
    width = max(float(np.median(np.diff(x))) * 2, (high - low) / 20)
    guess = [max(float(np.max(y) - baseline) * width * peak**2, 1e-15), peak, width, baseline]
    bounds = ([0, low, 1e-12, 0], [np.inf, high, high - low, np.inf])
    try:
        params, _ = curve_fit(_dho, x, y, p0=guess, bounds=bounds, maxfev=30000)
    except (ValueError, RuntimeError) as exc:
        raise AnalysisError(f"DHO fit failed: {exc}") from exc
    fit = _dho(x, *params)
    ss_total = float(np.sum((y - y.mean())**2))
    r2 = 1 - float(np.sum((y - fit)**2)) / ss_total if ss_total > 0 else 0.0
    return {"omega0": float(params[1]), "gamma": float(params[2]),
            "amplitude": float(params[0]), "baseline": float(params[3]),
            "r_squared": r2, "n_points": len(x)}


class _CSVSample:
    """Small adapter matching the Dynasor sample access used below."""

    def __init__(self, omega: np.ndarray, q_points: np.ndarray, spectra: dict[str, np.ndarray]):
        self.omega = omega
        self.q_points = q_points
        self._spectra = spectra

    def __getitem__(self, field: str) -> np.ndarray:
        return self._spectra[field]


def _read_spectra_csv(path: Path) -> _CSVSample:
    """Read a rectangular q/omega table; reject inconsistent grids or q labels."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fixed = {"q_index", "q_x", "q_y", "q_z", "omega"}
            fields = [x for x in (reader.fieldnames or []) if x not in fixed]
            if not fixed.issubset(reader.fieldnames or []) or not fields:
                raise AnalysisError(f"{path}: need q_index,q_x,q_y,q_z,omega and one spectrum column.")
            grouped: dict[int, list] = {}
            for row in reader:
                iq = int(row["q_index"])
                grouped.setdefault(iq, []).append((float(row["omega"]),
                    np.array([float(row[k]) for k in ("q_x", "q_y", "q_z")]),
                    [float(row[k]) for k in fields]))
    except (OSError, ValueError, KeyError) as exc:
        raise AnalysisError(f"Cannot read spectrum CSV {path}: {exc}") from exc
    if not grouped or sorted(grouped) != list(range(len(grouped))):
        raise AnalysisError("Spectrum CSV q_index values must be contiguous from 0.")
    q_points, spectra = [], {field: [] for field in fields}
    reference_omega = None
    for iq in sorted(grouped):
        records = sorted(grouped[iq], key=lambda x: x[0])
        omega = np.array([row[0] for row in records])
        coords = np.array([row[1] for row in records])
        if not np.all(np.isfinite(omega)) or not np.all(np.diff(omega) > 0):
            raise AnalysisError(f"Spectrum CSV q_index {iq} has invalid or duplicate frequencies.")
        if not np.all(np.isfinite(coords)) or not np.allclose(coords, coords[0], atol=1e-8):
            raise AnalysisError(f"Spectrum CSV q_index {iq} has inconsistent q coordinates.")
        if reference_omega is None:
            reference_omega = omega
        elif omega.shape != reference_omega.shape or not np.allclose(omega, reference_omega, atol=1e-9):
            raise AnalysisError("Every q point in spectrum CSV must use the same omega grid.")
        q_points.append(coords[0])
        values = np.array([row[2] for row in records])
        if not np.all(np.isfinite(values)):
            raise AnalysisError(f"Spectrum CSV q_index {iq} has non-finite values.")
        for j, field in enumerate(fields):
            spectra[field].append(values[:, j])
    return _CSVSample(reference_omega, np.array(q_points),
                      {field: np.array(values) for field, values in spectra.items()})


def analyze_dsf(config: dict, model: dict) -> Path:
    sources = [key for key in ("dsf_npz", "dsf_csv") if model.get(key)]
    if len(sources) != 1:
        raise AnalysisError(f"{model['name']}: set exactly one of dsf_npz or dsf_csv.")
    source = sources[0]
    path = resolve(config, model[source], f"{model['name']}.{source}")
    if not path.is_file():
        raise AnalysisError(f"Spectrum input not found: {path}")
    windows = model.get("dsf_fit_windows", [])
    if not isinstance(windows, list) or not windows:
        raise AnalysisError(f"{model['name']}: add one or more dsf_fit_windows.")
    try:
        if source == "dsf_npz":
            try:
                from dynasor import read_sample_from_npz
            except ImportError as exc:
                raise AnalysisError("dynasor is required for dsf_npz input.") from exc
            sample = read_sample_from_npz(str(path))
        else:
            sample = _read_spectra_csv(path)
        omega = np.asarray(sample.omega, dtype=float)
        qpoints = np.asarray(sample.q_points, dtype=float)
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Cannot read spectrum input {path}: {exc}") from exc
    if omega.ndim != 1 or qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise AnalysisError("Unexpected Dynasor omega/q_points shape.")
    unit = model.get("dsf_omega_unit")
    if not unit or not isinstance(unit, str):
        raise AnalysisError("Set dsf_omega_unit explicitly (for example rad/fs or THz).")
    rows = []
    for spec in windows:
        try:
            field = str(spec["field"])
            iq = int(spec["q_index"])
            low, high = float(spec["omega_min"]), float(spec["omega_max"])
            if not 0 <= iq < len(qpoints) or not low < high:
                raise ValueError("invalid q index or frequency bounds")
            raw = np.asarray(sample[field])
            if np.iscomplexobj(raw):
                if np.max(np.abs(raw.imag)) > 1e-8 * max(1.0, float(np.max(np.abs(raw.real)))):
                    raise AnalysisError(f"{field}: complex cross spectrum cannot use this real DHO fit.")
                raw = raw.real
            spectrum = np.asarray(raw, dtype=float)
            if spectrum.shape != (len(qpoints), len(omega)):
                raise AnalysisError(f"{field}: expected shape (n_q,n_omega), got {spectrum.shape}")
            fit = fit_peak(omega, spectrum[iq], low, high)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise AnalysisError(f"Invalid DSF fit specification {spec}: {exc}") from exc
        rows.append({"model": model["name"], "field": field, "q_index": iq,
                     "q_x": float(qpoints[iq, 0]), "q_y": float(qpoints[iq, 1]),
                     "q_z": float(qpoints[iq, 2]), "omega_unit": unit,
                     "omega_min": low, "omega_max": high, **fit})
    out = model_output(config, model, "dsf")
    write_csv(out / "fitted_peaks.csv", list(rows[0]), rows)
    write_json(out / "summary.json", {"model": model["name"], "analysis_version": __version__,
                                      "method": "dho_peak_fit", "source_type": source,
                                      "source_path": str(path), "omega_unit": unit,
                                      "input_sha256": {source: sha256(path)},
                                      "note": "Gamma is a DHO fit parameter; do not interpret as conductivity.",
                                      "fits": rows})
    return out
