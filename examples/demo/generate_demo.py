#!/usr/bin/env python3
"""Generate *entirely synthetic* KappaLens inputs for two toy systems.

This script uses no research structure, trajectory, force constant, or paper
data. HDF5 generation is optional so Green-Kubo and CSV spectral demos can run
with NumPy/SciPy alone; use --require-hdf5 to fail if h5py is unavailable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _colored_current(rng: np.random.Generator, n: int, persistence: float) -> np.ndarray:
    """Simple stationary AR(1) signal for code demonstration only."""
    noise = rng.normal(size=(n, 3))
    current = np.zeros_like(noise)
    for i in range(1, n):
        current[i] = persistence * current[i - 1] + noise[i]
    return current


def _dho(omega: np.ndarray, amplitude: float, omega0: float,
         gamma: float, baseline: float) -> np.ndarray:
    return baseline + amplitude * gamma * omega**2 / ((omega**2 - omega0**2)**2 + (gamma * omega)**2)


def generate(name: str, scale: float, seed: int, h5py_module) -> None:
    out = ROOT / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "atom_map.csv").write_text("atom_index_1based,element\n1,C\n2,H\n", encoding="utf-8")
    (out / "groups.csv").write_text(
        "atom_index_1based,element,group\n1,C,framework\n2,H,pendant\n", encoding="utf-8")

    rng = np.random.default_rng(seed)
    n, dt_ps = 2048, 0.01
    framework = _colored_current(rng, n, 0.82) * scale
    pendant = 0.25 * framework + _colored_current(rng, n, 0.72) * (0.7 + 0.1 * scale)
    columns = ["time_ps"] + [f"Q_total_{a}" for a in "xyz"]
    columns += [f"Q_{g}_{a}" for g in ("framework", "pendant") for a in "xyz"]
    with (out / "heat_current.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n):
            writer.writerow([i * dt_ps, *(framework[i] + pendant[i]),
                             *framework[i], *pendant[i]])

    omega = np.linspace(0.2, 3.8, 181)
    framework_s = _dho(omega, 1.4, 1.35 + 0.05 * scale, 0.22, 0.03)
    pendant_s = _dho(omega, 0.8, 2.1 + 0.03 * scale, 0.34, 0.02)
    with (out / "spectrum.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["q_index", "q_x", "q_y", "q_z", "omega",
                         "Clqw_framework_framework", "Clqw_pendant_pendant"])
        for x, a, b in zip(omega, framework_s, pendant_s):
            writer.writerow([0, 0, 0, 0, x, a, b])

    # Calculator-neutral electronic fixtures. Rates and orbital characters are
    # deliberately arbitrary; they demonstrate file matching and tensor math.
    k_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    with (out / "electron_states.csv").open("w", newline="", encoding="utf-8") as f_state, \
         (out / "electron_rates.csv").open("w", newline="", encoding="utf-8") as f_rate, \
         (out / "electron_character.csv").open("w", newline="", encoding="utf-8") as f_character:
        state_writer = csv.writer(f_state)
        rate_writer = csv.writer(f_rate)
        char_writer = csv.writer(f_character)
        state_writer.writerow(["state_id", "k_index", "band_index", "spin", "energy_ev",
                               "v_x_m_s", "v_y_m_s", "v_z_m_s", "k_weight"])
        rate_writer.writerow(["state_id", "acoustic_s_inv", "optical_s_inv", "impurity_s_inv"])
        char_writer.writerow(["state_id", "framework", "pendant"])
        for ik, vector in enumerate(k_vectors):
            for band in (1, 2):
                sid = f"k{ik:06d}_b{band:04d}_s0"
                energy = (-0.08 if band == 1 else 0.10) + 0.015*ik
                velocity = vector*scale*(1.0 if band == 1 else 0.7)*1e5
                state_writer.writerow([sid, ik, band, 0, energy, *velocity, 0.25])
                rate_writer.writerow([sid, 1.1e13+ik*1e12, 0.5e13+band*0.3e13, 0.2e13])
                character = 0.75 if band == 1 else 0.45
                char_writer.writerow([sid, character, 1-character])
    with (out / "transfer_fluctuations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ps", "pair", "transfer_ev"])
        for i in range(32):
            writer.writerow([i*0.05, "framework-pendant", 0.08+0.012*np.sin(i*0.8)])

    if h5py_module is not None:
        frequencies = np.array([[0.6, 0.9, 1.2, 1.6, 2.1, 2.6]])
        mode_kappa = np.zeros((1, 1, 6, 6))
        for band in range(6):
            mode_kappa[0, 0, band, 0] = scale * (0.08 + band * 0.01)
            mode_kappa[0, 0, band, 1] = scale * (0.05 + band * 0.008)
            mode_kappa[0, 0, band, 2] = scale * (0.03 + band * 0.005)
        with h5py_module.File(out / "kappa-demo.hdf5", "w") as f:
            f["mesh"] = [1, 1, 1]
            f["weight"] = [1]
            f["temperature"] = [300.0]
            f["frequency"] = frequencies
            f["qpoint"] = np.zeros((1, 3))
            f["mode_kappa"] = mode_kappa
            f["kappa"] = mode_kappa.sum(axis=(1, 2))
        with h5py_module.File(out / "phonon-demo.hdf5", "w") as f:
            f["mesh"] = [1, 1, 1]
            f["ir_grid_points"] = [0]
            f["ir_grid_weights"] = [1]
            f["grid_address"] = np.zeros((1, 3), dtype=int)
            f["frequency"] = frequencies
            f["eigenvector"] = np.eye(6, dtype=complex)[None, :, :]
    print(f"Generated synthetic input: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-hdf5", action="store_true",
                        help="fail unless h5py is available and HDF5 fixtures can be generated")
    args = parser.parse_args()
    try:
        import h5py
    except ImportError:
        if args.require_hdf5:
            parser.error("h5py is required for --require-hdf5; install kappalens[demo]")
        h5py = None
        print("h5py unavailable: generated CSV inputs only; skip the modes command.")
    generate("demo_a", 1.0, 24, h5py)
    generate("demo_b", 1.25, 25, h5py)


if __name__ == "__main__":
    main()
