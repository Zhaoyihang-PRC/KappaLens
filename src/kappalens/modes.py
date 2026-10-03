"""Project Phono3py's *intraband mode conductivity* onto chemical atom groups.

This is a mode-character attribution, not a microscopic group heat-current
measurement. In Phono3py's HDF5 layout, mode_kappa is already summed over the
q-star and must be divided by the total number of grid points. Consequently,
group weights from its representative q are valid only when chemical groups
are invariant under the q-star's crystal symmetries. We fail closed otherwise.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from . import __version__
from .common import (AnalysisError, directions, load_groups, model_output,
                     resolve, sha256, voigt_projection, write_json)


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise AnalysisError("h5py is required for 'modes'. Install it in an isolated environment.") from exc
    return h5py


def _matched_grid_points(kq: np.ndarray, pq: np.ndarray, tolerance: float) -> list[int]:
    """Match reduced q coordinates modulo reciprocal lattice translations."""
    if kq.ndim != 2 or kq.shape[1] != 3 or pq.ndim != 2 or pq.shape[1] != 3:
        raise AnalysisError("qpoint and grid_address must both have three coordinates.")
    used = set()
    indices = []
    for iq, q in enumerate(kq):
        delta = q[None, :] - pq
        delta -= np.rint(delta)
        norms = np.max(np.abs(delta), axis=1)
        candidates = [int(i) for i in np.where(norms < tolerance)[0] if int(i) not in used]
        if len(candidates) != 1:
            raise AnalysisError(f"q-point {iq} maps to {len(candidates)} unused phonon points; "
                                "check mesh, primitive cell and BZ boundary aliases.")
        indices.append(candidates[0])
        used.add(candidates[0])
    return indices


def _group_weights(eigenvectors: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Eigenvectors are columns; each row triplet is one primitive atom."""
    ev = np.asarray(eigenvectors)
    if not np.iscomplexobj(ev) and ev.ndim == 3 and ev.shape[-1] == 2:
        ev = ev[..., 0] + 1j * ev[..., 1]
    natoms = sum(len(ids) for ids in groups.values())
    if ev.shape != (3 * natoms, 3 * natoms):
        raise AnalysisError(f"Eigenvector shape {ev.shape} does not match {natoms} mapped atoms. "
                            "Check primitive-cell atom order and count.")
    squared = np.abs(ev.reshape(natoms, 3, 3 * natoms)) ** 2
    norm = squared.sum(axis=(0, 1))
    if np.any(norm <= 0):
        raise AnalysisError("An eigenvector has zero norm.")
    result = {name: squared[ids].sum(axis=(0, 1)) / norm for name, ids in groups.items()}
    if not np.allclose(sum(result.values()), 1.0, atol=1e-8):
        raise AnalysisError("Group weights do not sum to 1; check the atom map.")
    return result


def _average_degenerate(weights: dict[str, np.ndarray], frequencies: np.ndarray,
                        tolerance: float) -> None:
    """Remove arbitrary eigenvector mixing inside numerically degenerate bands."""
    first = 0
    for i in range(1, len(frequencies) + 1):
        if i < len(frequencies) and abs(frequencies[i] - frequencies[i-1]) <= tolerance:
            continue
        if i - first > 1:
            for array in weights.values():
                array[first:i] = float(array[first:i].mean())
        first = i


def analyze_modes(config: dict, model: dict) -> Path:
    h5py = _h5py()
    groups = load_groups(config, model)
    axes = directions(model)
    if model.get("primitive_order_matches_atom_map") is not True:
        raise AnalysisError(f"{model['name']}: confirm primitive atom order in config with "
                            "'primitive_order_matches_atom_map': true. Count alone is insufficient.")
    kp = resolve(config, model.get("kappa_hdf5"), f"{model['name']}.kappa_hdf5")
    pp = resolve(config, model.get("phonon_hdf5"), f"{model['name']}.phonon_hdf5")
    for p in [kp, pp]:
        if not p.is_file():
            raise AnalysisError(f"Required HDF5 file not found: {p}")

    target_t = float(model.get("temperature_k", config.get("temperature_k", 300.0)))
    qtol = float(config.get("q_tolerance", 1e-6))
    ftol = float(config.get("frequency_tolerance_thz", 0.02))
    dtol = float(config.get("degeneracy_tolerance_thz", 1e-4))
    out = model_output(config, model, "modes")
    # A large primitive cell can produce millions of rows. Stream to a temporary
    # file, then publish it only after all consistency checks have passed.
    fields = ["model", "temperature_k", "direction", "q_index", "band_index_1based",
              "q_x", "q_y", "q_z", "q_star_weight", "frequency_thz", "gamma_thz",
              "mode_kappa_w_mk"]
    for group in groups:
        fields += [f"weight_{group}", f"projected_{group}_w_mk"]
    stream = tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=out,
                                         prefix=".mode_projection-", suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    success = False
    summary: dict = {"model": model["name"], "analysis_version": __version__,
                     "method": "intraband_mode_character_projection",
                     "kappa_hdf5": str(kp), "phonon_hdf5": str(pp),
                     "input_sha256": {"kappa_hdf5": sha256(kp), "phonon_hdf5": sha256(pp),
                                      "groups_csv": sha256(resolve(config, model["groups_csv"], "groups_csv"))},
                     "groups": {k: len(v) for k, v in groups.items()}, "directions": {}}

    try:
        with h5py.File(kp, "r") as kh, h5py.File(pp, "r") as ph:
            required_k = ["mode_kappa", "kappa", "weight", "qpoint", "frequency", "temperature", "mesh"]
            required_p = ["eigenvector", "ir_grid_points", "grid_address", "mesh", "frequency"]
            missing = [x for x in required_k if x not in kh] + [x for x in required_p if x not in ph]
            if missing:
                raise AnalysisError(f"Missing HDF5 datasets: {', '.join(missing)}")
            mesh = np.asarray(kh["mesh"][:], dtype=int)
            pmesh = np.asarray(ph["mesh"][:], dtype=int)
            if mesh.shape != (3,) or np.any(mesh <= 0) or not np.array_equal(mesh, pmesh):
                raise AnalysisError("This reader requires matching diagonal 3-number meshes. "
                                    "Generalized regular grids need a separate mapping adapter.")
            weights_q = np.asarray(kh["weight"][:], dtype=int)
            if weights_q.ndim != 1 or np.any(weights_q <= 0) or weights_q.sum() != int(np.prod(mesh)):
                raise AnalysisError("Irreducible q weights do not cover the full mesh; do not mix partial grid files.")
            if np.any(weights_q > 1) and model.get("group_star_invariant") is not True:
                raise AnalysisError("mode_kappa contains q-star sums. Confirm all mapped atom groups "
                                    "are symmetry invariant and set 'group_star_invariant': true, "
                                    "or generate a full-grid output.")
            temps = np.asarray(kh["temperature"][:], dtype=float)
            if temps.size == 0 or not np.all(np.isfinite(temps)):
                raise AnalysisError("Temperature dataset is empty or invalid.")
            tidx = int(np.argmin(np.abs(temps - target_t)))
            if abs(temps[tidx] - target_t) > float(config.get("temperature_tolerance_k", 0.5)):
                raise AnalysisError(f"Requested {target_t} K not found; available values include "
                                    f"{temps[max(0,tidx-2):tidx+3].tolist()}")
            summary["temperature_k"] = float(temps[tidx])
            mk = np.asarray(kh["mode_kappa"][tidx], dtype=float)
            total_tensor = np.asarray(kh["kappa"][tidx], dtype=float)
            freq = np.asarray(kh["frequency"][:], dtype=float)
            kq = np.asarray(kh["qpoint"][:], dtype=float)
            n_q, n_band = freq.shape
            if n_q == 0 or not np.all(np.isfinite(freq)) or not np.all(np.isfinite(mk)):
                raise AnalysisError("Phonon frequency or mode_kappa data are empty or non-finite.")
            if mk.shape != (n_q, n_band, 6) or total_tensor.shape != (6,):
                raise AnalysisError("Unexpected mode_kappa/kappa shape; check the Phono3py version and calculation type.")
            if n_band != 3 * sum(len(v) for v in groups.values()):
                raise AnalysisError("Band count differs from 3 × mapped primitive atoms. Check primitive mapping.")
            if np.min(freq) < -float(config.get("imaginary_tolerance_thz", 0.1)) and not model.get("allow_imaginary_modes", False):
                raise AnalysisError("Significant imaginary frequencies found; inspect structural stability first.")
            reconstructed = mk.sum(axis=(0, 1)) / weights_q.sum()
            if not np.allclose(reconstructed, total_tensor, rtol=2e-3, atol=0.005):
                raise AnalysisError("mode_kappa does not reconstruct kappa. This may be a Wigner/interband "
                                    "or incomplete-grid output; do not project it as the full conductivity. "
                                    f"Maximum difference: {np.max(np.abs(reconstructed-total_tensor)):.6g} W/mK")
            ir = np.asarray(ph["ir_grid_points"][:], dtype=int)
            if len(ir) != n_q:
                raise AnalysisError("Phonon irreducible q count differs from kappa file.")
            pq = np.asarray(ph["grid_address"][ir], dtype=float) / mesh
            matching = _matched_grid_points(kq, pq, qtol)
            if "ir_grid_weights" in ph:
                pw = np.asarray(ph["ir_grid_weights"][:], dtype=int)
                if len(pw) == n_q and not np.array_equal(pw[matching], weights_q):
                    raise AnalysisError("q-star weights differ between phonon and kappa files.")
            group_arrays = {name: np.zeros((n_q, n_band), dtype=float) for name in groups}
            for iq, ip in enumerate(matching):
                eig = np.asarray(ph["eigenvector"][ir[ip]])
                pfreq = np.asarray(ph["frequency"][ir[ip]], dtype=float)
                if pfreq.shape != (n_band,) or np.max(np.abs(pfreq - freq[iq])) > ftol:
                    raise AnalysisError(f"Phonon frequencies/branch order do not match at q-point {iq}.")
                gw = _group_weights(eig, groups)
                _average_degenerate(gw, freq[iq], dtol)
                for group, arr in gw.items():
                    group_arrays[group][iq] = arr
            if not np.allclose(sum(group_arrays.values()), 1.0, atol=1e-8):
                raise AnalysisError("Projected group weights failed the sum check.")

            gamma = np.asarray(kh["gamma"][tidx], dtype=float) if "gamma" in kh else None
            if gamma is not None and gamma.shape != (n_q, n_band):
                raise AnalysisError("Unexpected gamma shape.")
            for direction, unit in axes.items():
                mode_values = voigt_projection(mk, unit) / weights_q.sum()
                expected = float(voigt_projection(total_tensor, unit))
                group_totals = {g: float(np.sum(w * mode_values)) for g, w in group_arrays.items()}
                if not np.isclose(sum(group_totals.values()), expected, rtol=2e-3, atol=0.005):
                    raise AnalysisError(f"{direction}: group projection does not sum to original kappa.")
                summary["directions"][direction] = {"direction_cart_unit": unit.tolist(),
                                                     "kappa_intraband_w_mk": expected,
                                                     "projected_w_mk": group_totals}
                for iq in range(n_q):
                    for band in range(n_band):
                        row = {"model": model["name"], "temperature_k": float(temps[tidx]),
                               "direction": direction, "q_index": iq, "band_index_1based": band + 1,
                               "q_x": float(kq[iq, 0]), "q_y": float(kq[iq, 1]), "q_z": float(kq[iq, 2]),
                               "q_star_weight": int(weights_q[iq]), "frequency_thz": float(freq[iq, band]),
                               "gamma_thz": float(gamma[iq, band]) if gamma is not None else "",
                               "mode_kappa_w_mk": float(mode_values[iq, band])}
                        for group, w in group_arrays.items():
                            row[f"weight_{group}"] = float(w[iq, band])
                            row[f"projected_{group}_w_mk"] = float(w[iq, band] * mode_values[iq, band])
                        writer.writerow(row)
        success = True
    except OSError as exc:
        raise AnalysisError(f"Cannot read HDF5 input: {exc}") from exc
    finally:
        stream.close()
        # A successful run replaces the temporary file below. On any failure,
        # this removes incomplete data and preserves an older valid output.
        if not success:
            temporary.unlink(missing_ok=True)
    temporary.replace(out / "mode_projection.csv")
    write_json(out / "summary.json", summary)
    return out
