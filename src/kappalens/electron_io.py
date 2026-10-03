"""Independent, explicit-unit readers for electronic band transport.

The legacy VASP adapter reads data formats, not TransOpt code or its output.
It expands a complete regular k mesh only when symmetry and coverage checks
pass. The CSV path is calculator-neutral and avoids any symmetry assumptions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import AnalysisError, resolve, sha256


@dataclass
class ElectronStates:
    state_id: list[str]
    energy_ev: np.ndarray
    velocity_m_s: np.ndarray
    k_weight: np.ndarray
    spin: np.ndarray
    spin_degeneracy: int
    tau_s: np.ndarray | None
    cell_volume_ang3: float | None
    reference_electrons_per_cell: float | None
    sources: dict[str, dict[str, str]]
    k_fractional: np.ndarray | None = None


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def _positive(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be a positive number.") from exc
    if not np.isfinite(number) or number <= 0:
        raise AnalysisError(f"{label} must be a positive finite number.")
    return number


def _spin_degeneracy(options: dict, spins: np.ndarray) -> int:
    value = options.get("spin_degeneracy")
    if value not in (1, 2):
        raise AnalysisError("electronic.spin_degeneracy must be explicitly set to 1 or 2.")
    if len(set(spins.tolist())) > 1 and value != 1:
        raise AnalysisError("Explicit spin channels require spin_degeneracy=1.")
    return value


def read_states_csv(config: dict, model: dict, options: dict) -> ElectronStates:
    if options.get("full_brillouin_zone") is not True:
        raise AnalysisError("states_csv needs full_brillouin_zone=true after verifying the table "
                            "covers the full k mesh. Irreducible k weights alone are insufficient "
                            "for a general tensor.")
    path = resolve(config, options.get("states_csv"), "electronic.states_csv")
    required = {"state_id", "k_index", "band_index", "spin", "energy_ev",
                "v_x_m_s", "v_y_m_s", "v_z_m_s", "k_weight"}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                raise AnalysisError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
            rows = list(reader)
    except OSError as exc:
        raise AnalysisError(f"Cannot read electronic states: {path}: {exc}") from exc
    if not rows:
        raise AnalysisError(f"Electronic states file is empty: {path}")
    ids = [r["state_id"].strip() for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise AnalysisError(f"{path}: state_id values must be nonempty and unique.")
    try:
        energy = np.array([float(r["energy_ev"]) for r in rows])
        velocity = np.array([[float(r[f"v_{axis}_m_s"]) for axis in "xyz"] for r in rows])
        weight = np.array([float(r["k_weight"]) for r in rows])
        spin = np.array([int(r["spin"]) for r in rows])
        k_indices = [r["k_index"].strip() for r in rows]
        bands = [int(r["band_index"]) for r in rows]
    except (ValueError, TypeError, KeyError) as exc:
        raise AnalysisError(f"{path}: invalid electronic state value: {exc}") from exc
    if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(velocity))
            and np.all(np.isfinite(weight)) and np.all(weight > 0)):
        raise AnalysisError(f"{path}: energies, velocities and positive weights must be finite.")
    if any(not k for k in k_indices) or any(b < 1 for b in bands) or np.any(spin < 0):
        raise AnalysisError(f"{path}: invalid k_index, band_index or spin.")
    channels = set(spin.tolist())
    if channels not in ({0}, {0, 1}):
        raise AnalysisError(f"{path}: spin channels must be 0, or 0 and 1.")
    degeneracy = _spin_degeneracy(options, spin)
    k_weights: dict[tuple[int, str], float] = {}
    state_keys = set()
    bands_by_k: dict[tuple[int, str], set[int]] = {}
    for i, row in enumerate(rows):
        key = (int(spin[i]), k_indices[i])
        previous = k_weights.setdefault(key, float(weight[i]))
        if not np.isclose(previous, weight[i], rtol=0, atol=1e-10):
            raise AnalysisError(f"{path}: inconsistent k_weight for spin/k_index {key}.")
        state_key = (*key, bands[i])
        if state_key in state_keys:
            raise AnalysisError(f"{path}: repeated spin/k_index/band_index {state_key}.")
        state_keys.add(state_key)
        bands_by_k.setdefault(key, set()).add(bands[i])
    for channel in channels:
        total = sum(w for (s, _), w in k_weights.items() if s == channel)
        if not np.isclose(total, 1.0, rtol=0, atol=1e-6):
            raise AnalysisError(f"{path}: k weights for spin {channel} sum to {total}, expected 1.")
    if channels == {0, 1}:
        k0 = {k for (s, k) in k_weights if s == 0}
        k1 = {k for (s, k) in k_weights if s == 1}
        if k0 != k1:
            raise AnalysisError(f"{path}: spin channels use different k-point IDs.")
    reference_bands = next(iter(bands_by_k.values()))
    if any(band_set != reference_bands for band_set in bands_by_k.values()):
        raise AnalysisError(f"{path}: every k point and spin channel must contain the same bands.")
    tau = None
    if "tau_s" in (reader.fieldnames or []):
        try:
            tau = np.array([float(r["tau_s"]) for r in rows])
        except (ValueError, TypeError) as exc:
            raise AnalysisError(f"{path}: invalid tau_s value.") from exc
        if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            raise AnalysisError(f"{path}: tau_s must be positive and finite.")
    k_fractional = None
    if {"k_x", "k_y", "k_z"}.issubset(reader.fieldnames or []):
        try:
            k_fractional = np.array([[float(row[f"k_{axis}"]) for axis in "xyz"] for row in rows])
        except (ValueError, TypeError) as exc:
            raise AnalysisError(f"{path}: invalid fractional k coordinate.") from exc
        if not np.all(np.isfinite(k_fractional)):
            raise AnalysisError(f"{path}: fractional k coordinates must be finite.")
    return ElectronStates(ids, energy, velocity, weight, spin, degeneracy, tau,
                          None, None, {"states_csv": _source(path)}, k_fractional)


def _poscar_volume(path: Path) -> float:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        scale = float(lines[1].split()[0])
        lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    except (OSError, ValueError, IndexError) as exc:
        raise AnalysisError(f"Cannot read POSCAR lattice: {path}: {exc}") from exc
    raw = abs(float(np.linalg.det(lattice)))
    if not np.isfinite(raw) or raw <= 0 or scale == 0:
        raise AnalysisError(f"{path}: invalid cell volume or POSCAR scale.")
    return raw * scale**3 if scale > 0 else -scale


def _float_tokens(line: str, minimum: int, label: str) -> list[float]:
    try:
        values = [float(x) for x in line.split()]
    except ValueError as exc:
        raise AnalysisError(f"{label}: expected numeric values.") from exc
    if len(values) < minimum or not np.all(np.isfinite(values)):
        raise AnalysisError(f"{label}: expected at least {minimum} finite numbers.")
    return values


def _next_content(handle, label: str) -> str:
    for line in handle:
        if line.strip():
            return line
    raise AnalysisError(f"Unexpected end of {label}.")


def _read_legacy_eigenval(path: Path):
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            header = [next(handle) for _ in range(6)]
            nspin = int(header[0].split()[-1])
            nelect, n_k, n_band = map(int, header[5].split()[:3])
            if nspin not in (1, 2) or n_k <= 0 or n_band <= 0:
                raise ValueError("invalid EIGENVAL dimensions")
            kpoints = np.empty((n_k, 3))
            weights = np.empty(n_k)
            energies = np.empty((n_k, n_band, nspin))
            for ik in range(n_k):
                values = _float_tokens(_next_content(handle, str(path)), 4, str(path))
                kpoints[ik], weights[ik] = values[:3], values[3]
                for ib in range(n_band):
                    band = _float_tokens(_next_content(handle, str(path)), 1+nspin, str(path))
                    if int(band[0]) != ib + 1:
                        raise AnalysisError(f"{path}: unexpected band order at k point {ik+1}.")
                    energies[ik, ib] = band[1:1+nspin]
    except (OSError, StopIteration, ValueError, IndexError) as exc:
        raise AnalysisError(f"Cannot parse EIGENVAL {path}: {exc}") from exc
    return nelect, kpoints, weights, energies


def _read_legacy_groupvec(path: Path, kpoints: np.ndarray, n_band: int, nspin: int):
    velocity = np.empty((len(kpoints), n_band, nspin, 3))
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for ik, k in enumerate(kpoints):
                position = _float_tokens(_next_content(handle, str(path)), 3, str(path))[:3]
                if not np.allclose(np.mod(position-k+0.5, 1)-0.5, 0, atol=2e-5):
                    raise AnalysisError(f"{path}: k point {ik+1} differs from EIGENVAL.")
                for ib in range(n_band):
                    values = _float_tokens(_next_content(handle, str(path)), 1+3*nspin, str(path))
                    if int(values[0]) != ib + 1:
                        raise AnalysisError(f"{path}: unexpected band order at k point {ik+1}.")
                    for direction in range(3):
                        velocity[ik, ib, :, direction] = values[1+direction*nspin:1+(direction+1)*nspin]
            if any(line.strip() for line in handle):
                raise AnalysisError(f"{path}: extra GROUPVEC records after expected bands.")
    except OSError as exc:
        raise AnalysisError(f"Cannot read GROUPVEC {path}: {exc}") from exc
    return velocity


def _mesh(path: Path) -> tuple[int, int, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        dims = tuple(int(x) for x in lines[3].split()[:3])
    except (OSError, ValueError, IndexError) as exc:
        raise AnalysisError(f"Cannot read regular KPOINTS mesh {path}: {exc}") from exc
    if len(dims) != 3 or any(x < 1 for x in dims):
        raise AnalysisError(f"{path}: regular k mesh needs three positive dimensions.")
    return dims


def _symmetry(path: Path) -> np.ndarray:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        count = int(lines[0].split()[0])
        if count < 1 or len(lines) < 1 + 3*count:
            raise ValueError("incomplete symmetry matrices")
        matrices = np.array([[_float_tokens(lines[1+3*i+j], 3, str(path))[:3]
                              for j in range(3)] for i in range(count)])
    except (OSError, ValueError, IndexError) as exc:
        raise AnalysisError(f"Cannot read SYMMETRY {path}: {exc}") from exc
    if not np.allclose(matrices, np.rint(matrices), atol=1e-7):
        raise AnalysisError(f"{path}: symmetry matrices must be integral in reciprocal coordinates.")
    return matrices


def _key(kpoint: np.ndarray) -> tuple[int, int, int]:
    """Periodic fractional k key with 1e-6 resolution; coverage is checked."""
    return tuple((np.rint(np.mod(kpoint, 1.0)*1_000_000).astype(int) % 1_000_000).tolist())


def _check_grid(keys: list[tuple[int, int, int]], mesh: tuple[int, int, int]) -> None:
    for axis, n in enumerate(mesh):
        values = np.array(sorted({key[axis] for key in keys}), dtype=float)
        if len(values) != n:
            raise AnalysisError(f"Expanded k mesh has {len(values)} values on axis {axis}; expected {n}.")
        if n > 1:
            spacing = np.diff(np.r_[values, values[0]+1_000_000])
            if not np.allclose(spacing, 1_000_000/n, atol=2):
                raise AnalysisError(f"Expanded k mesh is not regular on axis {axis}.")


def read_legacy_vasp(config: dict, model: dict, options: dict) -> ElectronStates:
    """Read EIGENVAL/GROUPVEC without accepting a silently partial k mesh.

    GROUPVEC values are converted only via an explicit scale. For dE/dk in
    eV*Angstrom, use velocity_unit='ev_angstrom'; for already Cartesian m/s,
    use 'm_per_s'. Other VASP patches need a documented custom scale.
    """
    paths = {key: resolve(config, options.get(key), f"electronic.{key}")
             for key in ("eigenval", "groupvec", "kpoints", "symmetry", "poscar")}
    nelect, ir_k, _, ir_e = _read_legacy_eigenval(paths["eigenval"])
    n_ir, n_band, nspin = ir_e.shape
    ir_v = _read_legacy_groupvec(paths["groupvec"], ir_k, n_band, nspin)
    mesh = _mesh(paths["kpoints"])
    matrices = _symmetry(paths["symmetry"])
    volume = _poscar_volume(paths["poscar"])
    if nspin == 2 and options.get("time_reversal", False):
        raise AnalysisError("Time reversal is not supported for explicit spin-polarized channels.")
    if not isinstance(options.get("time_reversal"), bool):
        raise AnalysisError("electronic.time_reversal must be explicitly true or false.")
    unit = options.get("velocity_unit")
    if unit == "m_per_s":
        factor = 1.0
    elif unit == "ev_angstrom":
        factor = 1.602176634e-19*1e-10/1.054571817e-34
    elif unit == "custom":
        factor = _positive(options.get("velocity_scale_to_m_s"), "velocity_scale_to_m_s")
    else:
        raise AnalysisError("velocity_unit must be m_per_s, ev_angstrom, or custom with explicit scale.")
    lattice = np.array([_float_tokens(line, 3, str(paths["poscar"]))[:3]
                        for line in paths["poscar"].read_text().splitlines()[2:5]])
    scale = float(paths["poscar"].read_text().splitlines()[1].split()[0])
    lattice *= scale if scale > 0 else (volume/abs(np.linalg.det(lattice)))**(1/3)
    reciprocal = np.linalg.inv(lattice).T
    mapping: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {}
    for matrix in matrices:
        # Raw SYMMETRY rows act on a fractional row vector from the right.
        # k_row = k_fractional_row @ reciprocal. For k' = k @ matrix,
        # the Cartesian column-vector rotation is the transpose of
        # reciprocal^{-1} @ matrix @ reciprocal. This remains orthogonal
        # for a skew cell; the superficially similar reversed product does not.
        cart_rotation = (np.linalg.inv(reciprocal) @ matrix @ reciprocal).T
        if not np.allclose(cart_rotation.T @ cart_rotation, np.eye(3), atol=2e-5):
            raise AnalysisError("SYMMETRY is incompatible with the POSCAR Cartesian metric.")
        for sign in ((1.0, -1.0) if options["time_reversal"] else (1.0,)):
            for ik, k in enumerate(ir_k):
                key = _key(sign * (k @ matrix))
                mapping.setdefault(key, (ik, sign*cart_rotation))
    if len(mapping) != int(np.prod(mesh)):
        raise AnalysisError(f"Symmetry expands to {len(mapping)} k points; KPOINTS requires {np.prod(mesh)}. "
                            "Check symmetry, time reversal, magnetic state and mesh.")
    _check_grid(list(mapping), mesh)
    keys = sorted(mapping)
    spins = np.arange(nspin, dtype=int)
    degeneracy = _spin_degeneracy(options, spins)
    per_k = n_band*nspin
    n_state = len(keys)*per_k
    ids: list[str] = []
    energies = np.empty(n_state)
    velocities = np.empty((n_state, 3))
    weights = np.full(n_state, 1.0/len(keys))
    spin_values = np.empty(n_state, dtype=int)
    k_fractional = np.empty((n_state, 3))
    for ik_full, key in enumerate(keys):
        ik_ir, rotation = mapping[key]
        start = ik_full*per_k
        stop = start+per_k
        energies[start:stop] = ir_e[ik_ir].reshape(-1)
        velocities[start:stop] = ir_v[ik_ir].reshape(-1, 3) @ rotation.T * factor
        spin_values[start:stop] = np.tile(np.arange(nspin), n_band)
        k_fractional[start:stop] = np.asarray(key, dtype=float)/1_000_000
        for ib in range(n_band):
            for spin in range(nspin):
                ids.append(f"k{ik_full:06d}_b{ib+1:04d}_s{spin}")
    return ElectronStates(ids, energies, velocities, weights, spin_values, degeneracy,
                          None, volume, float(nelect),
                          {key: _source(path) for key, path in paths.items()},
                          k_fractional)
