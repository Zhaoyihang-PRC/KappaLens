"""Electronic Boltzmann transport, isolated from phonon/MD estimators.

This is a fresh implementation of the relaxation-time transport integrals.
It accepts explicit state lifetimes or separately labelled scattering rates;
it does not infer a physical rate from a band structure alone. Full tensor
matrix products are performed on the active 2D or 3D subspace.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import __version__
from .common import AnalysisError, directions, model_output, resolve, write_csv, write_json
from .electron_io import ElectronStates, _positive, _poscar_volume, _source, read_legacy_vasp, read_states_csv

CHARGE_C = 1.602176634e-19
KB_J_K = 1.380649e-23
ANG_M = 1e-10


def _temperature(model: dict, config: dict, options: dict) -> float:
    return _positive(options.get("temperature_k", model.get("temperature_k", config.get("temperature_k"))),
                     "electronic.temperature_k")


def _occupancy(states: ElectronStates, mu_ev: float, temperature_k: float) -> np.ndarray:
    x = (states.energy_ev-mu_ev)*CHARGE_C/(KB_J_K*temperature_k)
    return 1.0/(1.0+np.exp(np.clip(x, -700.0, 700.0)))


def _electrons_per_cell(states: ElectronStates, mu_ev: float, temperature_k: float) -> float:
    return float(np.sum(states.k_weight*states.spin_degeneracy*_occupancy(states, mu_ev, temperature_k)))


def _chemical_potentials(states: ElectronStates, options: dict, temperature_k: float,
                         volume_ang3: float, sheet_length_ang: float | None) -> list[float]:
    names = ("chemical_potentials_ev", "extra_electrons_per_cell",
             "extra_carriers_cm3", "extra_carriers_cm2")
    selected = [name for name in names if options.get(name) is not None]
    if len(selected) != 1:
        raise AnalysisError(f"Set exactly one carrier scan: {', '.join(names)}.")
    name = selected[0]
    values = options[name]
    if not isinstance(values, list) or not values:
        raise AnalysisError("Chemical potentials or carrier targets must be a nonempty list.")
    try:
        values = [float(x) for x in values]
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Chemical potentials or carrier targets must be numeric.") from exc
    if not np.all(np.isfinite(values)):
        raise AnalysisError("Chemical potentials or carrier targets must be finite.")
    if name == "chemical_potentials_ev":
        return values
    if name == "extra_carriers_cm3":
        if sheet_length_ang is not None:
            raise AnalysisError("Use extra_carriers_cm2 for a 2D sheet.")
        values = [x*volume_ang3*1e-24 for x in values]
    elif name == "extra_carriers_cm2":
        if sheet_length_ang is None:
            raise AnalysisError("Use extra_carriers_cm3 for a 3D cell.")
        values = [x*(volume_ang3/sheet_length_ang)*1e-16 for x in values]
    if states.reference_electrons_per_cell is None:
        raise AnalysisError("Carrier targets require reference_electrons_per_cell or VASP EIGENVAL NELECT.")
    padding_ev = max(1.0, 100*KB_J_K*temperature_k/CHARGE_C)
    low = float(np.min(states.energy_ev)-padding_ev)
    high = float(np.max(states.energy_ev)+padding_ev)
    low_count = _electrons_per_cell(states, low, temperature_k)
    high_count = _electrons_per_cell(states, high, temperature_k)
    result = []
    for extra in values:
        target = states.reference_electrons_per_cell + extra
        if not low_count < target < high_count:
            raise AnalysisError(f"Carrier target {extra} per cell lies outside the supplied band window.")
        a, b = low, high
        for _ in range(90):
            mid = 0.5*(a+b)
            if _electrons_per_cell(states, mid, temperature_k) < target:
                a = mid
            else:
                b = mid
        result.append(0.5*(a+b))
    return result


def _rates(config: dict, options: dict, states: ElectronStates, temperature_k: float):
    relaxation = options.get("relaxation")
    if not isinstance(relaxation, dict):
        raise AnalysisError("electronic.relaxation must describe one lifetime source.")
    mode = relaxation.get("mode")
    if mode == "constant":
        tau = np.full(len(states.state_id), _positive(relaxation.get("tau_s"), "relaxation.tau_s"))
        return tau, {}, None
    if mode == "state_tau":
        if states.tau_s is None:
            raise AnalysisError("state_tau requires a tau_s column in states_csv.")
        if abs(_positive(relaxation.get("temperature_k"), "relaxation.temperature_k")-temperature_k) > 1e-6:
            raise AnalysisError("State lifetimes and transport temperatures differ.")
        return states.tau_s, {}, None
    if mode != "rates_csv":
        raise AnalysisError("relaxation.mode must be constant, state_tau, or rates_csv.")
    if abs(_positive(relaxation.get("temperature_k"), "relaxation.temperature_k")-temperature_k) > 1e-6:
        raise AnalysisError("Scattering-rate and transport temperatures differ.")
    mechanisms = relaxation.get("mechanisms")
    if (not isinstance(mechanisms, list) or not mechanisms
            or any(not isinstance(x, str) or not x.endswith("_s_inv") for x in mechanisms)
            or len(set(mechanisms)) != len(mechanisms)):
        raise AnalysisError("rates_csv needs unique mechanism columns ending in _s_inv.")
    path = resolve(config, relaxation.get("rates_csv"), "electronic.relaxation.rates_csv")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            needed = {"state_id", *mechanisms}
            if not needed.issubset(reader.fieldnames or []):
                raise AnalysisError(f"{path}: missing columns {sorted(needed-set(reader.fieldnames or []))}")
            rows = list(reader)
    except OSError as exc:
        raise AnalysisError(f"Cannot read scattering rates {path}: {exc}") from exc
    if len(rows) != len(states.state_id):
        raise AnalysisError(f"{path}: one scattering-rate row is required for every electronic state.")
    by_id = {}
    for row in rows:
        key = row["state_id"].strip()
        if not key or key in by_id:
            raise AnalysisError(f"{path}: empty or repeated state_id.")
        by_id[key] = row
    if set(by_id) != set(states.state_id):
        raise AnalysisError(f"{path}: state_id set differs from electronic states.")
    try:
        rates = {name: np.array([float(by_id[s][name]) for s in states.state_id])
                 for name in mechanisms}
    except (ValueError, TypeError) as exc:
        raise AnalysisError(f"{path}: scattering rates must be numeric.") from exc
    for name, rate in rates.items():
        if not np.all(np.isfinite(rate)) or np.any(rate < 0):
            raise AnalysisError(f"{path}: {name} has a negative or nonfinite rate.")
    total = np.sum(list(rates.values()), axis=0)
    if np.any(total <= 0):
        raise AnalysisError(f"{path}: every state needs a positive total scattering rate.")
    return 1.0/total, rates, _source(path)


def _transport_basis(options: dict) -> np.ndarray:
    dim = options.get("dimensionality")
    if dim == "3d":
        return np.eye(3)
    if dim != "2d":
        raise AnalysisError("electronic.dimensionality must be explicitly 2d or 3d.")
    normal = np.asarray(options.get("sheet_normal_cart"), dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
        raise AnalysisError("2d transport needs a nonzero sheet_normal_cart vector.")
    normal /= np.linalg.norm(normal)
    seed = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.column_stack((first, second))


def _solve_tensors(states: ElectronStates, tau: np.ndarray, temperature_k: float,
                   mu_ev: float, volume_ang3: float, basis: np.ndarray):
    occupation = _occupancy(states, mu_ev, temperature_k)
    derivative = occupation*(1-occupation)/(KB_J_K*temperature_k)
    excess_j = (states.energy_ev-mu_ev)*CHARGE_C
    velocity = states.velocity_m_s @ basis
    kernel = states.k_weight*states.spin_degeneracy*tau*derivative/(volume_ang3*ANG_M**3)
    moments = [np.einsum("n,ni,nj->ij", kernel*excess_j**order,
                         velocity, velocity, optimize=True) for order in range(3)]
    zero, first, second = moments
    eigenvalues = np.linalg.eigvalsh(zero)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= eigenvalues[-1]*1e-12:
        raise AnalysisError("Electronic conductivity is singular or ill-conditioned in the active subspace; "
                            "check bands, Fermi level, velocities and transport dimension.")
    solve_first = np.linalg.solve(zero, first)
    sigma = CHARGE_C**2*zero
    seebeck = -solve_first/(CHARGE_C*temperature_k)
    kappa_e = (second-first@solve_first)/temperature_k
    # Symmetrize numerical roundoff in kappa_e; no diagonal-only shortcut.
    kappa_e = 0.5*(kappa_e+kappa_e.T)
    cart = lambda tensor: basis @ tensor @ basis.T
    return cart(sigma), cart(seebeck), cart(kappa_e), kernel, velocity


def _characters(config: dict, options: dict, states: ElectronStates):
    if not options.get("state_character_csv"):
        return {}, None
    path = resolve(config, options["state_character_csv"], "electronic.state_character_csv")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            groups = [name for name in (reader.fieldnames or []) if name != "state_id"]
            rows = list(reader)
    except OSError as exc:
        raise AnalysisError(f"Cannot read electronic state character {path}: {exc}") from exc
    if not groups or "state_id" not in (reader.fieldnames or []):
        raise AnalysisError(f"{path}: need state_id and at least one group-weight column.")
    if len(rows) != len(states.state_id):
        raise AnalysisError(f"{path}: one character row is required for every state.")
    by_id = {row["state_id"].strip(): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(states.state_id):
        raise AnalysisError(f"{path}: state_id values must match electronic states exactly.")
    try:
        weights = {group: np.array([float(by_id[s][group]) for s in states.state_id])
                   for group in groups}
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{path}: character weights must be numeric.") from exc
    values = np.stack(list(weights.values()))
    if (not np.all(np.isfinite(values)) or np.any(values < -1e-8)
            or np.any(values > 1+1e-8) or not np.allclose(values.sum(axis=0), 1, atol=1e-6)):
        raise AnalysisError(f"{path}: each state's nonnegative character weights must sum to 1.")
    return weights, _source(path)


def _disorder(config: dict, options: dict):
    if not options.get("transfer_fluctuations_csv"):
        return None, None
    path = resolve(config, options["transfer_fluctuations_csv"],
                   "electronic.transfer_fluctuations_csv")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"time_ps", "pair", "transfer_ev"}.issubset(reader.fieldnames or []):
                raise AnalysisError(f"{path}: need time_ps,pair,transfer_ev.")
            rows = list(reader)
    except OSError as exc:
        raise AnalysisError(f"Cannot read transfer fluctuations {path}: {exc}") from exc
    grouped = {}
    try:
        for row in rows:
            grouped.setdefault(row["pair"].strip(), []).append((float(row["time_ps"]),
                                                                   float(row["transfer_ev"])))
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{path}: invalid fluctuation value.") from exc
    if not grouped or "" in grouped:
        raise AnalysisError(f"{path}: pair labels must be nonempty.")
    diagnostics = {}
    for pair, samples in grouped.items():
        if len(samples) < 8:
            raise AnalysisError(f"{path}: pair {pair} needs at least 8 samples.")
        array = np.asarray(samples)
        t, coupling = array[:, 0], array[:, 1]
        if not np.all(np.isfinite(array)) or np.any(np.diff(t) <= 0):
            raise AnalysisError(f"{path}: pair {pair} needs finite, increasing time and coupling.")
        dt = np.diff(t)
        if not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-10):
            raise AnalysisError(f"{path}: pair {pair} needs a uniform time step.")
        mean, std = float(coupling.mean()), float(coupling.std(ddof=1))
        diagnostics[pair] = {"mean_transfer_ev": mean, "std_transfer_ev": std,
                             "relative_fluctuation": std/abs(mean) if abs(mean) > 1e-15 else None,
                             "sample_count": len(samples), "time_step_ps": float(dt[0])}
    return diagnostics, _source(path)


def analyze_electrons(config: dict, model: dict) -> Path:
    options = model.get("electronic")
    if not isinstance(options, dict):
        raise AnalysisError(f"{model['name']}: set an electronic section in the project config.")
    if options.get("source") == "states_csv":
        states = read_states_csv(config, model, options)
    elif options.get("source") == "vasp_legacy":
        states = read_legacy_vasp(config, model, options)
    else:
        raise AnalysisError("electronic.source must be states_csv or vasp_legacy.")
    temperature_k = _temperature(model, config, options)
    volume = states.cell_volume_ang3
    if options.get("cell_volume_ang3") is not None:
        manual = _positive(options["cell_volume_ang3"], "electronic.cell_volume_ang3")
        if volume is not None and not np.isclose(manual, volume, rtol=1e-5):
            raise AnalysisError("Configured cell volume differs from POSCAR.")
        volume = manual
    if volume is None and options.get("poscar"):
        poscar = resolve(config, options["poscar"], "electronic.poscar")
        volume = _poscar_volume(poscar)
        states.sources["poscar"] = _source(poscar)
    if volume is None:
        raise AnalysisError("Electronic transport needs cell_volume_ang3 or POSCAR.")
    states.reference_electrons_per_cell = options.get("reference_electrons_per_cell",
                                                      states.reference_electrons_per_cell)
    if states.reference_electrons_per_cell is not None:
        states.reference_electrons_per_cell = float(states.reference_electrons_per_cell)
        if not np.isfinite(states.reference_electrons_per_cell):
            raise AnalysisError("reference_electrons_per_cell must be finite.")
    basis = _transport_basis(options)
    sheet_length = None
    if options["dimensionality"] == "2d":
        sheet_length = _positive(options.get("sheet_repeat_length_ang"),
                                 "electronic.sheet_repeat_length_ang")
    direction_model = dict(model)
    if options.get("directions_cart"):
        direction_model["directions_cart"] = options["directions_cart"]
    vectors = directions(direction_model)
    for name, vector in vectors.items():
        if options["dimensionality"] == "2d" and np.linalg.norm(basis@basis.T@vector-vector) > 1e-6:
            raise AnalysisError(f"Electronic direction {name} is not in the declared 2D sheet plane.")
    tau, mechanisms, rates_source = _rates(config, options, states, temperature_k)
    characters, character_source = _characters(config, options, states)
    disorder, disorder_source = _disorder(config, options)
    if rates_source:
        states.sources["rates_csv"] = rates_source
    if character_source:
        states.sources["state_character_csv"] = character_source
    if disorder_source:
        states.sources["transfer_fluctuations_csv"] = disorder_source
    mus = _chemical_potentials(states, options, temperature_k, volume, sheet_length)
    rows, mechanism_rows, character_rows = [], [], []
    summary = {"model": model["name"], "analysis_version": __version__,
               "method": "independent-band Boltzmann RTA, full tensor",
               "source": options["source"], "dimensionality": options["dimensionality"],
               "temperature_k": temperature_k, "cell_volume_ang3": volume,
               "sheet_repeat_length_ang": sheet_length,
               "spin_degeneracy": states.spin_degeneracy,
               "reference_electrons_per_cell": states.reference_electrons_per_cell,
               "state_count": len(states.state_id), "relaxation_mode": options["relaxation"]["mode"],
               "input_sources": states.sources, "results": [],
               "interpretation": {
                   "state_character": "Transport-kernel-weighted orbital character, not unique group-owned conductivity.",
                   "mechanism_fraction": "Rate fraction weighted by conductivity integrand; not additive conductivity.",
                   "disorder": "Transfer-integral fluctuation metrics only; not a hopping/localization model."}}
    if disorder is not None:
        summary["transfer_fluctuations"] = disorder
    for mu_ev in mus:
        sigma, seebeck, kappa_e, kernel, velocities = _solve_tensors(
            states, tau, temperature_k, mu_ev, volume, basis)
        electron_count = _electrons_per_cell(states, mu_ev, temperature_k)
        extra = (electron_count-states.reference_electrons_per_cell
                 if states.reference_electrons_per_cell is not None else None)
        point = {"chemical_potential_ev": mu_ev, "electrons_per_cell": electron_count,
                 "extra_electrons_per_cell": extra,
                 "sigma_s_m_tensor": sigma.tolist(), "seebeck_v_k_tensor": seebeck.tolist(),
                 "kappa_e_w_mk_tensor": kappa_e.tolist(), "directions": {}}
        if extra is not None:
            point["extra_carriers_cm3"] = extra/(volume*1e-24)
            if sheet_length is not None:
                point["extra_carriers_cm2"] = extra/(volume/sheet_length*1e-16)
        for name, vector in vectors.items():
            sig = float(vector @ sigma @ vector)
            s = float(vector @ seebeck @ vector)
            ke = float(vector @ kappa_e @ vector)
            record = {"model": model["name"], "temperature_k": temperature_k,
                      "chemical_potential_ev": mu_ev, "direction": name,
                      "direction_cart_unit": vector.tolist(), "sigma_s_m": sig,
                      "seebeck_v_k": s, "kappa_e_w_mk": ke,
                      "power_factor_w_mk2": s*s*sig}
            if sheet_length is not None:
                record["sheet_conductance_s"] = sig*sheet_length*ANG_M
                record["sheet_kappa_e_w_k"] = ke*sheet_length*ANG_M
                record["sheet_power_factor_w_k2"] = record["power_factor_w_mk2"]*sheet_length*ANG_M
            point["directions"][name] = record
            rows.append(record)
            transport_weight = (kernel * (velocities @ (basis.T@vector))**2)
            weight_sum = float(np.sum(transport_weight))
            if weight_sum <= 0:
                raise AnalysisError(f"No electronic transport weight along direction {name}.")
            if mechanisms:
                total_rate = np.sum(list(mechanisms.values()), axis=0)
                for mechanism, rate in mechanisms.items():
                    mechanism_rows.append({"model": model["name"], "chemical_potential_ev": mu_ev,
                                           "direction": name, "mechanism": mechanism,
                                           "transport_weighted_rate_fraction": float(
                                               np.sum(transport_weight*rate/total_rate)/weight_sum)})
            if characters:
                projection_sum = 0.0
                for group, weights in characters.items():
                    projected = float(CHARGE_C**2*np.sum(transport_weight*weights))
                    projection_sum += projected
                    character_rows.append({"model": model["name"], "chemical_potential_ev": mu_ev,
                                           "direction": name, "group": group,
                                           "sigma_character_s_m": projected})
                if not np.isclose(projection_sum, sig, rtol=1e-6, atol=1e-12):
                    raise AnalysisError("Electronic character projections do not reconstruct conductivity.")
        summary["results"].append(point)
    out = model_output(config, model, "electrons")
    write_json(out / "summary.json", summary)
    write_csv(out / "transport.csv", list(rows[0]), rows)
    if mechanism_rows:
        write_csv(out / "mechanism_fractions.csv", list(mechanism_rows[0]), mechanism_rows)
    if character_rows:
        write_csv(out / "state_character_projection.csv", list(character_rows[0]), character_rows)
    return out


def export_electron_states(config: dict, model: dict) -> Path:
    """Expose a stable state ID before generating external scattering rates.

    This does not compute a lifetime or transport coefficient. The converter
    checks that legacy VASP symmetries cover the declared full regular mesh.
    """
    options = model.get("electronic")
    if not isinstance(options, dict) or options.get("source") != "vasp_legacy":
        raise AnalysisError("electron-export requires electronic.source=vasp_legacy.")
    states = read_legacy_vasp(config, model, options)
    out = model_output(config, model, "electron_states")
    columns = ["state_id", "k_index", "band_index", "k_x", "k_y", "k_z", "spin", "energy_ev",
               "v_x_m_s", "v_y_m_s", "v_z_m_s", "k_weight"]
    with (out / "expanded_states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, sid in enumerate(states.state_id):
            k_label, band_label, _ = sid.split("_")
            writer.writerow([sid, int(k_label[1:]), int(band_label[1:]),
                             *states.k_fractional[i], int(states.spin[i]),
                             states.energy_ev[i], *states.velocity_m_s[i], states.k_weight[i]])
    write_json(out / "summary.json", {"model": model["name"], "analysis_version": __version__,
                                      "method": "legacy VASP state export, no transport calculation",
                                      "state_count": len(states.state_id),
                                      "spin_degeneracy": states.spin_degeneracy,
                                      "cell_volume_ang3": states.cell_volume_ang3,
                                      "reference_electrons_per_cell": states.reference_electrons_per_cell,
                                      "input_sources": states.sources})
    return out
