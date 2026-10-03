"""Optional electronic/lattice joining with explicit normalization checks."""

from __future__ import annotations

import json

import numpy as np

from . import __version__
from .common import AnalysisError, model_output, resolve, write_csv, write_json


def _summary(config: dict, model: dict, stage: str) -> dict:
    root = resolve(config, config.get("output_dir", "analysis_results"), "output_dir")
    path = root / model["name"] / stage / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Run '{stage}' first; cannot read {path}: {exc}") from exc


def analyze_thermoelectric(config: dict, model: dict):
    options = model.get("thermoelectric")
    if not isinstance(options, dict):
        raise AnalysisError(f"{model['name']}: set a thermoelectric section.")
    stage = options.get("lattice_stage")
    if stage not in ("modes", "gk"):
        raise AnalysisError("thermoelectric.lattice_stage must be modes or gk.")
    if options.get("same_cell_normalization_verified") is not True:
        raise AnalysisError("Set same_cell_normalization_verified=true after checking both source cells and units.")
    electron, lattice = _summary(config, model, "electrons"), _summary(config, model, stage)
    if abs(float(electron["temperature_k"])-float(lattice["temperature_k"])) > 0.5:
        raise AnalysisError("Electronic and lattice temperatures differ.")
    dim = electron["dimensionality"]
    if options.get("lattice_dimensionality") != dim:
        raise AnalysisError("Lattice dimensionality must explicitly match electronic dimensionality.")
    if dim == "2d":
        sheet = float(electron["sheet_repeat_length_ang"])
        if not np.isclose(float(options.get("lattice_sheet_repeat_length_ang", 0)), sheet, rtol=1e-6):
            raise AnalysisError("Electronic and lattice 2D repeat lengths differ.")
    else:
        sheet = None
    if stage == "gk" and not np.isclose(float(lattice["volume_ang3"]),
                                         float(electron["cell_volume_ang3"]), rtol=1e-5):
        raise AnalysisError("Electronic and Green-Kubo cell volumes differ.")
    rows = []
    for point in electron["results"]:
        for name, electronic_direction in point["directions"].items():
            if name not in lattice["directions"]:
                raise AnalysisError(f"Lattice result lacks electronic direction {name}.")
            lattice_direction = lattice["directions"][name]
            if not np.allclose(electronic_direction["direction_cart_unit"],
                               lattice_direction["direction_cart_unit"], atol=1e-8):
                raise AnalysisError(f"Direction vector {name} differs between calculations.")
            if stage == "modes":
                kl = float(lattice_direction["kappa_intraband_w_mk"])
                kl_sem = None
            else:
                kl = float(lattice_direction["total"]["mean_w_mk"])
                kl_sem = float(lattice_direction["total"]["sem_w_mk"])
            ke = float(electronic_direction["kappa_e_w_mk"])
            total = ke+kl
            if not np.isfinite(total) or total <= 0:
                raise AnalysisError(f"{name}: total thermal conductivity is not positive.")
            pf = float(electronic_direction["power_factor_w_mk2"])
            zt = pf*float(electron["temperature_k"])/total
            record = {"model": model["name"], "lattice_stage": stage,
                      "lattice_scope": "intraband_only" if stage == "modes" else "green_kubo_total",
                      "temperature_k": electron["temperature_k"],
                      "chemical_potential_ev": point["chemical_potential_ev"], "direction": name,
                      "sigma_s_m": electronic_direction["sigma_s_m"],
                      "seebeck_v_k": electronic_direction["seebeck_v_k"],
                      "power_factor_w_mk2": pf, "kappa_e_w_mk": ke,
                      "kappa_l_w_mk": kl, "kappa_l_sem_w_mk": kl_sem,
                      "zt_or_intraband_proxy": zt,
                      "zt_lattice_sem_only": abs(zt/total*kl_sem) if kl_sem is not None else None}
            if sheet is not None:
                repeat_m = sheet*1e-10
                record.update({"sheet_repeat_length_ang": sheet,
                               "sheet_conductance_s": electronic_direction["sheet_conductance_s"],
                               "sheet_kappa_e_w_k": ke*repeat_m,
                               "sheet_kappa_l_w_k": kl*repeat_m})
            rows.append(record)
    out = model_output(config, model, "thermoelectric")
    write_csv(out / "combined.csv", list(rows[0]), rows)
    write_json(out / "summary.json", {"model": model["name"], "analysis_version": __version__,
                                      "lattice_stage": stage, "dimensionality": dim,
                                      "temperature_k": electron["temperature_k"],
                                      "sheet_repeat_length_ang": sheet,
                                      "same_cell_normalization_verified": True,
                                      "result_count": len(rows),
                                      "warning": "modes gives an intraband-only ZT proxy; "
                                                 "GK uncertainty excludes electronic-model errors."})
    return out
