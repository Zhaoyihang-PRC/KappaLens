"""Compare models while keeping Phono3py and MD estimators separate."""

from __future__ import annotations

import json
from pathlib import Path

from .common import AnalysisError, resolve, write_csv, write_json


def compare_models(config: dict, stage: str) -> Path:
    if stage == "electrons":
        return compare_electrons(config)
    if stage not in {"modes", "gk"}:
        raise AnalysisError("Comparison stage must be modes, gk or electrons.")
    root = resolve(config, config.get("output_dir", "analysis_results"), "output_dir")
    summaries = {}
    for model in config["models"]:
        path = root / model["name"] / stage / "summary.json"
        if not path.is_file():
            raise AnalysisError(f"Run '{stage}' for {model['name']} first; missing {path}")
        try:
            summaries[model["name"]] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"Cannot read {path}: {exc}") from exc
    reference = next(iter(summaries.values()))
    for name, summary in summaries.items():
        if abs(summary["temperature_k"] - reference["temperature_k"]) > 0.5:
            raise AnalysisError(f"{name}: temperature differs; comparison would be invalid.")
        if set(summary["directions"]) != set(reference["directions"]):
            raise AnalysisError(f"{name}: direction labels differ.")
        for direction, result in summary["directions"].items():
            ref_unit = reference["directions"][direction].get("direction_cart_unit")
            if ref_unit is None or result.get("direction_cart_unit") != ref_unit:
                raise AnalysisError(f"{name}: Cartesian vector for '{direction}' differs.")
    rows = []
    for name, summary in summaries.items():
        for direction, result in summary["directions"].items():
            if stage == "modes":
                total = result["kappa_intraband_w_mk"]
                components = result["projected_w_mk"]
            else:
                total = result["total"]["mean_w_mk"]
                components = {k: v["mean_w_mk"] for k, v in result.items()
                              if isinstance(v, dict) and "mean_w_mk" in v and k != "total"}
            rows.append({"model": name, "direction": direction, "component": "total", "kappa_w_mk": total})
            for component, value in components.items():
                rows.append({"model": name, "direction": direction,
                             "component": component, "kappa_w_mk": value})
    comparisons = []
    for item in config.get("comparisons", []):
        left, right = item.get("left"), item.get("right")
        if left not in summaries or right not in summaries:
            raise AnalysisError(f"Comparison references an unknown model: {item}")
        for direction in reference["directions"]:
            left_rows = {r["component"]: r["kappa_w_mk"] for r in rows
                         if r["model"] == left and r["direction"] == direction}
            right_rows = {r["component"]: r["kappa_w_mk"] for r in rows
                          if r["model"] == right and r["direction"] == direction}
            if set(left_rows) != set(right_rows):
                raise AnalysisError(f"{left}/{right}: different group components; align group definitions.")
            for component in left_rows:
                comparisons.append({"name": item.get("name", f"{left}_vs_{right}"),
                                    "direction": direction, "component": component,
                                    "left": left, "right": right,
                                    "left_w_mk": left_rows[component],
                                    "right_w_mk": right_rows[component],
                                    "right_minus_left_w_mk": right_rows[component] - left_rows[component]})
    out = root / "comparisons" / stage
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "model_components.csv", list(rows[0]), rows)
    if comparisons:
        write_csv(out / "contrasts.csv", list(comparisons[0]), comparisons)
    write_json(out / "summary.json", {"stage": stage, "temperature_k": reference["temperature_k"],
                                      "models": list(summaries), "comparison_count": len(comparisons)})
    return out


def compare_electrons(config: dict) -> Path:
    """Compare like-for-like electronic scans; never align unrelated mu by row number."""
    basis = config.get("electronic_comparison_basis")
    if basis not in {"chemical_potential_ev", "extra_carriers_cm3", "extra_carriers_cm2"}:
        raise AnalysisError("Set electronic_comparison_basis to chemical_potential_ev, "
                            "extra_carriers_cm3 or extra_carriers_cm2.")
    root = resolve(config, config.get("output_dir", "analysis_results"), "output_dir")
    summaries = {}
    for model in config["models"]:
        path = root / model["name"] / "electrons" / "summary.json"
        if not path.is_file():
            raise AnalysisError(f"Run electrons for {model['name']} first; missing {path}")
        try:
            summaries[model["name"]] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"Cannot read {path}: {exc}") from exc
    first = next(iter(summaries.values()))
    dimensionality = first["dimensionality"]
    if basis == "extra_carriers_cm2" and dimensionality != "2d":
        raise AnalysisError("2D carrier density comparison requires 2D electronic results.")
    if basis == "extra_carriers_cm3" and dimensionality != "3d":
        raise AnalysisError("3D carrier density comparison requires 3D electronic results.")
    reference_points = first["results"]
    reference_directions = reference_points[0]["directions"]
    rows = []
    for model_name, summary in summaries.items():
        if summary["dimensionality"] != dimensionality:
            raise AnalysisError(f"{model_name}: electronic dimensionality differs.")
        if abs(summary["temperature_k"]-first["temperature_k"]) > 0.5:
            raise AnalysisError(f"{model_name}: electronic temperature differs.")
        if len(summary["results"]) != len(reference_points):
            raise AnalysisError(f"{model_name}: scan lengths differ.")
        for index, (point, ref_point) in enumerate(zip(summary["results"], reference_points)):
            current = point.get(basis)
            expected = ref_point.get(basis)
            if current is None or expected is None or abs(current-expected) > max(1e-7, abs(expected)*1e-6):
                raise AnalysisError(f"{model_name}: carrier scan point {index+1} differs for {basis}.")
            if set(point["directions"]) != set(reference_directions):
                raise AnalysisError(f"{model_name}: electronic direction labels differ.")
            for direction, record in point["directions"].items():
                if record["direction_cart_unit"] != reference_directions[direction]["direction_cart_unit"]:
                    raise AnalysisError(f"{model_name}: vector for direction {direction} differs.")
                if dimensionality == "2d":
                    fields = ("sheet_conductance_s", "seebeck_v_k", "sheet_kappa_e_w_k",
                              "sheet_power_factor_w_k2")
                else:
                    fields = ("sigma_s_m", "seebeck_v_k", "kappa_e_w_mk", "power_factor_w_mk2")
                rows.append({"model": model_name, "scan_index": index+1, "scan_basis": basis,
                             "scan_value": current, "direction": direction,
                             **{field: record[field] for field in fields}})
    contrasts = []
    for item in config.get("comparisons", []):
        left, right = item.get("left"), item.get("right")
        if left not in summaries or right not in summaries:
            raise AnalysisError(f"Comparison references an unknown model: {item}")
        for left_row in [row for row in rows if row["model"] == left]:
            matching = [row for row in rows if row["model"] == right
                        and row["scan_index"] == left_row["scan_index"]
                        and row["direction"] == left_row["direction"]]
            right_row = matching[0]
            for field in left_row:
                if field in {"model", "scan_index", "scan_basis", "scan_value", "direction"}:
                    continue
                contrasts.append({"name": item.get("name", f"{left}_vs_{right}"),
                                  "left": left, "right": right, "scan_index": left_row["scan_index"],
                                  "scan_basis": basis, "scan_value": left_row["scan_value"],
                                  "direction": left_row["direction"], "property": field,
                                  "left_value": left_row[field], "right_value": right_row[field],
                                  "right_minus_left": right_row[field]-left_row[field]})
    out = root / "comparisons" / "electrons"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "model_transport.csv", list(rows[0]), rows)
    if contrasts:
        write_csv(out / "contrasts.csv", list(contrasts[0]), contrasts)
    write_json(out / "summary.json", {"stage": "electrons", "scan_basis": basis,
                                      "dimensionality": dimensionality,
                                      "temperature_k": first["temperature_k"],
                                      "models": list(summaries), "comparison_count": len(contrasts)})
    return out
