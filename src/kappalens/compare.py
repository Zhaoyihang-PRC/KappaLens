"""Compare models while keeping Phono3py and MD estimators separate."""

from __future__ import annotations

import json
from pathlib import Path

from .common import AnalysisError, resolve, write_csv, write_json


def compare_models(config: dict, stage: str) -> Path:
    if stage not in {"modes", "gk"}:
        raise AnalysisError("Comparison stage must be 'modes' or 'gk'.")
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
