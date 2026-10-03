"""Shared input, atom-order, direction, and output checks.

Mapped atom indices on disk are one-based, matching the user's reference cell.
Arrays inside Python use zero-based indices. We require explicit group
assignment: an element symbol cannot identify a structural component by itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


class AnalysisError(Exception):
    """Input or scientific consistency failure with a user-readable message."""


def load_config(filename: str | Path) -> dict:
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise AnalysisError(f"Config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), list) or not data["models"]:
        raise AnalysisError("Config needs a nonempty 'models' list.")
    names = [m.get("name") for m in data["models"] if isinstance(m, dict)]
    if len(names) != len(data["models"]) or any(not isinstance(x, str) or not x for x in names):
        raise AnalysisError("Every model needs a nonempty string 'name'.")
    if len(set(names)) != len(names):
        raise AnalysisError("Model names must be unique.")
    if any(x in {".", ".."} or not all(c.isalnum() or c in "_-." for c in x)
           for x in names):
        raise AnalysisError("Model names may use only letters, numbers, _, -, or .")
    data["_base"] = path.parent
    data["_path"] = path
    return data


def resolve(config: dict, value: str | None, label: str) -> Path:
    if not value or not isinstance(value, str):
        raise AnalysisError(f"Missing path setting: {label}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config["_base"] / path).resolve()


def model_output(config: dict, model: dict, stage: str) -> Path:
    root = resolve(config, config.get("output_dir", "analysis_results"), "output_dir")
    out = root / model["name"] / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_models(config: dict, name: str | None) -> list[dict]:
    models = config["models"]
    chosen = [m for m in models if name is None or m["name"] == name]
    if not chosen:
        raise AnalysisError(f"Unknown model: {name}")
    return chosen


def _group_label(value: str) -> str:
    """One spelling for group labels in CSV headers and result keys."""
    label = value.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", label):
        raise AnalysisError(f"Group name must start with a letter and use only a-z, 0-9, _: {value}")
    return label


def _index_column(fieldnames: list[str] | None, path: Path) -> str:
    """Accept the original VASP column and a calculator-neutral equivalent."""
    for name in ("atom_index_1based", "poscar_index_1based"):
        if name in (fieldnames or []):
            return name
    raise AnalysisError(f"{path}: need atom_index_1based or poscar_index_1based column.")


def load_groups(config: dict, model: dict) -> dict[str, np.ndarray]:
    """Check every reference/primitive-cell atom has exactly one mapped group."""
    atom_path = resolve(config, model.get("atom_map"), f"{model['name']}.atom_map")
    group_path = resolve(config, model.get("groups_csv"), f"{model['name']}.groups_csv")
    if not atom_path.is_file():
        raise AnalysisError(f"Atom map not found: {atom_path}")
    if not group_path.is_file():
        raise AnalysisError(f"Group map not found: {group_path}. Run 'init-groups'.")
    with atom_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        atom_key = _index_column(reader.fieldnames, atom_path)
        atom_rows = list(reader)
    if not atom_rows:
        raise AnalysisError(f"Empty atom map: {atom_path}")
    elements: dict[int, str] = {}
    for row in atom_rows:
        try:
            index = int(row[atom_key])
            element = row["element"].strip()
        except (KeyError, ValueError) as exc:
            raise AnalysisError(f"Invalid atom map row in {atom_path}: {row}") from exc
        if index in elements:
            raise AnalysisError(f"Repeated atom {index} in {atom_path}")
        elements[index] = element
    if set(elements) != set(range(1, len(elements) + 1)):
        raise AnalysisError(f"Atom indices must be contiguous from 1 in {atom_path}")

    assigned: dict[int, str] = {}
    with group_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        group_key = _index_column(reader.fieldnames, group_path)
        for row in reader:
            try:
                index = int(row[group_key])
                element = row["element"].strip()
                group = _group_label(row["group"])
            except (KeyError, ValueError) as exc:
                raise AnalysisError(f"Invalid group map row in {group_path}: {row}") from exc
            if index in assigned:
                raise AnalysisError(f"Repeated atom {index} in {group_path}")
            if index not in elements or elements[index] != element:
                raise AnalysisError(f"Atom {index} element/order differs between maps: {group_path}")
            if group == "unassigned":
                raise AnalysisError(f"Atom {index} still has no group in {group_path}")
            assigned[index] = group
    if set(assigned) != set(elements):
        missing = sorted(set(elements) - set(assigned))
        raise AnalysisError(f"Group map must contain every atom once; missing {missing[:12]}")
    return {name: np.array([i - 1 for i, g in assigned.items() if g == name], dtype=int)
            for name in sorted(set(assigned.values()))}


def current_groups(config: dict, model: dict) -> list[str]:
    """Get heat-current CSV groups; allow an MD-only system without phonon atom maps.

    For a project with both atom maps and an explicit current group list, the
    names must agree. We can verify CSV heat-current additivity, but we cannot
    infer which trajectory atoms LAMMPS assigned to a group from this CSV.
    """
    explicit = model.get("heat_current_groups")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit or not all(isinstance(x, str) for x in explicit):
            raise AnalysisError(f"{model['name']}: heat_current_groups must be a nonempty list of names.")
        names = [_group_label(x) for x in explicit]
        if len(set(names)) != len(names):
            raise AnalysisError(f"{model['name']}: heat_current_groups contains duplicates.")
        if model.get("atom_map") or model.get("groups_csv"):
            mapped = set(load_groups(config, model))
            if set(names) != mapped:
                raise AnalysisError(f"{model['name']}: heat_current_groups differs from mapped groups.")
        return sorted(names)
    return sorted(load_groups(config, model))


def init_groups(config: dict, model: dict, overwrite: bool = False) -> Path:
    """Create a deliberately unassigned template; never guess chemical groups."""
    atom_path = resolve(config, model.get("atom_map"), f"{model['name']}.atom_map")
    group_path = resolve(config, model.get("groups_csv"), f"{model['name']}.groups_csv")
    if not atom_path.is_file():
        raise AnalysisError(f"Atom map not found: {atom_path}")
    if group_path.exists() and not overwrite:
        raise AnalysisError(f"Group map already exists: {group_path}")
    with atom_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        atom_key = _index_column(reader.fieldnames, atom_path)
        if "element" not in (reader.fieldnames or []):
            raise AnalysisError(f"{atom_path}: need element column.")
        rows = list(reader)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    with group_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[atom_key, "element", "group"])
        writer.writeheader()
        for row in rows:
            writer.writerow({atom_key: row[atom_key],
                             "element": row["element"], "group": "UNASSIGNED"})
    return group_path


def directions(model: dict) -> dict[str, np.ndarray]:
    raw = model.get("directions_cart")
    if not isinstance(raw, dict) or not raw:
        raise AnalysisError(f"{model['name']}: set 'directions_cart' in config.")
    result = {}
    for name, vector in raw.items():
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.linalg.norm(arr) == 0:
            raise AnalysisError(f"{model['name']}: invalid direction '{name}'")
        result[name] = arr / np.linalg.norm(arr)
    return result


def voigt_projection(data: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
    """Phono3py order is xx, yy, zz, yz, xz, xy."""
    x, y, z = unit_vector
    return (x*x*data[..., 0] + y*y*data[..., 1] + z*z*data[..., 2]
            + 2*y*z*data[..., 3] + 2*x*z*data[..., 4] + 2*x*y*data[..., 5])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    """Record exact input identity so later path changes are easy to audit."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
