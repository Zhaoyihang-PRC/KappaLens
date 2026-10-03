"""Read VASP's electron-phonon HDF5 gap output without running VASP.

This intentionally does not turn a gap correction into a transport lifetime or
reconstruct a whole renormalized band structure. VASP only evaluates the
self-energy for the states selected by the calculation. Ambiguous accumulator,
spin-channel and temperature axes are rejected rather than guessed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import __version__
from .common import AnalysisError, model_output, resolve, sha256, write_csv, write_json

_ELECTRONS = "/results/electron_phonon/electrons"
_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
_CHECKED_TAGS = ("ISPIN", "ISYM", "LHFCALC", "ELPH_MODE", "ELPH_RUN",
                 "ELPH_SELFEN_FAN", "ELPH_SELFEN_DW", "ELPH_SELFEN_GAPS")


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise AnalysisError("renorm needs h5py; install kappalens[demo] or kappalens[elph].") from exc
    return h5py


def _text(value) -> str:
    """Decode small scalar/array HDF5 string datasets, including VASP text copies."""
    array = np.asarray(value)
    if array.size == 0 or array.dtype.kind not in "SUO":
        return ""
    if array.dtype.kind == "S" and array.dtype.itemsize == 1:
        return b"".join(array.reshape(-1).tolist()).decode("utf-8", errors="replace")
    parts = []
    for item in array.reshape(-1):
        if isinstance(item, bytes):
            parts.append(item.decode("utf-8", errors="replace"))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _version(h5, h5py) -> tuple[int, int, int]:
    if "version" not in h5:
        raise AnalysisError("vaspout.h5 has no /version; cannot verify the VASP release.")
    obj = h5["version"]
    values = {}
    strings = []
    if isinstance(obj, h5py.Dataset):
        strings.append(_text(obj[()]))
    else:
        def collect(name, item):
            if isinstance(item, h5py.Dataset) and item.size <= 32:
                key = name.rsplit("/", 1)[-1].lower()
                raw = item[()]
                if np.asarray(raw).size == 1 and np.asarray(raw).dtype.kind in "iu":
                    values[key] = int(np.asarray(raw).item())
                strings.append(_text(raw))
        obj.visititems(collect)
        strings.extend(_text(value) for value in obj.attrs.values())
    if "major" in values and "minor" in values and ("patch" in values or "revision" in values):
        return values["major"], values["minor"], values.get("patch", values.get("revision"))
    for candidate in strings:
        match = _VERSION_RE.search(candidate)
        if match:
            return tuple(int(part) for part in match.groups())
    raise AnalysisError("Cannot read a VASP X.Y.Z version from /version; supply a supported vaspout.h5.")


def _incar_tags(text: str) -> dict[str, str]:
    tags = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].split("!", 1)[0]
        for part in line.split(";"):
            match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", part)
            if match:
                tags[match.group(1).upper()] = match.group(2).strip().upper()
    return tags


def _embedded_incar(h5, h5py) -> str:
    # /original contains the supplied input text. /input contains effective
    # parsed settings but its schema varies, so it is not treated as text.
    for name in ("/original/incar", "/original/INCAR"):
        if name in h5 and isinstance(h5[name], h5py.Dataset):
            return _text(h5[name][()])
    return ""


def _check_original_file(h5, h5py, config: dict, options: dict, key: str) -> dict | None:
    """Compare an optional local input with VASP's own saved original text."""
    if not options.get(key):
        return None
    path = resolve(config, options[key], f"electron_phonon.{key}")
    try:
        external = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"Cannot read {key.upper()} {path}: {exc}") from exc
    internal_path = f"/original/{key}"
    if internal_path not in h5 or not isinstance(h5[internal_path], h5py.Dataset):
        raise AnalysisError(f"Cannot compare {key.upper()}: {internal_path} is absent from HDF5.")
    internal = _text(h5[internal_path][()])
    # VASP's original input is a text copy; newline encoding and final blank
    # lines are immaterial, but any input-line content difference is rejected.
    normalize = lambda value: value.replace("\r\n", "\n").rstrip("\n")
    if normalize(external) != normalize(internal):
        raise AnalysisError(f"{key.upper()} differs between HDF5 and {path}.")
    return {"path": str(path), "sha256": sha256(path), "matches_hdf5_original": True}


def _check_outcar(path: Path, version: tuple[int, int, int]) -> dict[str, str]:
    """An optional second version and completion check, without loading OUTCAR."""
    try:
        with path.open("rb") as handle:
            head = handle.read(8192).decode("utf-8", errors="replace")
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise AnalysisError(f"Cannot read OUTCAR {path}: {exc}") from exc
    match = re.search(r"\bvasp[.\s]+(\d+\.\d+\.\d+)\b", head, re.IGNORECASE)
    if not match:
        raise AnalysisError(f"Cannot find VASP version in OUTCAR header: {path}")
    found = tuple(int(value) for value in match.group(1).split("."))
    if found != version:
        raise AnalysisError(f"OUTCAR version {match.group(1)} differs from HDF5 version "
                            f"{'.'.join(map(str, version))}.")
    if "General timing and accounting informations for this job" not in tail:
        raise AnalysisError(f"OUTCAR completion footer not found: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def _boolean(tags: dict[str, str], key: str, default: bool = False) -> bool:
    value = tags.get(key)
    if value is None:
        return default
    token = value.split()[0].strip(".")
    if token in ("T", "TRUE"):
        return True
    if token in ("F", "FALSE"):
        return False
    raise AnalysisError(f"INCAR {key} must be true or false, got {value!r}.")


def _integer(tags: dict[str, str], key: str, default: int) -> int:
    try:
        return int(tags.get(key, str(default)).split()[0])
    except ValueError as exc:
        raise AnalysisError(f"INCAR {key} must be an integer.") from exc


def _scalar(dataset, label: str) -> float:
    array = np.asarray(dataset[()])
    if array.size != 1 or array.dtype.kind not in "iuf":
        raise AnalysisError(f"{label} must be one real number.")
    number = float(array.reshape(-1)[0])
    if not np.isfinite(number):
        raise AnalysisError(f"{label} must be finite.")
    return number


def _real_values(dataset, label: str) -> np.ndarray:
    raw = np.asarray(dataset[()])
    if raw.dtype.kind not in "iuf":
        raise AnalysisError(f"{label} must contain real numeric values.")
    return raw.astype(float)


def _finite_dataset(dataset, label: str) -> None:
    """Check the complete dataset in bounded slabs instead of loading it all."""
    if dataset.dtype.kind not in "iufc" or dataset.size == 0:
        raise AnalysisError(f"{label} must contain numeric data.")
    if dataset.ndim == 0:
        if not np.isfinite(dataset[()]):
            raise AnalysisError(f"{label} contains NaN or infinity.")
        return
    tail = int(np.prod(dataset.shape[1:]))
    block = max(1, 8_000_000 // max(1, tail * dataset.dtype.itemsize))
    for start in range(0, dataset.shape[0], block):
        if not np.all(np.isfinite(dataset[start:start + block])):
            raise AnalysisError(f"{label} contains NaN or infinity.")


def _gap_ks(dataset, label: str, channel: int | None) -> tuple[float, int]:
    raw = _real_values(dataset, label).reshape(-1)
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        raise AnalysisError(f"{label} must contain finite gap values.")
    if raw.size > 1 and channel is None:
        raise AnalysisError(f"{label} has {raw.size} channels; set electron_phonon.gap_channel (zero based).")
    index = channel if channel is not None else 0
    if index < 0 or index >= raw.size:
        raise AnalysisError(f"{label}: gap_channel {index} is outside 0..{raw.size - 1}.")
    return float(raw[index]), index


def _gap_qp(dataset, label: str, ntemps: int, channel: int,
            temperature_axis: int | None, channel_explicit: bool) -> np.ndarray:
    raw = _real_values(dataset, label)
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        raise AnalysisError(f"{label} must contain finite gap values.")
    if raw.ndim == 0:
        if ntemps != 1 or channel != 0:
            raise AnalysisError(f"{label}: scalar QP gap does not match temperature/channel selection.")
        return raw.reshape(1)
    if temperature_axis is None:
        axes = [axis for axis, size in enumerate(raw.shape) if size == ntemps]
        # Singleton axes carry no independent information for one temperature.
        if ntemps == 1:
            axes = [axis for axis in axes if axis == raw.ndim - 1] or axes
        if len(axes) != 1:
            raise AnalysisError(f"{label}: ambiguous temperature axis in shape {raw.shape}; "
                                "set electron_phonon.temperature_axis (zero based).")
        temperature_axis = axes[0]
    if not 0 <= temperature_axis < raw.ndim or raw.shape[temperature_axis] != ntemps:
        raise AnalysisError(f"{label}: temperature_axis does not match {ntemps} temperatures.")
    series = np.moveaxis(raw, temperature_axis, -1).reshape(-1, ntemps)
    if len(series) > 1 and not channel_explicit:
        raise AnalysisError(f"{label} has {len(series)} channels; "
                            "set electron_phonon.gap_channel (zero based).")
    if channel >= len(series):
        raise AnalysisError(f"{label}: gap_channel {channel} is absent from QP data.")
    return series[channel]


def _report(summary: dict, rows: list[dict]) -> str:
    lines = ["# Electron–phonon gap renormalization", "",
             f"Model: `{summary['model']}`  ",
             f"VASP: `{summary['vasp_version']}`  ",
             f"Accumulator: `{summary['accumulator']}`  ",
             f"Input: `{summary['input']['path']}`  ",
             f"SHA-256: `{summary['input']['sha256']}`", "",
             "**Synthetic demo data: no physical interpretation.**" if summary["synthetic_demo"] else "",
             "",
             "**Status:** HDF5 structure and numeric checks passed; physical convergence and the "
             "original supercell/potential have not been validated.", "",
             "| Gap | T (K) | KS (eV) | Renormalized (eV) | Shift (meV) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['gap_type']} | {row['temperature_k']:.3f} | "
                     f"{row['ks_gap_ev']:.6f} | {row['renormalized_gap_ev']:.6f} | "
                     f"{row['shift_mev']:.3f} |")
    lines.extend(["", "## Checks", "",
                  f"- INCAR source: {summary['incar_source']}",
                  "- POSCAR original match: " + ("passed" if summary["original_inputs"].get("poscar") else "not checked"),
                  "- KPOINTS original match: " + ("passed" if summary["original_inputs"].get("kpoints") else "not checked"),
                  "- OUTCAR version/completion: " + ("passed" if summary["outcar"] else "not checked"),
                  f"- ISPIN={summary['incar']['ISPIN']}, ISYM={summary['incar']['ISYM']}",
                  f"- Fan shape: {summary['self_energy']['fan_shape']}",
                  f"- Debye–Waller shape: {summary['self_energy']['dw_shape']}",
                  f"- Broadening delta: {summary['self_energy']['delta_ev']} eV",
                  f"- Intermediate bands: {summary['self_energy']['nbands_sum']}", "",
                  "The Fan/Debye–Waller arrays are checked but are **not** assigned "
                  "to separate gap shifts; that requires validated band-edge state mapping.",
                  "This report does not calculate electronic transport or group-resolved scattering.",
                  "Check k/q mesh, supercell, ENCUT, NBANDS and broadening convergence before using values in research.",
                  "", "## Literature context", "",
                  "- VASP documentation: [gap workflow](https://vasp.at/wiki/Bandgap_renormalization_due_to_electron-phonon_coupling), "
                  "[accumulator format](https://vasp.at/wiki/Electron-phonon_accumulators), "
                  "and [known issues](https://vasp.at/wiki/Known_issues).",
                  "- Ning, Lei, Yang & Xi, *Phys. Chem. Chem. Phys.* **25**, 26006–26013 (2023), "
                  "[doi:10.1039/D3CP03596D](https://doi.org/10.1039/D3CP03596D): "
                  "temperature-dependent band renormalization and state linewidths.",
                  "- Zhao, Li, Xi & Yang, *Comput. Mater. Today* **5**, 100019 (2025), "
                  "[doi:10.1016/j.commt.2024.100019](https://doi.org/10.1016/j.commt.2024.100019): "
                  "different 2D band edges can shift differently with temperature.",
                  "These papers give context; they do not validate this file or calculation.", ""])
    return "\n".join(lines)


def analyze_renorm(config: dict, model: dict) -> Path:
    """Produce a gap-vs-temperature report from one selected VASP accumulator."""
    options = model.get("electron_phonon")
    if not isinstance(options, dict):
        raise AnalysisError(f"{model['name']}: set electron_phonon.vaspout_h5 in the project config.")
    path = resolve(config, options.get("vaspout_h5"), "electron_phonon.vaspout_h5")
    if not path.is_file():
        raise AnalysisError(f"VASP HDF5 file not found: {path}")
    h5py = _h5py()
    rows = []
    try:
        with h5py.File(path, "r") as h5:
            synthetic_demo = bool(h5.attrs.get("kappalens_synthetic_demo", False))
            version = _version(h5, h5py)
            if version < (6, 5, 0):
                raise AnalysisError("Electron-phonon renormalization requires VASP 6.5.0 or newer.")
            outcar_source = (_check_outcar(resolve(config, options["outcar"], "electron_phonon.outcar"),
                                           version) if options.get("outcar") else None)
            original_inputs = {key: result for key in ("poscar", "kpoints")
                               if (result := _check_original_file(h5, h5py, config, options, key))}
            embedded = _embedded_incar(h5, h5py)
            embedded_tags = _incar_tags(embedded)
            incar_source = "vaspout.h5:/original/incar" if embedded else None
            tags = embedded_tags.copy()
            if options.get("incar"):
                incar_path = resolve(config, options["incar"], "electron_phonon.incar")
                try:
                    external_tags = _incar_tags(incar_path.read_text(encoding="utf-8"))
                except OSError as exc:
                    raise AnalysisError(f"Cannot read INCAR {incar_path}: {exc}") from exc
                for key in _CHECKED_TAGS:
                    if key in tags and key in external_tags and tags[key] != external_tags[key]:
                        raise AnalysisError(f"INCAR {key} differs between HDF5 and {incar_path}.")
                tags.update(external_tags)
                incar_source = (incar_source + " and " if incar_source else "") + str(incar_path)
            if not incar_source:
                raise AnalysisError("Cannot verify INCAR settings; supply electron_phonon.incar.")
            ispin, isym = _integer(tags, "ISPIN", 1), _integer(tags, "ISYM", 2)
            if ispin not in (1, 2):
                raise AnalysisError("INCAR ISPIN must be 1 or 2.")
            if ispin == 2 and version < (6, 6, 0):
                raise AnalysisError("VASP 6.5.0/6.5.1 ISPIN=2 electron-phonon results are affected "
                                    "by known issues 54 and 65; use a corrected VASP release.")
            if ispin == 2 and isym != 0:
                raise AnalysisError("Magnetic renormalization requires ISYM=0 for this demo "
                                    "because of VASP known issue 109.")
            if _boolean(tags, "LHFCALC"):
                raise AnalysisError("Perturbative electron-phonon renormalization with hybrid "
                                    "functionals is unsupported in this demo.")
            if tags.get("ELPH_MODE", "").split()[:1] == ["TRANSPORT"]:
                raise AnalysisError("ELPH_MODE=TRANSPORT is not a band-gap renormalization run.")
            if "ELPH_RUN" in tags and not _boolean(tags, "ELPH_RUN"):
                raise AnalysisError("INCAR disables ELPH_RUN.")
            for key in ("ELPH_SELFEN_FAN", "ELPH_SELFEN_DW", "ELPH_SELFEN_GAPS"):
                if key in tags and not _boolean(tags, key):
                    raise AnalysisError(f"INCAR disables {key}; cannot report a complete gap renormalization.")
            if _ELECTRONS not in h5:
                raise AnalysisError(f"Missing {_ELECTRONS}; this file has no electron-phonon results.")
            electrons = h5[_ELECTRONS]
            groups = sorted(int(match.group(1)) for name in electrons
                            if (match := re.fullmatch(r"self_energy_(\d+)", name)))
            if not groups:
                raise AnalysisError("No self_energy_N accumulator found in vaspout.h5.")
            selected = options.get("accumulator")
            if selected is None:
                if len(groups) != 1:
                    raise AnalysisError(f"Multiple self-energy accumulators {groups}; "
                                        "set electron_phonon.accumulator explicitly.")
                selected = groups[0]
            if isinstance(selected, bool) or not isinstance(selected, int) or selected not in groups:
                raise AnalysisError(f"accumulator must be one of {groups}.")
            group = electrons[f"self_energy_{selected}"]
            required = ("temps", "selfen_fan", "selfen_dw", "delta", "nbands_sum")
            missing = [key for key in required if key not in group]
            if missing:
                raise AnalysisError(f"self_energy_{selected} is missing {missing}.")
            temps = _real_values(group["temps"], "temps")
            if (temps.ndim != 1 or not len(temps) or not np.all(np.isfinite(temps))
                    or np.any(temps < 0) or np.any(np.diff(temps) <= 0)):
                raise AnalysisError("Self-energy temperatures must be finite, nonnegative and increasing.")
            fan, dw = group["selfen_fan"], group["selfen_dw"]
            if (fan.ndim < 3 or dw.ndim < 2 or fan.shape[0] != dw.shape[0]
                    or len(temps) not in fan.shape[1:-1] or len(temps) not in dw.shape[1:]):
                raise AnalysisError("Fan/Debye-Waller shapes do not match states and temperatures.")
            _finite_dataset(fan, "selfen_fan")
            _finite_dataset(dw, "selfen_dw")
            delta = _scalar(group["delta"], "selfen_delta")
            nbands_sum = _scalar(group["nbands_sum"], "nbands_sum")
            if delta < 0 or nbands_sum < 1 or not nbands_sum.is_integer():
                raise AnalysisError("delta must be nonnegative and nbands_sum a positive integer.")
            channel = options.get("gap_channel")
            axis = options.get("temperature_axis")
            for key, value in (("gap_channel", channel), ("temperature_axis", axis)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise AnalysisError(f"electron_phonon.{key} must be a nonnegative integer.")
            for kind in ("direct", "fundamental"):
                ks_name, qp_name = f"{kind}_gap", f"{kind}_gap_renorm"
                if ks_name not in group and qp_name not in group:
                    continue
                if ks_name not in group or qp_name not in group:
                    raise AnalysisError(f"{kind} gap needs both {ks_name} and {qp_name}.")
                ks, selected_channel = _gap_ks(group[ks_name], ks_name, channel)
                qp = _gap_qp(group[qp_name], qp_name, len(temps), selected_channel,
                             axis, channel is not None)
                for temperature, value in zip(temps, qp):
                    rows.append({"gap_type": kind, "temperature_k": float(temperature),
                                 "ks_gap_ev": ks, "renormalized_gap_ev": float(value),
                                 "shift_mev": 1000 * (float(value) - ks)})
            if not rows:
                raise AnalysisError("No direct/fundamental gap datasets found; use ELPH_SELFEN_GAPS=.TRUE. "
                                    "or select a gap-renormalization accumulator.")
            if "self_energy_meta" in electrons and "ncalculators" in electrons["self_energy_meta"]:
                count = int(_scalar(electrons["self_energy_meta/ncalculators"], "ncalculators"))
                if count != len(groups):
                    raise AnalysisError("self_energy_meta/ncalculators disagrees with self_energy_N groups.")
            summary = {"model": model["name"], "analysis_version": __version__,
                       "method": "VASP HDF5 electron-phonon gap renormalization, read only",
                       "status": "format_checks_passed_physical_convergence_unverified",
                       "vasp_version": ".".join(map(str, version)), "accumulator": selected,
                       "synthetic_demo": synthetic_demo,
                       "available_accumulators": groups,
                       "incar_source": incar_source,
                       "incar": {"ISPIN": ispin, "ISYM": isym, "ELPH_MODE": tags.get("ELPH_MODE")},
                       "input": {"path": str(path), "sha256": sha256(path)},
                       "outcar": outcar_source,
                       "original_inputs": original_inputs,
                       "self_energy": {"fan_shape": list(fan.shape), "dw_shape": list(dw.shape),
                                       "delta_ev": delta, "nbands_sum": int(nbands_sum)},
                       "temperatures_k": temps.tolist(),
                       "gap_types": sorted({row["gap_type"] for row in rows}),
                       "limitations": ["No k/q mesh, supercell, ENCUT, band-sum or broadening convergence proof.",
                                       "No independent check of the generating electron-phonon potential.",
                                       "Original INCAR text is checked; effective parsed INCAR settings in /input are not audited.",
                                       "Run completion is unverified unless an OUTCAR is provided.",
                                       "No state-resolved Fan/Debye-Waller gap attribution or transport calculation."]}
    except OSError as exc:
        raise AnalysisError(f"Cannot open VASP HDF5 file {path}: {exc}") from exc
    out = model_output(config, model, "renorm")
    write_json(out / "summary.json", summary)
    write_csv(out / "gaps.csv", list(rows[0]), rows)
    (out / "report.md").write_text(_report(summary, rows), encoding="utf-8")
    return out
