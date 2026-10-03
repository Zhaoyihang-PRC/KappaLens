"""Command line entry point. All diagnostic messages are intentionally plain English."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .common import AnalysisError, current_groups, get_models, init_groups, load_config, load_groups, resolve
from .compare import compare_models
from .dsf import analyze_dsf
from .electrons import analyze_electrons, export_electron_states
from .gk import analyze_gk
from .modes import analyze_modes
from .thermoelectric import analyze_thermoelectric


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kappalens", description="Independent lattice and electronic transport analysis")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["check", "init-groups", "modes", "gk", "dsf", "electrons",
                 "electron-export", "thermoelectric"]:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, help="project JSON file")
        command.add_argument("--model", help="one model name; default: all models")
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--config", required=True)
    cmp_parser.add_argument("--stage", choices=["modes", "gk", "electrons"], required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "compare":
            print(f"Written: {compare_models(config, args.stage)}")
            return 0
        models = get_models(config, args.model)
        for model in models:
            if args.command == "init-groups":
                result = init_groups(config, model)
            elif args.command == "check":
                if model.get("atom_map") or model.get("groups_csv"):
                    groups = load_groups(config, model)
                    print(f"{model['name']} mapped atoms: {', '.join(f'{k}={len(v)}' for k,v in groups.items())}")
                if model.get("heat_current_csv"):
                    print(f"{model['name']} heat-current groups: {', '.join(current_groups(config, model))}")
                for key in ("kappa_hdf5", "phonon_hdf5", "heat_current_csv", "dsf_npz", "dsf_csv"):
                    if model.get(key):
                        path = resolve(config, model[key], f"{model['name']}.{key}")
                        print(f"{model['name']} {key}: {'found' if path.is_file() else 'missing'} {path}")
                electronic = model.get("electronic", {})
                for key in ("states_csv", "eigenval", "groupvec", "kpoints", "symmetry", "poscar",
                            "state_character_csv", "transfer_fluctuations_csv"):
                    if electronic.get(key):
                        path = resolve(config, electronic[key], f"{model['name']}.electronic.{key}")
                        print(f"{model['name']} electronic.{key}: {'found' if path.is_file() else 'missing'} {path}")
                relaxation = electronic.get("relaxation", {})
                if relaxation.get("rates_csv"):
                    path = resolve(config, relaxation["rates_csv"], "electronic.relaxation.rates_csv")
                    print(f"{model['name']} electronic.relaxation.rates_csv: "
                          f"{'found' if path.is_file() else 'missing'} {path}")
                continue
            elif args.command == "thermoelectric":
                result = analyze_thermoelectric(config, model)
            elif args.command == "electron-export":
                result = export_electron_states(config, model)
            else:
                result = {"modes": analyze_modes, "gk": analyze_gk, "dsf": analyze_dsf,
                          "electrons": analyze_electrons}[args.command](config, model)
            print(f"Written: {result}")
        return 0
    except (ValueError, TypeError) as exc:
        print(f"ERROR: Invalid numeric value in config or input: {exc}", file=sys.stderr)
        return 2
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
