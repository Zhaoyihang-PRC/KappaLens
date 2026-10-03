"""Small synthetic regression checks; no VASP/MD result is claimed here."""

from __future__ import annotations

import csv
import json
import tempfile
import types
import unittest
from pathlib import Path
from sys import modules
from unittest.mock import patch

import numpy as np

from kappalens.common import AnalysisError, init_groups, load_config, load_groups
from kappalens.compare import compare_models
from kappalens.dsf import _dho, analyze_dsf, fit_peak
from kappalens.gk import _cross_corr, analyze_gk
from kappalens.modes import _group_weights, analyze_modes


class FakeFile:
    def __init__(self, datasets):
        self.datasets = datasets

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __contains__(self, key):
        return key in self.datasets

    def __getitem__(self, key):
        return self.datasets[key]


class ProjectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        atom_map = self.root / "atom_map.csv"
        atom_map.write_text("poscar_index_1based,element\n1,C\n2,H\n")
        self.model = {"name": "test", "atom_map": "atom_map.csv", "groups_csv": "groups.csv",
                      "directions_cart": {"x": [1, 0, 0]}, "temperature_k": 300,
                      "primitive_order_matches_atom_map": True,
                      "kappa_hdf5": "kappa.hdf5", "phonon_hdf5": "phonon.hdf5",
                      "heat_current_csv": "current.csv", "volume_ang3": 1000,
                      "gk_blocks": 4, "max_lag_ps": 0.2, "plateau_ps": [0.05, 0.15]}
        (self.root / "project.json").write_text(json.dumps({"models": [self.model]}))
        self.config = load_config(self.root / "project.json")

    def assign_groups(self):
        (self.root / "groups.csv").write_text(
            "poscar_index_1based,element,group\n1,C,backbone\n2,H,sidechain\n")

    def test_group_template_requires_assignment(self):
        init_groups(self.config, self.model)
        with self.assertRaisesRegex(AnalysisError, "no group"):
            load_groups(self.config, self.model)
        self.assign_groups()
        groups = load_groups(self.config, self.model)
        self.assertEqual(groups["sidechain"].tolist(), [1])
        self.assertEqual(_group_weights(np.eye(6, dtype=complex), groups)["backbone"].tolist(),
                         [1, 1, 1, 0, 0, 0])

    def test_neutral_atom_indices_and_arbitrary_groups(self):
        (self.root / "atom_map.csv").write_text("atom_index_1based,element\n1,C\n2,H\n")
        path = init_groups(self.config, self.model)
        with path.open() as f:
            self.assertEqual(csv.DictReader(f).fieldnames,
                             ["atom_index_1based", "element", "group"])
        path.write_text("atom_index_1based,element,group\n1,C,framework\n2,H,pendant\n")
        groups = load_groups(self.config, self.model)
        self.assertEqual(set(groups), {"framework", "pendant"})

    def test_mode_projection_reconstructs_phono3py(self):
        self.assign_groups()
        (self.root / "kappa.hdf5").touch()
        (self.root / "phonon.hdf5").touch()
        mk = np.zeros((1, 1, 6, 6))
        mk[0, 0, :, 0] = 1.0
        kappa = np.array([[6.0, 0, 0, 0, 0, 0]])
        k = {"mesh": np.array([1, 1, 1]), "weight": np.array([1]),
             "temperature": np.array([300]), "mode_kappa": mk, "kappa": kappa,
             "frequency": np.arange(1, 7, dtype=float)[None, :],
             "qpoint": np.zeros((1, 3))}
        p = {"mesh": np.array([1, 1, 1]), "ir_grid_points": np.array([0]),
             "grid_address": np.zeros((1, 3)), "frequency": k["frequency"],
             "eigenvector": np.eye(6, dtype=complex)[None, :, :]}
        fake_h5 = type("FakeH5", (), {"File": staticmethod(lambda path, mode: FakeFile(
            k if Path(path).name == "kappa.hdf5" else p))})
        with patch("kappalens.modes._h5py", return_value=fake_h5):
            out = analyze_modes(self.config, self.model)
        result = json.loads((out / "summary.json").read_text())
        self.assertAlmostEqual(result["directions"]["x"]["projected_w_mk"]["backbone"], 3)
        self.assertAlmostEqual(result["directions"]["x"]["projected_w_mk"]["sidechain"], 3)
        with (out / "mode_projection.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 6)
        comparison = compare_models(self.config, "modes")
        self.assertTrue((comparison / "model_components.csv").is_file())
        k["kappa"][0, 0] = 7.0
        with patch("kappalens.modes._h5py", return_value=fake_h5):
            with self.assertRaisesRegex(AnalysisError, "does not reconstruct"):
                analyze_modes(self.config, self.model)
        self.assertEqual(list(out.glob(".mode_projection-*.tmp")), [])

    def test_gk_cross_terms_reconstruct_total(self):
        self.assign_groups()
        rng = np.random.default_rng(7)
        backbone = rng.normal(size=(512, 3))
        sidechain = 0.3 * backbone + rng.normal(size=(512, 3))
        columns = ["time_ps"] + [f"Q_total_{a}" for a in "xyz"]
        columns += [f"Q_{g}_{a}" for g in ("backbone", "sidechain") for a in "xyz"]
        with (self.root / "current.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(len(backbone)):
                writer.writerow([i * 0.01, *(backbone[i] + sidechain[i]),
                                 *backbone[i], *sidechain[i]])
        out = analyze_gk(self.config, self.model)
        result = json.loads((out / "summary.json").read_text())["directions"]["x"]
        pieces = sum(value["mean_w_mk"] for key, value in result.items()
                     if "__" in key)
        self.assertAlmostEqual(pieces, result["total"]["mean_w_mk"], places=9)
        x, y = backbone[:16, 0], sidechain[:16, 0]
        brute = [np.mean((x[:16-lag] - x.mean()) * (y[lag:] - y.mean()))
                 for lag in range(4)]
        np.testing.assert_allclose(_cross_corr(x, y, 3), brute, rtol=1e-12, atol=1e-12)

    def test_gk_md_only_without_phonon_atom_map(self):
        md_model = {key: value for key, value in self.model.items()
                    if key not in ("atom_map", "groups_csv")}
        md_model["heat_current_groups"] = ["chain", "pendant"]
        rng = np.random.default_rng(12)
        chain = rng.normal(size=(512, 3))
        pendant = rng.normal(size=(512, 3))
        columns = ["time_ps"] + [f"Q_total_{a}" for a in "xyz"]
        columns += [f"Q_{g}_{a}" for g in ("chain", "pendant") for a in "xyz"]
        with (self.root / "current.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(512):
                writer.writerow([i * 0.01, *(chain[i] + pendant[i]), *chain[i], *pendant[i]])
        out = analyze_gk(self.config, md_model)
        result = json.loads((out / "summary.json").read_text())
        self.assertEqual(result["heat_current_groups"], ["chain", "pendant"])
        self.assertNotIn("groups_csv", result["input_sha256"])
        metal_kappa = result["directions"]["x"]["total"]["mean_w_mk"]
        # 1 kcal/mol Angstrom/fs = 43.36410424180093 eV Angstrom/ps.
        factor = 43.36410424180093
        columns[0] = "time_fs"
        with (self.root / "current.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(512):
                writer.writerow([i * 10.0, *((chain[i] + pendant[i]) / factor),
                                 *(chain[i] / factor), *(pendant[i] / factor)])
        md_model["heat_current_unit"] = "kcal_per_mol_angstrom_per_fs"
        real_result = json.loads((analyze_gk(self.config, md_model) / "summary.json").read_text())
        self.assertEqual(real_result["source_time_column"], "time_fs")
        self.assertAlmostEqual(real_result["directions"]["x"]["total"]["mean_w_mk"],
                               metal_kappa, places=9)

    def test_dsf_peak_fit(self):
        omega = np.linspace(0.5, 4, 200)
        spectrum = _dho(omega, 1.5, 2.0, 0.25, 0.1)
        fit = fit_peak(omega, spectrum, 1.0, 3.0)
        self.assertAlmostEqual(fit["omega0"], 2.0, places=2)
        self.assertAlmostEqual(fit["gamma"], 0.25, places=2)
        self.assertGreater(fit["r_squared"], 0.99)

    def test_dsf_native_sample_adapter(self):
        omega = np.linspace(0.5, 4, 200)
        signal = _dho(omega, 1.5, 2.0, 0.25, 0.1)
        class Sample:
            def __init__(self):
                self.omega = omega
                self.q_points = np.array([[0, 0, 0]])

            def __getitem__(self, field):
                return signal[None, :]

        sample = Sample()
        self.model.update({"dsf_npz": "sample.npz", "dsf_omega_unit": "rad/fs",
                           "dsf_fit_windows": [{"field": "Clqw_backbone_backbone", "q_index": 0,
                                                "omega_min": 1.0, "omega_max": 3.0}]})
        (self.root / "sample.npz").touch()
        fake = types.ModuleType("dynasor")
        fake.read_sample_from_npz = lambda path: sample
        with patch.dict(modules, {"dynasor": fake}):
            out = analyze_dsf(self.config, self.model)
        self.assertTrue((out / "fitted_peaks.csv").is_file())

    def test_dsf_csv_adapter(self):
        omega = np.linspace(0.5, 4, 200)
        signal = _dho(omega, 1.5, 2.0, 0.25, 0.1)
        with (self.root / "spectrum.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["q_index", "q_x", "q_y", "q_z", "omega", "Clqw_framework_framework"])
            for x, y in zip(omega, signal):
                writer.writerow([0, 0, 0, 0, x, y])
        self.model.update({"dsf_csv": "spectrum.csv", "dsf_omega_unit": "rad/ps",
                           "dsf_fit_windows": [{"field": "Clqw_framework_framework", "q_index": 0,
                                                "omega_min": 1.0, "omega_max": 3.0}]})
        out = analyze_dsf(self.config, self.model)
        result = json.loads((out / "summary.json").read_text())
        self.assertEqual(result["source_type"], "dsf_csv")
        self.assertAlmostEqual(result["fits"][0]["omega0"], 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
