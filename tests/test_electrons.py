"""Analytic electron tensors and input guards; all fixtures are synthetic."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kappalens.common import AnalysisError, load_config
from kappalens.compare import compare_models
from kappalens.electron_io import read_legacy_vasp
from kappalens.electrons import CHARGE_C, KB_J_K, analyze_electrons, export_electron_states
from kappalens.thermoelectric import analyze_thermoelectric


class ElectronTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.velocities = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],
                                    [1.0, 1.0, 0.3]])*1e5
        self.energies = np.array([-0.08, 0.05, -0.03, 0.11])
        self.ids = [f"k{i}_b1_s0" for i in range(4)]
        self._states()
        self.model = {"name": "toy", "temperature_k": 300,
                      "directions_cart": {"x": [1, 0, 0], "y": [0, 1, 0]},
                      "electronic": {"source": "states_csv", "states_csv": "states.csv",
                                     "full_brillouin_zone": True,
                                     "spin_degeneracy": 2, "cell_volume_ang3": 1000,
                                     "dimensionality": "3d", "chemical_potentials_ev": [0.0],
                                     "relaxation": {"mode": "constant", "tau_s": 2e-14}}}
        self._config()

    def _states(self):
        with (self.root / "states.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state_id", "k_index", "band_index", "spin", "energy_ev",
                             "v_x_m_s", "v_y_m_s", "v_z_m_s", "k_weight"])
            for i, (energy, velocity) in enumerate(zip(self.energies, self.velocities)):
                writer.writerow([self.ids[i], i, 1, 0, energy, *velocity, 0.25])

    def _config(self):
        (self.root / "project.json").write_text(json.dumps({"models": [self.model]}))
        self.config = load_config(self.root / "project.json")

    def _result(self):
        out = analyze_electrons(self.config, self.model)
        return json.loads((out / "summary.json").read_text())["results"][0]

    def test_full_tensor_agrees_with_independent_matrix_formula(self):
        result = self._result()
        temperature = 300.0
        x = self.energies*CHARGE_C/(KB_J_K*temperature)
        f = 1/(1+np.exp(x))
        derivative = f*(1-f)/(KB_J_K*temperature)
        factor = (2*0.25*2e-14*derivative)/(1000*1e-30)
        moments = [np.einsum("n,ni,nj->ij", factor*(self.energies*CHARGE_C)**p,
                             self.velocities, self.velocities) for p in range(3)]
        zero, first, second = moments
        expected_s = -np.linalg.solve(zero, first)/(CHARGE_C*temperature)
        expected_k = (second-first@np.linalg.solve(zero, first))/temperature
        np.testing.assert_allclose(result["sigma_s_m_tensor"], CHARGE_C**2*zero, rtol=1e-12)
        np.testing.assert_allclose(result["seebeck_v_k_tensor"], expected_s, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(result["kappa_e_w_mk_tensor"], expected_k, rtol=1e-12)
        # The discarded elementwise formula is observably different here.
        wrong_s = -np.linalg.inv(zero)*first/(CHARGE_C*temperature)
        self.assertGreater(np.max(abs(expected_s-wrong_s)), 1e-6)

    def test_rate_channels_match_states_and_sum_to_one(self):
        with (self.root / "rates.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state_id", "acoustic_s_inv", "optical_s_inv"])
            for sid in reversed(self.ids):
                writer.writerow([sid, 1e13, 3e13])
        self.model["electronic"]["relaxation"] = {"mode": "rates_csv", "rates_csv": "rates.csv",
                                                  "temperature_k": 300,
                                                  "mechanisms": ["acoustic_s_inv", "optical_s_inv"]}
        self._config()
        out = analyze_electrons(self.config, self.model)
        with (out / "mechanism_fractions.csv").open() as f:
            rows = list(csv.DictReader(f))
        for direction in ("x", "y"):
            shares = [float(r["transport_weighted_rate_fraction"]) for r in rows
                      if r["direction"] == direction]
            self.assertAlmostEqual(sum(shares), 1.0)
            self.assertAlmostEqual(shares[0], 0.25)
        self.model["electronic"]["relaxation"]["temperature_k"] = 400
        with self.assertRaisesRegex(AnalysisError, "temperatures differ"):
            analyze_electrons(self.config, self.model)

    def test_sheet_result_is_invariant_to_artificial_vacuum(self):
        options = self.model["electronic"]
        options.update({"dimensionality": "2d", "sheet_normal_cart": [0, 0, 1],
                        "sheet_repeat_length_ang": 10})
        self._config()
        first = self._result()["directions"]["x"]
        options["cell_volume_ang3"] = 2000
        options["sheet_repeat_length_ang"] = 20
        self._config()
        second = self._result()["directions"]["x"]
        self.assertAlmostEqual(second["sigma_s_m"]/first["sigma_s_m"], 0.5)
        self.assertAlmostEqual(second["sheet_conductance_s"], first["sheet_conductance_s"])
        self.assertAlmostEqual(second["sheet_kappa_e_w_k"], first["sheet_kappa_e_w_k"])
        options["directions_cart"] = {"out_of_plane": [0, 0, 1]}
        with self.assertRaisesRegex(AnalysisError, "not in the declared 2D sheet plane"):
            analyze_electrons(self.config, self.model)

    def test_character_projection_and_disorder_are_diagnostics(self):
        with (self.root / "characters.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state_id", "framework", "sidechain"])
            for sid in self.ids:
                writer.writerow([sid, 0.7, 0.3])
        with (self.root / "fluctuations.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_ps", "pair", "transfer_ev"])
            for i in range(12):
                writer.writerow([i*0.01, "bridge", 0.1+0.01*np.sin(i)])
        self.model["electronic"].update({"state_character_csv": "characters.csv",
                                          "transfer_fluctuations_csv": "fluctuations.csv"})
        self._config()
        out = analyze_electrons(self.config, self.model)
        with (out / "state_character_projection.csv").open() as f:
            rows = list(csv.DictReader(f))
        sigma_x = self._result()["directions"]["x"]["sigma_s_m"]
        self.assertAlmostEqual(sum(float(r["sigma_character_s_m"]) for r in rows
                                   if r["direction"] == "x"), sigma_x)
        summary = json.loads((out / "summary.json").read_text())
        self.assertIn("bridge", summary["transfer_fluctuations"])

    def test_carrier_target_inverts_fermi_occupancy(self):
        self.model["electronic"].pop("chemical_potentials_ev")
        self.model["electronic"].update({"reference_electrons_per_cell": 1.0,
                                          "extra_electrons_per_cell": [0.2]})
        self._config()
        result = self._result()
        self.assertAlmostEqual(result["extra_electrons_per_cell"], 0.2, places=9)

    def test_legacy_vasp_odd_mesh_symmetry_expansion(self):
        (self.root / "POSCAR").write_text("synthetic\n1\n10 0 0\n0 10 0\n0 0 10\n")
        (self.root / "KPOINTS").write_text("mesh\n0\nGamma\n3 1 1\n0 0 0\n")
        (self.root / "SYMMETRY").write_text(
            "2\n1 0 0\n0 1 0\n0 0 1\n\n-1 0 0\n0 -1 0\n0 0 -1\n")
        with (self.root / "EIGENVAL").open("w") as eigen, (self.root / "GROUPVEC").open("w") as velocity:
            eigen.write("0 0 0 1\nheader\nheader\nheader\nheader\n2 2 1\n")
            for k in (0, 1/3):
                eigen.write(f"\n{k:.10f} 0 0 0.5\n1 0.0 1.0\n")
                velocity.write(f"{k:.10f} 0 0 0.5\n1 1000 0 0\n")
        options = {"source": "vasp_legacy", "eigenval": "EIGENVAL", "groupvec": "GROUPVEC",
                   "kpoints": "KPOINTS", "symmetry": "SYMMETRY", "poscar": "POSCAR",
                   "velocity_unit": "m_per_s", "time_reversal": False,
                   "spin_degeneracy": 2}
        states = read_legacy_vasp(self.config, self.model, options)
        self.assertEqual(len(states.state_id), 3)
        self.assertEqual(len(set(states.state_id)), 3)
        self.assertAlmostEqual(states.k_weight.sum(), 1.0)
        self.assertEqual(sorted(states.velocity_m_s[:, 0].tolist()), [-1000, 1000, 1000])
        self.assertAlmostEqual(states.cell_volume_ang3, 1000)
        self.model["electronic"] = options
        self._config()
        out = export_electron_states(self.config, self.model)
        with (out / "expanded_states.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows[0]), {"state_id", "k_index", "band_index", "k_x", "k_y", "k_z",
                                        "spin", "energy_ev", "v_x_m_s", "v_y_m_s", "v_z_m_s", "k_weight"})

    def test_nonorthogonal_symmetry_rotation_preserves_speed(self):
        # 120-degree reciprocal operation in a skew hexagonal cell.
        matrix = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
        lattice = np.array([[1, 0, 0], [-0.5, np.sqrt(3)/2, 0], [0, 0, 2]])
        reciprocal = np.linalg.inv(lattice).T
        rotation = (np.linalg.inv(reciprocal) @ matrix @ reciprocal).T
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(rotation @ [1.0, 2.0, 0.5]),
                                   np.linalg.norm([1.0, 2.0, 0.5]), atol=1e-12)

    def test_electron_comparison_requires_explicit_matching_scan(self):
        first = self._result()
        self.assertIn("x", first["directions"])
        self.model["name"] = "toy_b"
        self.model["electronic"]["cell_volume_ang3"] = 1200
        self._config()
        analyze_electrons(self.config, self.model)
        compare_config = load_config(self.root / "project.json")
        compare_config["models"] = [{"name": "toy"}, {"name": "toy_b"}]
        compare_config["electronic_comparison_basis"] = "chemical_potential_ev"
        compare_config["comparisons"] = [{"left": "toy", "right": "toy_b"}]
        out = compare_models(compare_config, "electrons")
        self.assertTrue((out / "contrasts.csv").is_file())
        compare_config["electronic_comparison_basis"] = "extra_carriers_cm3"
        with self.assertRaisesRegex(AnalysisError, "carrier scan point"):
            compare_models(compare_config, "electrons")

    def test_combination_checks_matching_sources(self):
        electron_result = self._result()
        out = self.root / "analysis_results" / "toy" / "gk"
        out.mkdir(parents=True)
        lattice = {"temperature_k": 300, "volume_ang3": 1000,
                   "directions": {name: {"direction_cart_unit": v["direction_cart_unit"],
                                         "total": {"mean_w_mk": 2.0, "sem_w_mk": 0.1}}
                                  for name, v in electron_result["directions"].items()}}
        (out / "summary.json").write_text(json.dumps(lattice))
        self.model["thermoelectric"] = {"lattice_stage": "gk", "lattice_dimensionality": "3d",
                                        "same_cell_normalization_verified": True}
        self._config()
        result = analyze_thermoelectric(self.config, self.model)
        with (result / "combined.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertGreater(float(rows[0]["zt_or_intraband_proxy"]), 0)
        self.model["thermoelectric"]["same_cell_normalization_verified"] = False
        with self.assertRaisesRegex(AnalysisError, "same_cell_normalization_verified"):
            analyze_thermoelectric(self.config, self.model)
        mode_out = self.root / "analysis_results" / "toy" / "modes"
        mode_out.mkdir(parents=True)
        mode_summary = {"temperature_k": 300,
                        "directions": {name: {"direction_cart_unit": v["direction_cart_unit"],
                                              "kappa_intraband_w_mk": 1.5}
                                       for name, v in electron_result["directions"].items()}}
        (mode_out / "summary.json").write_text(json.dumps(mode_summary))
        self.model["thermoelectric"].update({"same_cell_normalization_verified": True,
                                              "lattice_stage": "modes"})
        mode_combined = analyze_thermoelectric(self.config, self.model)
        with (mode_combined / "combined.csv").open() as f:
            mode_rows = list(csv.DictReader(f))
        self.assertTrue(all(row["lattice_scope"] == "intraband_only" for row in mode_rows))


if __name__ == "__main__":
    unittest.main()
