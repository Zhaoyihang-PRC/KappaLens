"""Synthetic schema and failure tests for the independent VASP HDF5 reader."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kappalens.common import AnalysisError, load_config
from kappalens.renorm import analyze_renorm

try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipUnless(h5py is not None, "h5py is needed for HDF5 tests")
class RenormTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "vaspout.h5"
        self.model = {"name": "synthetic", "electron_phonon": {"vaspout_h5": "vaspout.h5"}}
        (self.root / "project.json").write_text(json.dumps({"models": [self.model]}))
        self.config = load_config(self.root / "project.json")
        self._fixture()

    def _fixture(self):
        with h5py.File(self.path, "w") as h5:
            h5.attrs["kappalens_synthetic_demo"] = True
            h5["version/major"] = 6
            h5["version/minor"] = 5
            h5["version/patch"] = 1
            h5["original/incar"] = np.bytes_("ISPIN=1\nELPH_MODE=RENORM\n")
            h5["original/poscar"] = np.bytes_("Synthetic cell\n1.0\n")
            h5["original/kpoints"] = np.bytes_("Synthetic mesh\n0\n")
            group = h5.require_group("results/electron_phonon/electrons/self_energy_1")
            group["temps"] = [0, 300, 600]
            group["delta"] = 0.01
            group["nbands_sum"] = 8
            group["selfen_fan"] = np.zeros((2, 1, 3, 2))
            group["selfen_dw"] = np.zeros((2, 3))
            group["direct_gap"] = [1.2]
            group["direct_gap_renorm"] = [[1.16, 1.14, 1.11]]
            group["fundamental_gap"] = [0.9]
            group["fundamental_gap_renorm"] = [[0.87, 0.84, 0.8]]

    def test_report_calculates_shifts_and_marks_synthetic_origin(self):
        out = analyze_renorm(self.config, self.model)
        with (out / "gaps.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(float(rows[0]["shift_mev"]), -40.0)
        summary = json.loads((out / "summary.json").read_text())
        self.assertTrue(summary["synthetic_demo"])
        self.assertEqual(summary["vasp_version"], "6.5.1")
        self.assertIn("physical_convergence_unverified", summary["status"])
        self.assertIn("not** assigned", (out / "report.md").read_text())

    def test_spin_polarized_651_is_blocked(self):
        with h5py.File(self.path, "r+") as h5:
            h5["original/incar"][()] = np.bytes_("ISPIN=2\nISYM=0\nELPH_MODE=RENORM\n")
        with self.assertRaisesRegex(AnalysisError, "known issues 54 and 65"):
            analyze_renorm(self.config, self.model)

    def test_original_structure_and_kpoints_must_match(self):
        (self.root / "POSCAR").write_text("Different cell\n1.0\n")
        self.model["electron_phonon"]["poscar"] = "POSCAR"
        with self.assertRaisesRegex(AnalysisError, "POSCAR differs"):
            analyze_renorm(self.config, self.model)
        (self.root / "POSCAR").write_text("Synthetic cell\n1.0\n")
        (self.root / "KPOINTS").write_text("Synthetic mesh\n0\n")
        self.model["electron_phonon"]["kpoints"] = "KPOINTS"
        summary = json.loads((analyze_renorm(self.config, self.model) / "summary.json").read_text())
        self.assertTrue(summary["original_inputs"]["poscar"]["matches_hdf5_original"])
        self.assertTrue(summary["original_inputs"]["kpoints"]["matches_hdf5_original"])
        (self.root / "KPOINTS").write_text("Other mesh\n0\n")
        with self.assertRaisesRegex(AnalysisError, "KPOINTS differs"):
            analyze_renorm(self.config, self.model)

    def test_qp_channels_need_explicit_selection(self):
        with h5py.File(self.path, "r+") as h5:
            group = h5["results/electron_phonon/electrons/self_energy_1"]
            del group["direct_gap_renorm"]
            group["direct_gap_renorm"] = [[1.16, 1.14, 1.11], [1.15, 1.13, 1.10]]
        with self.assertRaisesRegex(AnalysisError, "channels; set electron_phonon.gap_channel"):
            analyze_renorm(self.config, self.model)
        self.model["electron_phonon"]["gap_channel"] = 0
        self.assertTrue((analyze_renorm(self.config, self.model) / "gaps.csv").is_file())

    def test_missing_dw_or_bad_temperature_axis_is_rejected(self):
        with h5py.File(self.path, "r+") as h5:
            del h5["results/electron_phonon/electrons/self_energy_1/selfen_dw"]
        with self.assertRaisesRegex(AnalysisError, "missing.*selfen_dw"):
            analyze_renorm(self.config, self.model)
        self._fixture()
        with h5py.File(self.path, "r+") as h5:
            group = h5["results/electron_phonon/electrons/self_energy_1"]
            del group["direct_gap_renorm"]
            group["direct_gap_renorm"] = [[1.16, 1.14]]
        with self.assertRaisesRegex(AnalysisError, "temperature axis"):
            analyze_renorm(self.config, self.model)

    def test_multiple_accumulators_require_selection(self):
        with h5py.File(self.path, "r+") as h5:
            parent = h5["results/electron_phonon/electrons"]
            parent.copy("self_energy_1", "self_energy_2")
        with self.assertRaisesRegex(AnalysisError, "Multiple self-energy accumulators"):
            analyze_renorm(self.config, self.model)
        self.model["electron_phonon"]["accumulator"] = 2
        self.assertEqual(json.loads((analyze_renorm(self.config, self.model) / "summary.json").read_text())
                         ["accumulator"], 2)

    def test_external_incar_cannot_silently_disagree(self):
        (self.root / "INCAR").write_text("ISPIN=2\nELPH_MODE=RENORM\n")
        self.model["electron_phonon"]["incar"] = "INCAR"
        with self.assertRaisesRegex(AnalysisError, "INCAR ISPIN differs"):
            analyze_renorm(self.config, self.model)

    def test_outcar_version_and_completion_are_checked(self):
        outcar = self.root / "OUTCAR"
        outcar.write_text("vasp.6.6.0 synthetic\nGeneral timing and accounting informations for this job:\n")
        self.model["electron_phonon"]["outcar"] = "OUTCAR"
        with self.assertRaisesRegex(AnalysisError, "OUTCAR version 6.6.0 differs"):
            analyze_renorm(self.config, self.model)
        outcar.write_text("vasp.6.5.1 synthetic\n")
        with self.assertRaisesRegex(AnalysisError, "completion footer not found"):
            analyze_renorm(self.config, self.model)
        outcar.write_text("vasp.6.5.1 synthetic\nGeneral timing and accounting informations for this job:\n")
        summary = json.loads((analyze_renorm(self.config, self.model) / "summary.json").read_text())
        self.assertEqual(summary["outcar"]["path"], str(outcar.resolve()))


if __name__ == "__main__":
    unittest.main()
