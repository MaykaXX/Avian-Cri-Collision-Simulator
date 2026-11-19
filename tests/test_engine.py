"""Unit tests for collision_risk_engine.py (stdlib unittest; also runnable with pytest)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collision_risk_engine as cre  # noqa: E402


class TestNormalisation(unittest.TestCase):
    def test_lux_450_maps_to_075(self):
        self.assertAlmostEqual(float(cre.normalize_lux(450.0, 600.0)), 0.75)

    def test_visibility_limits_and_hazard_value(self):
        self.assertAlmostEqual(float(cre.normalize_visibility(3000.0, 3000.0, 50.0)), 0.0)
        self.assertAlmostEqual(float(cre.normalize_visibility(50.0, 3000.0, 50.0)), 1.0)
        self.assertAlmostEqual(float(cre.normalize_visibility(180.0, 3000.0, 50.0)), 0.69, places=2)

    def test_snr_and_clipping(self):
        self.assertEqual(float(cre.normalize_snr(-5.0, 0.0, 20.0)), 0.0)
        self.assertEqual(float(cre.normalize_snr(30.0, 0.0, 20.0)), 1.0)
        self.assertAlmostEqual(float(cre.normalize_snr(10.0, 0.0, 20.0)), 0.5)


class TestConfig(unittest.TestCase):
    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            cre.CriWeights(radar=0.5)
        cre.CriWeights()  # defaults are valid


class TestEngineLogic(unittest.TestCase):
    def setUp(self):
        self.cfg = cre.EngineConfig()
        self.engine = cre.CollisionRiskEngine(self.cfg, step_s=10.0)

    def test_single_channel_cannot_trigger(self):
        """Saturating the acoustic channel alone must stay below the trigger threshold."""
        cri = self.engine.compute_cri(d_radar=0.35, a_calls=1.0, m_weather=0.25, l_lux=0.75)
        self.assertLess(cri, self.cfg.threshold_on)

    def test_activation_requires_confirmation(self):
        high = (0.9, 0.9, 0.9, 0.9)
        first = self.engine.update(*high)
        self.assertIsNone(first.event)
        second = self.engine.update(*high)
        self.assertEqual(second.event, "ACTIVATE")
        self.assertTrue(second.active)

    def test_isolated_spike_is_ignored(self):
        high, low = (0.9, 0.9, 0.9, 0.9), (0.2, 0.0, 0.2, 0.75)
        for sample in (high, low, high, low, high, low):
            self.assertIsNone(self.engine.update(*sample).event)

    def test_hold_time_and_release_confirmation(self):
        high, low = (0.9, 0.9, 0.9, 0.9), (0.1, 0.0, 0.1, 0.2)
        for _ in range(self.cfg.confirm_steps):
            self.engine.update(*high)
        self.assertTrue(self.engine.active)
        events = [self.engine.update(*low).event for _ in range(int(self.cfg.min_hold_s / 10.0) - 1)]
        self.assertNotIn("RELEASE", events)  # still within the minimum hold time
        events = [self.engine.update(*low).event for _ in range(self.cfg.release_confirm_steps + 2)]
        self.assertIn("RELEASE", events)
        self.assertFalse(self.engine.active)

    def test_value_inside_hysteresis_band_keeps_protection_active(self):
        for _ in range(self.cfg.confirm_steps):
            self.engine.update(0.9, 0.9, 0.9, 0.9)
        mid = (0.6, 0.6, 0.6, 0.6)  # CRI 0.6 lies between threshold_off and threshold_on
        for _ in range(200):
            self.engine.update(*mid)
        self.assertTrue(self.engine.active)


class TestAnalysisHelpers(unittest.TestCase):
    def test_find_windows(self):
        active = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
        self.assertEqual(cre.find_windows(active), [(1, 3), (5, 6), (7, 9)])

    def test_find_windows_empty(self):
        self.assertEqual(cre.find_windows(np.zeros(5, dtype=bool)), [])


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = cre.SimConfig(seed=42)
        cls.sensors = cre.generate_sensor_traces(cls.sim, cre.RadarModel(), cre.WeatherModel(), cre.AcousticModel())
        cls.result = cre.run_engine(cls.sim, cls.sensors, cre.EngineConfig(), cre.LightingModel(), cre.ActuationConfig())

    def test_inputs_are_normalised(self):
        for arr in (self.sensors.d_radar, self.sensors.a_calls, self.sensors.m_weather, self.result.l_lux):
            self.assertGreaterEqual(float(arr.min()), 0.0)
            self.assertLessEqual(float(arr.max()), 1.0)

    def test_reproducible_for_same_seed(self):
        again = cre.generate_sensor_traces(self.sim, cre.RadarModel(), cre.WeatherModel(), cre.AcousticModel())
        np.testing.assert_array_equal(self.sensors.a_calls, again.a_calls)
        other = cre.generate_sensor_traces(cre.SimConfig(seed=7), cre.RadarModel(), cre.WeatherModel(), cre.AcousticModel())
        self.assertFalse(np.array_equal(self.sensors.a_calls, other.a_calls))

    def test_cpu_latencies_recorded_for_every_step(self):
        self.assertEqual(len(self.result.cpu_latencies_ms), self.sim.n_steps)

    def test_events_alternate_between_activate_and_release(self):
        events = [d.event for _, d in self.result.decisions]
        self.assertTrue(events)
        self.assertEqual(events[0], "ACTIVATE")
        for a, b in zip(events, events[1:]):
            self.assertNotEqual(a, b)

    def test_scripted_noise_burst_does_not_trigger_dimming(self):
        burst_step = int(cre.AcousticModel().stress_burst_min * 60 / self.sim.step_s)
        self.assertFalse(self.result.actuation_active[burst_step - 6: burst_step + 30].any())

    def test_dimming_reduces_measured_illuminance(self):
        windows = cre.find_windows(self.result.actuation_active)
        self.assertTrue(windows)
        start, end = windows[0]
        self.assertLess(float(np.median(self.result.lux[start + 6: end])), 150.0)


class TestNetworkAndLog(unittest.TestCase):
    def test_inference_stays_within_budget(self):
        rng = np.random.default_rng(0)
        samples = [cre.NetworkModel().sample(rng) for _ in range(2000)]
        self.assertTrue(all(s["edge_inference_ms"] < 45.0 for s in samples))
        for s in samples[:50]:
            parts = s["edge_inference_ms"] + s["uplink_ms"] + s["bms_processing_ms"] + s["ack_downlink_ms"]
            self.assertAlmostEqual(parts, s["end_to_end_ms"])

    def test_cli_writes_valid_log_and_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            cre.main(["--no-show", "--out-dir", tmp])
            log = json.loads((Path(tmp) / "bms_mqtt_log.json").read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "cri_simulation_graph.png").exists())
        self.assertEqual(log["meta"]["message_count"], len(log["messages"]))
        first = log["messages"][0]
        self.assertEqual(first["payload"]["command"], "ACTIVATE_BIRD_PROTECTION")
        self.assertEqual(first["payload"]["facade_dimming_pct"], 75)
        self.assertEqual(first["payload"]["acoustic_emitters"]["freq_khz_min"], 4.0)
        self.assertEqual(first["payload"]["acoustic_emitters"]["freq_khz_max"], 6.0)
        self.assertGreaterEqual(first["payload"]["cri"], 0.75)
        self.assertIn("end_to_end_ms", first["latency_ms"])


if __name__ == "__main__":
    unittest.main()
