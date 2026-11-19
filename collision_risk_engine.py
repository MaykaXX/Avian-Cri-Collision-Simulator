#!/usr/bin/env python3
"""
Collision Risk Engine (CRE) - simulation prototype
===================================================

Simulates one night of nocturnal bird migration over an illuminated facade and
evaluates a fused Collision Risk Index (CRI) from four sensor channels:

    CRI = w1 * D_radar + w2 * A_calls + w3 * M_weather + w4 * L_lux

    D_radar   - radar-derived migration traffic rate (MTR, birds/km/h), normalised
    A_calls   - confirmed nocturnal flight calls (NFC) from edge acoustic classifiers
    M_weather - fog hazard derived from meteorological visibility (m), log-normalised
    L_lux     - facade illuminance (lux), measured in a closed loop with the dimming

When the CRI crosses the trigger threshold, the engine emits a JSON command for the
Building Management System (BMS) over (simulated) MQTT: facade dimming plus acoustic
deterrent emitters. A hysteresis band, a confirmation counter and a minimum hold
time prevent relay chatter at the threshold.

Scope and honesty notes
-----------------------
* All sensor streams are SYNTHETIC but generated from explicit physical models in
  physical units (dBA, kHz, lux, metres, birds/km/h). Every model constant is a
  configurable assumption, not a measurement.
* Network and edge-inference latencies are sampled from statistical models
  (log-normal / truncated normal). They are NOT measurements of a real link.
* "CPU latency" is the measured wall-clock time of the CRI update on the host
  running this script (NumPy/CPython), not on the ESP32-S3 target.

Usage
-----
    python collision_risk_engine.py [--seed 42] [--out-dir .] [--no-show]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CriWeights:
    """Weights of the CRI formula. They must sum to 1.0."""

    radar: float = 0.15
    calls: float = 0.35
    weather: float = 0.30
    lux: float = 0.20

    def __post_init__(self) -> None:
        total = self.radar + self.calls + self.weather + self.lux
        if not np.isclose(total, 1.0):
            raise ValueError(f"CRI weights must sum to 1.0, got {total:.3f}")


@dataclass(frozen=True)
class SimConfig:
    """Global simulation parameters."""

    duration_h: float = 8.0
    step_s: float = 10.0
    start_time: datetime = datetime(2026, 10, 15, 21, 0, tzinfo=timezone.utc)
    seed: int = 42

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_h * 3600.0 / self.step_s))

    @property
    def time_min(self) -> np.ndarray:
        return np.arange(self.n_steps) * self.step_s / 60.0


@dataclass(frozen=True)
class RadarModel:
    """Radar migration traffic rate (MTR) model, in birds/km/h."""

    mtr_base: float = 350.0
    mtr_peak: float = 800.0
    mtr_ref_max: float = 1000.0  # MTR that maps to D_radar = 1.0
    noise_sigma_rel: float = 0.04
    noise_phi: float = 0.95


@dataclass(frozen=True)
class WeatherModel:
    """Meteorological visibility model, in metres."""

    visibility_base_m: float = 1100.0
    visibility_fog_min_m: float = 130.0
    fog_center_min: float = 240.0
    fog_width_min: float = 130.0
    clear_visibility_m: float = 3000.0  # M_weather = 0 at or above this value
    floor_visibility_m: float = 50.0  # M_weather = 1 at or below this value
    hazard_visibility_m: float = 180.0  # operational fog hazard limit
    noise_sigma_ln: float = 0.06
    noise_phi: float = 0.98


@dataclass(frozen=True)
class LightingModel:
    """Facade illuminance model, in lux."""

    nominal_lux: float = 450.0
    lux_norm_max: float = 600.0  # lux that maps to L_lux = 1.0 (450 lux -> 0.75)
    ramp_tau_s: float = 15.0  # first-order dimming ramp time constant
    sensor_noise_rel: float = 0.02


@dataclass(frozen=True)
class AcousticModel:
    """Edge acoustic NFC detection chain, in dBA / Hz."""

    city_noise_dba: float = 58.0
    city_noise_sigma_db: float = 1.2
    nfc_level_range_dba: Tuple[float, float] = (62.0, 78.0)
    nfc_band_hz: Tuple[float, float] = (2000.0, 9000.0)
    confidence_threshold: float = 0.88
    # Classifier confidence = sigmoid((SNR - midpoint) / slope) + Gaussian jitter.
    conf_midpoint_snr_db: float = 9.0
    conf_slope_db: float = 2.5
    conf_noise_sigma: float = 0.04
    # SNR normalisation range (dB) mapped to [0, 1].
    snr_floor_db: float = 0.0
    snr_ceiling_db: float = 20.0
    # Call arrival rate (calls per minute) with two migration pulses.
    call_rate_base_per_min: float = 1.0
    call_rate_peak_per_min: float = 30.0
    pulses: Tuple[Tuple[float, float], ...] = ((165.0, 34.0), (310.0, 36.0))  # (centre, width) in min
    # Evidence integration: A = 1 - exp(-E / E_ref), E = sum of confirmed-call SNR scores in window.
    evidence_window_s: float = 90.0
    evidence_ref: float = 3.5
    # Urban noise transients (sirens, vehicle pass-bys).
    transient_rate_per_hour: float = 6.0
    siren_probability: float = 0.4
    siren_level_dba: Tuple[float, float] = (72.0, 85.0)
    siren_duration_s: Tuple[float, float] = (20.0, 40.0)
    vehicle_level_dba: Tuple[float, float] = (66.0, 78.0)
    vehicle_duration_s: Tuple[float, float] = (5.0, 15.0)
    spurious_gate_probability: float = 0.06  # P(classifier misfires > threshold on a transient step)
    stress_burst_min: Optional[float] = 45.0  # scripted worst-case noise burst (None disables)


@dataclass(frozen=True)
class EngineConfig:
    """Decision logic parameters."""

    weights: CriWeights = CriWeights()
    threshold_on: float = 0.75
    threshold_off: float = 0.48  # hysteresis release threshold (below the CRI seen while dimmed)
    confirm_steps: int = 2  # consecutive steps >= threshold_on required to activate
    release_confirm_steps: int = 12  # consecutive steps < threshold_off required to release (120 s)
    min_hold_s: float = 600.0  # minimum time the protection stays active


@dataclass(frozen=True)
class ActuationConfig:
    """BMS / MQTT actuation parameters."""

    dimming_level: float = 0.75
    emitter_freq_khz: Tuple[float, float] = (4.0, 6.0)
    topic: str = "facade/bms/bird-protection/command"
    qos: int = 1


@dataclass(frozen=True)
class NetworkModel:
    """Statistical model of command delivery latency (all values in ms)."""

    rtt_median_ms: float = 110.0  # median round-trip time of the uplink (e.g. LTE gateway)
    rtt_sigma_ln: float = 0.40  # log-normal shape parameter
    inference_mean_ms: float = 30.0  # ESP32-S3 model inference time
    inference_sigma_ms: float = 6.0
    inference_min_ms: float = 12.0
    inference_max_ms: float = 44.9  # design budget: < 45 ms
    bms_processing_mean_ms: float = 35.0  # controller parsing + relay actuation
    bms_processing_sigma_ms: float = 8.0

    def sample(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample one end-to-end latency breakdown (ms)."""
        inference = float(np.clip(rng.normal(self.inference_mean_ms, self.inference_sigma_ms),
                                  self.inference_min_ms, self.inference_max_ms))
        rtt = float(rng.lognormal(np.log(self.rtt_median_ms), self.rtt_sigma_ln))
        bms = float(max(5.0, rng.normal(self.bms_processing_mean_ms, self.bms_processing_sigma_ms)))
        uplink, downlink = rtt / 2.0, rtt / 2.0  # command delivery + controller acknowledgement
        return {
            "edge_inference_ms": inference,
            "uplink_ms": uplink,
            "bms_processing_ms": bms,
            "ack_downlink_ms": downlink,
            "end_to_end_ms": inference + uplink + bms + downlink,
        }


# =============================================================================
# 2. NORMALISATION FUNCTIONS (physical units -> [0, 1])
# =============================================================================


def clip01(x):
    """Clip a scalar or array to [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def normalize_radar_mtr(mtr_birds_km_h, mtr_ref_max: float):
    """Linear normalisation of migration traffic rate."""
    return clip01(np.asarray(mtr_birds_km_h) / mtr_ref_max)


def normalize_lux(lux, lux_norm_max: float):
    """Linear normalisation of facade illuminance (450 lux -> 0.75 with 600 lux full scale)."""
    return clip01(np.asarray(lux) / lux_norm_max)


def normalize_visibility(visibility_m, clear_m: float, floor_m: float):
    """
    Logarithmic fog-hazard score: 0 at >= clear_m, 1 at <= floor_m.
    Log scale reflects that hazard grows quickly once visibility drops below a few hundred metres.
    With the defaults, 180 m maps to approximately 0.69.
    """
    vis = np.maximum(np.asarray(visibility_m, dtype=float), 1e-6)
    return clip01(np.log(clear_m / vis) / np.log(clear_m / floor_m))


def normalize_snr(snr_db, floor_db: float, ceiling_db: float):
    """Linear normalisation of a signal-to-noise ratio in dB."""
    return clip01((np.asarray(snr_db) - floor_db) / (ceiling_db - floor_db))


def saturating_evidence(evidence, evidence_ref: float):
    """Map accumulated evidence E >= 0 to [0, 1): 1 - exp(-E / E_ref)."""
    return 1.0 - np.exp(-np.asarray(evidence) / evidence_ref)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# 3. SYNTHETIC SENSOR DATA IN PHYSICAL UNITS
# =============================================================================


@dataclass(frozen=True)
class TransientEvent:
    """Short urban noise event (siren or vehicle pass-by)."""

    kind: str
    start_s: float
    duration_s: float
    level_dba: float
    force_spurious: bool = False


@dataclass
class SensorTraces:
    """Open-loop sensor streams (L_lux is produced in the closed loop)."""

    time_min: np.ndarray
    d_radar: np.ndarray
    a_calls: np.ndarray
    m_weather: np.ndarray
    mtr_birds_km_h: np.ndarray
    visibility_m: np.ndarray
    noise_total_dba: np.ndarray
    confirmed_calls: np.ndarray
    spurious_detections: np.ndarray
    events: List[TransientEvent]
    candidate_calls_total: int


def ar1_noise(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Zero-mean AR(1) process with stationary standard deviation `sigma`."""
    x = np.zeros(n)
    x[0] = rng.normal(0.0, sigma)
    innovation_sigma = sigma * np.sqrt(1.0 - phi ** 2)
    eps = rng.normal(0.0, innovation_sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def generate_radar(sim: SimConfig, model: RadarModel, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (MTR in birds/km/h, normalised D_radar)."""
    n = sim.n_steps
    envelope = np.sin(np.linspace(0.0, np.pi, n))
    mtr = (model.mtr_base + (model.mtr_peak - model.mtr_base) * envelope)
    mtr = mtr * np.exp(ar1_noise(rng, n, model.noise_phi, model.noise_sigma_rel))
    return mtr, normalize_radar_mtr(mtr, model.mtr_ref_max)


def generate_weather(sim: SimConfig, model: WeatherModel, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (visibility in m, normalised M_weather). Fog is interpolated in log space."""
    n = sim.n_steps
    fog = np.exp(-((sim.time_min - model.fog_center_min) / model.fog_width_min) ** 2)
    ln_vis = (np.log(model.visibility_base_m)
              + (np.log(model.visibility_fog_min_m) - np.log(model.visibility_base_m)) * fog
              + ar1_noise(rng, n, model.noise_phi, model.noise_sigma_ln))
    visibility = np.maximum(np.exp(ln_vis), model.floor_visibility_m)
    return visibility, normalize_visibility(visibility, model.clear_visibility_m, model.floor_visibility_m)


def schedule_transients(sim: SimConfig, model: AcousticModel, rng: np.random.Generator) -> List[TransientEvent]:
    """Poisson-scheduled sirens and vehicle pass-bys, plus an optional scripted stress burst."""
    total_s = sim.duration_h * 3600.0
    n_events = rng.poisson(model.transient_rate_per_hour * sim.duration_h)
    events: List[TransientEvent] = []
    for start in rng.uniform(0.0, total_s, n_events):
        if rng.random() < model.siren_probability:
            events.append(TransientEvent("siren", float(start),
                                         float(rng.uniform(*model.siren_duration_s)),
                                         float(rng.uniform(*model.siren_level_dba))))
        else:
            events.append(TransientEvent("vehicle", float(start),
                                         float(rng.uniform(*model.vehicle_duration_s)),
                                         float(rng.uniform(*model.vehicle_level_dba))))
    if model.stress_burst_min is not None:
        t0 = model.stress_burst_min * 60.0
        # Worst case: a siren followed by a loud vehicle, both misclassified as bird calls.
        events.append(TransientEvent("siren", t0, 35.0, 84.0, force_spurious=True))
        events.append(TransientEvent("vehicle", t0 + 48.0, 12.0, 78.0, force_spurious=True))
    return sorted(events, key=lambda e: e.start_s)


def generate_acoustics(sim: SimConfig, model: AcousticModel, rng: np.random.Generator):
    """
    Edge acoustic chain: noise floor -> NFC arrivals -> SNR -> classifier confidence
    -> confidence gate -> evidence integration -> A_calls in [0, 1].

    Transients raise the measured noise level (masking real calls) and can occasionally
    be misclassified as calls (spurious detections passing the confidence gate).
    """
    n = sim.n_steps
    step = sim.step_s

    # Background city noise (dBA) with slow fluctuations.
    background_dba = model.city_noise_dba + ar1_noise(rng, n, 0.9, model.city_noise_sigma_db)

    # Transient noise: energy-sum their contribution into each step's equivalent level.
    events = schedule_transients(sim, model, rng)
    transient_energy = np.zeros(n)
    spurious = np.zeros(n, dtype=bool)
    spurious_score = np.zeros(n)
    for ev in events:
        first = int(ev.start_s // step)
        last = min(int((ev.start_s + ev.duration_s) // step), n - 1)
        for i in range(first, last + 1):
            overlap = (min(ev.start_s + ev.duration_s, (i + 1) * step) - max(ev.start_s, i * step)) / step
            if overlap <= 0.0:
                continue
            transient_energy[i] += 10.0 ** ((ev.level_dba + 10.0 * np.log10(overlap)) / 10.0)
            if ev.force_spurious or rng.random() < model.spurious_gate_probability:
                spurious[i] = True
                snr = ev.level_dba - background_dba[i]
                spurious_score[i] = max(spurious_score[i], float(normalize_snr(snr, model.snr_floor_db, model.snr_ceiling_db)))
    noise_total_dba = 10.0 * np.log10(10.0 ** (background_dba / 10.0) + transient_energy)

    # NFC arrivals: two migration pulses over a low baseline rate.
    t = sim.time_min
    pulse = np.zeros(n)
    for centre, width in model.pulses:
        pulse += np.exp(-((t - centre) / width) ** 2)
    rate_per_min = model.call_rate_base_per_min + (model.call_rate_peak_per_min - model.call_rate_base_per_min) * np.clip(pulse, 0.0, 1.0)
    counts = rng.poisson(rate_per_min / 60.0 * step)

    evidence = spurious_score.copy()
    confirmed = np.zeros(n, dtype=int)
    for i in range(n):
        k = counts[i]
        if k == 0:
            continue
        call_level = rng.uniform(*model.nfc_level_range_dba, k)  # calls lie within 2-9 kHz by construction
        snr = call_level - noise_total_dba[i]  # transients mask calls by raising the noise level
        confidence = sigmoid((snr - model.conf_midpoint_snr_db) / model.conf_slope_db) + rng.normal(0.0, model.conf_noise_sigma, k)
        passed = (confidence > model.confidence_threshold) & (snr > model.snr_floor_db)
        confirmed[i] = int(passed.sum())
        evidence[i] += float(normalize_snr(snr[passed], model.snr_floor_db, model.snr_ceiling_db).sum())

    window = max(1, int(round(model.evidence_window_s / step)))
    rolling = np.convolve(evidence, np.ones(window), mode="full")[:n]
    a_calls = saturating_evidence(rolling, model.evidence_ref)
    return a_calls, noise_total_dba, confirmed, spurious, events, int(counts.sum())


def generate_sensor_traces(sim: SimConfig, radar: RadarModel, weather: WeatherModel,
                           acoustic: AcousticModel) -> SensorTraces:
    """Generate all open-loop sensor streams with independent, reproducible RNG streams."""
    seeds = np.random.SeedSequence(sim.seed).spawn(3)
    mtr, d_radar = generate_radar(sim, radar, np.random.default_rng(seeds[0]))
    visibility, m_weather = generate_weather(sim, weather, np.random.default_rng(seeds[1]))
    a_calls, noise_dba, confirmed, spurious, events, n_candidates = generate_acoustics(
        sim, acoustic, np.random.default_rng(seeds[2]))
    return SensorTraces(sim.time_min, d_radar, a_calls, m_weather, mtr, visibility,
                        noise_dba, confirmed, spurious, events, n_candidates)


# =============================================================================
# 4. RISK ENGINE (CRI + HYSTERESIS STATE MACHINE)
# =============================================================================


@dataclass(frozen=True)
class Decision:
    """Result of one engine update."""

    cri: float
    active: bool
    event: Optional[str]  # "ACTIVATE", "RELEASE" or None


class CollisionRiskEngine:
    """Stateful CRI evaluator with confirmation counter, hysteresis and minimum hold time."""

    def __init__(self, cfg: EngineConfig, step_s: float) -> None:
        self.cfg = cfg
        self.step_s = step_s
        self.active = False
        self._steps_above = 0
        self._steps_below = 0
        self._held_s = 0.0

    def compute_cri(self, d_radar: float, a_calls: float, m_weather: float, l_lux: float) -> float:
        w = self.cfg.weights
        return w.radar * d_radar + w.calls * a_calls + w.weather * m_weather + w.lux * l_lux

    def update(self, d_radar: float, a_calls: float, m_weather: float, l_lux: float) -> Decision:
        cri = self.compute_cri(d_radar, a_calls, m_weather, l_lux)
        event: Optional[str] = None
        if not self.active:
            self._steps_above = self._steps_above + 1 if cri >= self.cfg.threshold_on else 0
            if self._steps_above >= self.cfg.confirm_steps:
                self.active, self._held_s, self._steps_above, event = True, 0.0, 0, "ACTIVATE"
        else:
            self._held_s += self.step_s
            self._steps_below = self._steps_below + 1 if cri < self.cfg.threshold_off else 0
            if self._steps_below >= self.cfg.release_confirm_steps and self._held_s >= self.cfg.min_hold_s:
                self.active, self._steps_below, event = False, 0, "RELEASE"
        return Decision(cri, self.active, event)


class FacadeLightingSim:
    """Closed-loop facade illuminance: first-order ramp towards the commanded level."""

    def __init__(self, model: LightingModel, actuation: ActuationConfig, step_s: float,
                 rng: np.random.Generator) -> None:
        self.model, self.actuation, self.rng = model, actuation, rng
        self.alpha = 1.0 - np.exp(-step_s / model.ramp_tau_s)
        self.true_lux = model.nominal_lux

    def step(self, dimming_active: bool) -> float:
        """Advance one step and return the measured illuminance in lux."""
        target = self.model.nominal_lux * ((1.0 - self.actuation.dimming_level) if dimming_active else 1.0)
        self.true_lux += self.alpha * (target - self.true_lux)
        return float(self.true_lux * (1.0 + self.rng.normal(0.0, self.model.sensor_noise_rel)))


@dataclass
class RunResult:
    cri: np.ndarray
    actuation_active: np.ndarray
    lux: np.ndarray
    l_lux: np.ndarray
    decisions: List[Tuple[int, Decision]]
    cpu_latencies_ms: List[float]


def run_engine(sim: SimConfig, sensors: SensorTraces, engine_cfg: EngineConfig,
               lighting: LightingModel, actuation: ActuationConfig) -> RunResult:
    """Run the engine step by step; the CRI update is timed individually per step."""
    n = sim.n_steps
    engine = CollisionRiskEngine(engine_cfg, sim.step_s)
    facade = FacadeLightingSim(lighting, actuation, sim.step_s, np.random.default_rng(sim.seed + 1))
    cri = np.zeros(n)
    active = np.zeros(n, dtype=bool)
    lux = np.zeros(n)
    l_norm = np.zeros(n)
    decisions: List[Tuple[int, Decision]] = []
    latencies_ms: List[float] = []

    for i in range(n):
        lux[i] = facade.step(engine.active)  # reflects the command issued in an earlier step
        l_norm[i] = float(normalize_lux(lux[i], lighting.lux_norm_max))

        t_start = time.perf_counter_ns()
        decision = engine.update(sensors.d_radar[i], sensors.a_calls[i], sensors.m_weather[i], l_norm[i])
        latencies_ms.append((time.perf_counter_ns() - t_start) / 1e6)

        cri[i], active[i] = decision.cri, decision.active
        if decision.event is not None:
            decisions.append((i, decision))
    return RunResult(cri, active, lux, l_norm, decisions, latencies_ms)


# =============================================================================
# 5. BMS / MQTT COMMAND GENERATION
# =============================================================================


def iso_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_bms_record(msg_id: int, decision: Decision, step_index: int, sim: SimConfig,
                     sensors: SensorTraces, run: RunResult, engine_cfg: EngineConfig,
                     actuation: ActuationConfig, latency: Dict[str, float]) -> dict:
    """Build one MQTT log record: transport metadata, command payload and delivery latency."""
    activate = decision.event == "ACTIVATE"
    t_decision = sim.start_time + timedelta(seconds=step_index * sim.step_s)
    t_ack = t_decision + timedelta(milliseconds=latency["end_to_end_ms"])
    emitters = ({"enabled": True, "freq_khz_min": actuation.emitter_freq_khz[0],
                 "freq_khz_max": actuation.emitter_freq_khz[1]} if activate else {"enabled": False})
    payload = {
        "schema": "bms.bird_protection.command/1.0",
        "message_id": f"cre-{msg_id:04d}",
        "timestamp": iso_utc(t_decision),
        "command": "ACTIVATE_BIRD_PROTECTION" if activate else "RELEASE_BIRD_PROTECTION",
        "cri": round(decision.cri, 3),
        "cri_threshold": engine_cfg.threshold_on if activate else engine_cfg.threshold_off,
        "facade_dimming_pct": int(round(actuation.dimming_level * 100)) if activate else 0,
        "acoustic_emitters": emitters,
        "inputs": {
            "d_radar": round(float(sensors.d_radar[step_index]), 3),
            "a_calls": round(float(sensors.a_calls[step_index]), 3),
            "m_weather": round(float(sensors.m_weather[step_index]), 3),
            "l_lux": round(float(run.l_lux[step_index]), 3),
            "facade_lux": round(float(run.lux[step_index]), 1),
            "visibility_m": round(float(sensors.visibility_m[step_index]), 0),
        },
    }
    return {
        "topic": actuation.topic,
        "qos": actuation.qos,
        "retain": False,
        "published_at": iso_utc(t_decision),
        "bms_ack_at": iso_utc(t_ack),
        "latency_ms": {k: round(v, 2) for k, v in latency.items()},
        "payload": payload,
    }


def dispatch_commands(sim: SimConfig, sensors: SensorTraces, run: RunResult, engine_cfg: EngineConfig,
                      actuation: ActuationConfig, network: NetworkModel) -> List[dict]:
    """Create one log record per engine event (activation or release)."""
    rng = np.random.default_rng(sim.seed + 2)
    records = []
    for msg_id, (step_index, decision) in enumerate(run.decisions, start=1):
        latency = network.sample(rng)
        records.append(build_bms_record(msg_id, decision, step_index, sim, sensors, run,
                                        engine_cfg, actuation, latency))
    return records


def save_mqtt_log(records: List[dict], sim: SimConfig, engine_cfg: EngineConfig, path: Path) -> None:
    document = {
        "meta": {
            "generator": "collision_risk_engine.py",
            "note": "Synthetic simulation log. Latencies are sampled from statistical models, not measured.",
            "simulation_start": iso_utc(sim.start_time),
            "seed": sim.seed,
            "threshold_on": engine_cfg.threshold_on,
            "threshold_off": engine_cfg.threshold_off,
            "message_count": len(records),
        },
        "messages": records,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# 6. ANALYSIS
# =============================================================================


def find_windows(active: np.ndarray) -> List[Tuple[int, int]]:
    """Return contiguous (start, end_exclusive) windows found via False -> True / True -> False transitions."""
    padded = np.concatenate(([False], active, [False])).astype(int)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of a boolean mask by `radius` samples on each side."""
    return np.convolve(mask.astype(int), np.ones(2 * radius + 1, dtype=int), mode="same") > 0


def clock_label(sim: SimConfig, step_index: float) -> str:
    return (sim.start_time + timedelta(seconds=step_index * sim.step_s)).strftime("%H:%M")


# =============================================================================
# 7. REPORT
# =============================================================================


def print_report(sim: SimConfig, sensors: SensorTraces, run: RunResult, naive: RunResult,
                 records: List[dict], engine_cfg: EngineConfig, acoustic: AcousticModel,
                 weather: WeatherModel, lighting: LightingModel, actuation: ActuationConfig) -> None:
    windows = find_windows(run.actuation_active)
    dimmed_min = float(run.actuation_active.sum() * sim.step_s / 60.0)
    night_min = sim.duration_h * 60.0
    cpu = np.asarray(run.cpu_latencies_ms)
    e2e = np.asarray([r["latency_ms"]["end_to_end_ms"] for r in records])
    spurious_idx = np.flatnonzero(sensors.spurious_detections)
    quiet = ~dilate(run.actuation_active, 6)
    # Steps within the 60 s evidence window after a spurious detection, outside any risk window.
    after_spurious = np.convolve(sensors.spurious_detections.astype(int), np.ones(6, dtype=int), mode="full")[:sim.n_steps] > 0
    transient_quiet_mask = after_spurious & quiet
    max_cri_spurious = float(run.cri[transient_quiet_mask].max()) if transient_quiet_mask.any() else float("nan")
    hazard_min = float((sensors.visibility_m < weather.hazard_visibility_m).sum() * sim.step_s / 60.0)
    steady = [run.lux[s + 6:e] for s, e in windows if e - s > 6]
    steady_lux = np.concatenate(steady) if steady else np.array([])

    line = "=" * 66
    print(line)
    print("          COLLISION RISK ENGINE - SIMULATION REPORT")
    print(line)
    print(f"Simulated duration            : {sim.duration_h:.0f} h ({sim.n_steps} steps of {sim.step_s:.0f} s), seed {sim.seed}")
    print(f"Weights (radar/calls/wx/lux)  : {engine_cfg.weights.radar}/{engine_cfg.weights.calls}/"
          f"{engine_cfg.weights.weather}/{engine_cfg.weights.lux}")
    print(f"Trigger / release thresholds  : CRI >= {engine_cfg.threshold_on:.2f} (x{engine_cfg.confirm_steps} steps) / "
          f"CRI < {engine_cfg.threshold_off:.2f} (x{engine_cfg.release_confirm_steps} steps, hold >= {engine_cfg.min_hold_s:.0f} s)")
    print("-" * 66)
    print("INPUT MODEL")
    print(f"  Urban noise floor           : {acoustic.city_noise_dba:.0f} dBA (mean measured {sensors.noise_total_dba.mean():.1f} dBA incl. transients)")
    print(f"  NFC band / level / gate     : {acoustic.nfc_band_hz[0] / 1000:.0f}-{acoustic.nfc_band_hz[1] / 1000:.0f} kHz / "
          f"{acoustic.nfc_level_range_dba[0]:.0f}-{acoustic.nfc_level_range_dba[1]:.0f} dBA / confidence > {acoustic.confidence_threshold}")
    print(f"  NFC candidates / confirmed  : {sensors.candidate_calls_total} / {int(sensors.confirmed_calls.sum())}")
    print(f"  Noise transients injected   : {len(sensors.events)} (sirens/vehicles, incl. scripted stress burst)")
    print(f"  Spurious gate detections    : {len(spurious_idx)} steps (transient misclassified as call)")
    if np.isfinite(max_cri_spurious):
        print(f"  Max CRI during those bursts : {max_cri_spurious:.3f} outside risk windows (threshold {engine_cfg.threshold_on:.2f})")
    print(f"  Time with visibility < {weather.hazard_visibility_m:.0f} m : {hazard_min:.0f} min (min {sensors.visibility_m.min():.0f} m)")
    print(f"  Peak radar MTR              : {sensors.mtr_birds_km_h.max():.0f} birds/km/h")
    print(f"  Facade lux nominal / dimmed : {lighting.nominal_lux:.0f} / "
          + (f"{np.median(steady_lux):.0f} (median, steady state at {actuation.dimming_level * 100:.0f}% dimming)" if steady_lux.size else "n/a"))
    print("-" * 66)
    print("DECISION RESULTS")
    print(f"  Peak / mean CRI             : {run.cri.max():.3f} / {run.cri.mean():.3f}")
    print(f"  Dimming windows (F->T edges): {len(windows)}")
    for k, (s, e) in enumerate(windows, start=1):
        print(f"    window {k}: {clock_label(sim, s)}-{clock_label(sim, e)} UTC, "
              f"{(e - s) * sim.step_s / 60:.1f} min, peak CRI {run.cri[s:e].max():.3f}")
    print(f"  Total dimming runtime       : {dimmed_min:.1f} min of {night_min:.0f} min")
    print(f"  Facade at full output       : {100.0 * (night_min - dimmed_min) / night_min:.1f}% of the night")
    print(f"  Relay transitions           : {len(run.decisions)} with hysteresis vs "
          f"{len(naive.decisions)} for a naive single-threshold controller")
    print("-" * 66)
    print("LATENCY")
    print(f"  CRI update CPU time (host)  : mean {cpu.mean():.4f} ms, p99 {np.percentile(cpu, 99):.4f} ms, max {cpu.max():.4f} ms (n={cpu.size})")
    if e2e.size:
        print(f"  MQTT/BMS end-to-end (model) : mean {e2e.mean():.1f} ms, min {e2e.min():.1f}, max {e2e.max():.1f} ms (n={e2e.size} commands)")
    else:
        print("  MQTT/BMS end-to-end (model) : no commands were issued")
    print(line)


# =============================================================================
# 8. VISUALISATION
# =============================================================================


def plot_results(sim: SimConfig, sensors: SensorTraces, run: RunResult, engine_cfg: EngineConfig,
                 path: Path, show: bool) -> None:
    """Two stacked subplots: CRI with actuation windows, and the four normalised sensor inputs."""
    hours = sim.time_min / 60.0
    windows = find_windows(run.actuation_active)
    dimmed_min = run.actuation_active.sum() * sim.step_s / 60.0

    fig, (ax_cri, ax_in) = plt.subplots(2, 1, figsize=(11, 8.5), dpi=120, sharex=True,
                                        gridspec_kw={"height_ratios": [1.2, 1.0]})

    ax_cri.plot(hours, run.cri, color="#1f77b4", lw=1.8, label="Collision Risk Index (CRI)")
    ax_cri.axhline(engine_cfg.threshold_on, color="#d62728", ls="--", lw=1.6,
                   label=f"Trigger threshold (CRI >= {engine_cfg.threshold_on:.2f})")
    ax_cri.axhline(engine_cfg.threshold_off, color="#7f7f7f", ls=":", lw=1.4,
                   label=f"Release threshold (CRI < {engine_cfg.threshold_off:.2f})")
    ax_cri.set_ylabel("CRI [0-1]")
    ax_cri.set_ylim(0.0, 1.0)
    ax_cri.set_title("Simulated Nocturnal CRI Dynamics and Automated Facade Dimming Windows",
                     fontsize=13, fontweight="bold", pad=10)

    for ax in (ax_cri, ax_in):
        for k, (s, e) in enumerate(windows):
            ax.axvspan(hours[s], hours[min(e, sim.n_steps - 1)], color="#ff7f0e", alpha=0.22, lw=0,
                       label=(f"BMS active dimming ({len(windows)} windows, {dimmed_min:.0f} min total)"
                              if (ax is ax_cri and k == 0) else None))
        ax.grid(True, ls=":", alpha=0.6)
    ax_cri.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_cri.text(0.995, 0.03, "CRI falls during dimming because the measured L_lux drops (closed loop);\n"
                "hysteresis keeps the protection active until risk subsides.",
                transform=ax_cri.transAxes, ha="right", va="bottom", fontsize=8.5, color="#444444")

    ax_in.plot(hours, sensors.d_radar, color="#2ca02c", lw=1.5, label=r"$D_{radar}$ (radar MTR)")
    ax_in.plot(hours, sensors.a_calls, color="#9467bd", lw=1.5, label=r"$A_{calls}$ (confirmed NFC)")
    ax_in.plot(hours, sensors.m_weather, color="#8c564b", lw=1.5, label=r"$M_{weather}$ (fog, visibility)")
    ax_in.plot(hours, run.l_lux, color="#e6b800", lw=1.5, label=r"$L_{lux}$ (facade illuminance)")
    spurious = np.flatnonzero(sensors.spurious_detections)
    if spurious.size:
        ax_in.scatter(hours[spurious], sensors.a_calls[spurious], marker="x", color="#d62728", s=28, zorder=5,
                      label="Spurious detection passing confidence gate")
    ax_in.set_ylabel("Normalised input [0-1]")
    ax_in.set_ylim(0.0, 1.3)
    ax_in.set_xlim(0.0, sim.duration_h)
    ax_in.set_xlabel("Time of night (UTC)")
    ax_in.legend(loc="upper center", ncol=3, framealpha=0.9, fontsize=8.5)

    ticks = np.arange(0, sim.duration_h + 1, 1)
    ax_in.set_xticks(ticks)
    ax_in.set_xticklabels([(sim.start_time + timedelta(hours=float(h))).strftime("%H:%M") for h in ticks])

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# 9. ENTRY POINT
# =============================================================================


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Collision Risk Engine simulation")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="output directory")
    parser.add_argument("--no-show", action="store_true", help="do not open a plot window")
    args = parser.parse_args(argv)

    sim = SimConfig(seed=args.seed)
    engine_cfg = EngineConfig()
    radar, weather, lighting = RadarModel(), WeatherModel(), LightingModel()
    acoustic, actuation, network = AcousticModel(), ActuationConfig(), NetworkModel()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sensors = generate_sensor_traces(sim, radar, weather, acoustic)
    run = run_engine(sim, sensors, engine_cfg, lighting, actuation)

    # Baseline for comparison: single threshold, no confirmation, no hold time.
    naive_cfg = EngineConfig(weights=engine_cfg.weights, threshold_on=engine_cfg.threshold_on,
                             threshold_off=engine_cfg.threshold_on, confirm_steps=1,
                             release_confirm_steps=1, min_hold_s=0.0)
    naive = run_engine(sim, sensors, naive_cfg, lighting, actuation)

    records = dispatch_commands(sim, sensors, run, engine_cfg, actuation, network)
    log_path = args.out_dir / "bms_mqtt_log.json"
    save_mqtt_log(records, sim, engine_cfg, log_path)

    print_report(sim, sensors, run, naive, records, engine_cfg, acoustic, weather, lighting, actuation)
    plot_results(sim, sensors, run, engine_cfg, args.out_dir / "cri_simulation_graph.png", show=not args.no_show)
    print(f"Artifacts written: {log_path}, {args.out_dir / 'cri_simulation_graph.png'}")


if __name__ == "__main__":
    main()
