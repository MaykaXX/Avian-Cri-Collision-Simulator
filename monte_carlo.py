#!/usr/bin/env python3
"""
Multi-seed robustness study for collision_risk_engine.py.

Runs the full simulated night for many random seeds (the scenario itself - pulse timing,
fog profile, radar envelope - is fixed; only stochastic noise, call arrivals, transients
and network latencies change) and summarises the distribution of key outcomes.

Reported metrics (definitions are assumptions of this study, not field measurements):
  * "pulse period"        - time within the generated migration pulses (envelope > 0.25),
                            extended by 10 min on each side.
  * "outside-pulse onset" - an actuation that starts outside every pulse period. NOTE: this
                            is NOT necessarily a false alarm, because the CRI also reacts
                            legitimately to peak radar traffic and fog between the pulses.
  * "noise-linked onset"  - an outside-pulse onset with a spurious detection (transient
                            misclassified as a call) in the preceding 90 s.
  * "short window"        - an actuation window shorter than 10 minutes (fragmentation).

Usage:
    python monte_carlo.py [--runs 100] [--out-dir .]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

import collision_risk_engine as cre


def true_risk_mask(sim: cre.SimConfig, acoustic: cre.AcousticModel, tolerance_min: float = 10.0) -> np.ndarray:
    """Boolean mask of steps inside a generated migration pulse, with tolerance."""
    t = sim.time_min
    pulse = np.zeros_like(t)
    for centre, width in acoustic.pulses:
        pulse += np.exp(-((t - centre) / width) ** 2)
    mask = np.clip(pulse, 0.0, 1.0) > 0.25
    return cre.dilate(mask, int(round(tolerance_min * 60.0 / sim.step_s)))


def run_one(seed: int, min_hold_s: float = 600.0, release_confirm_steps: int = 12) -> Dict[str, float]:
    sim = cre.SimConfig(seed=seed)
    engine_cfg = cre.EngineConfig(min_hold_s=min_hold_s, release_confirm_steps=release_confirm_steps)
    radar, weather, lighting = cre.RadarModel(), cre.WeatherModel(), cre.LightingModel()
    acoustic, actuation, network = cre.AcousticModel(), cre.ActuationConfig(), cre.NetworkModel()

    sensors = cre.generate_sensor_traces(sim, radar, weather, acoustic)
    run = cre.run_engine(sim, sensors, engine_cfg, lighting, actuation)
    naive_cfg = cre.EngineConfig(weights=engine_cfg.weights, threshold_on=engine_cfg.threshold_on,
                                 threshold_off=engine_cfg.threshold_on, confirm_steps=1,
                                 release_confirm_steps=1, min_hold_s=0.0)
    naive = cre.run_engine(sim, sensors, naive_cfg, lighting, actuation)
    records = cre.dispatch_commands(sim, sensors, run, engine_cfg, actuation, network)

    windows = cre.find_windows(run.actuation_active)
    truth = true_risk_mask(sim, acoustic)
    outside_onsets = [s for s, _ in windows if not truth[s]]
    noise_linked = sum(1 for s in outside_onsets if sensors.spurious_detections[max(0, s - 9):s + 1].any())
    short_windows = sum(1 for s, e in windows if (e - s) * sim.step_s < 600.0)
    burst_step = int(acoustic.stress_burst_min * 60.0 / sim.step_s)
    burst_triggered = any(burst_step - 6 <= s <= burst_step + 60 for s, _ in windows)
    e2e = [r["latency_ms"]["end_to_end_ms"] for r in records]
    in_truth_steps = truth.sum()
    covered = (run.actuation_active & truth).sum()

    return {
        "seed": seed,
        "windows": len(windows),
        "dimming_min": run.actuation_active.sum() * sim.step_s / 60.0,
        "peak_cri": float(run.cri.max()),
        "relay_transitions": len(run.decisions),
        "naive_transitions": len(naive.decisions),
        "outside_pulse_onsets": len(outside_onsets),
        "noise_linked_onsets": noise_linked,
        "short_windows": short_windows,
        "stress_burst_triggered": int(burst_triggered),
        "missed_risk_periods": int(len(true_pulse_periods(sim, acoustic)) - covered_periods(sim, acoustic, run.actuation_active)),
        "risk_coverage_pct": 100.0 * covered / in_truth_steps if in_truth_steps else float("nan"),
        "e2e_mean_ms": float(np.mean(e2e)) if e2e else float("nan"),
        "cpu_mean_ms": float(np.mean(run.cpu_latencies_ms)),
        "cpu_p99_ms": float(np.percentile(run.cpu_latencies_ms, 99)),
    }


def true_pulse_periods(sim: cre.SimConfig, acoustic: cre.AcousticModel) -> List[tuple]:
    """Each configured migration pulse as a (start_step, end_step) core period (|t - centre| < width)."""
    periods = []
    for centre, width in acoustic.pulses:
        periods.append((int((centre - width) * 60 / sim.step_s), int((centre + width) * 60 / sim.step_s)))
    return periods


def covered_periods(sim: cre.SimConfig, acoustic: cre.AcousticModel, active: np.ndarray) -> int:
    """Number of pulse core periods during which dimming was active at some point."""
    return sum(1 for s, e in true_pulse_periods(sim, acoustic) if active[s:e].any())


def summarise(values: np.ndarray) -> str:
    return (f"mean {np.nanmean(values):8.2f} | median {np.nanmedian(values):8.2f} | "
            f"min {np.nanmin(values):8.2f} | max {np.nanmax(values):8.2f} | "
            f"p5-p95 {np.nanpercentile(values, 5):.2f}-{np.nanpercentile(values, 95):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed study of the Collision Risk Engine")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--min-hold-s", type=float, default=600.0, help="engine minimum hold time (s)")
    parser.add_argument("--release-confirm-steps", type=int, default=12, help="engine release confirmation steps")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_one(seed, args.min_hold_s, args.release_confirm_steps) for seed in range(args.runs)]
    csv_path = args.out_dir / "monte_carlo_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    n = len(rows)
    line = "=" * 92
    print(line)
    print(f"  MULTI-SEED STUDY: {n} simulated nights (fixed scenario, random noise/arrivals/latencies)")
    print(f"  Engine: min hold {args.min_hold_s:.0f} s, release confirmation {args.release_confirm_steps} steps")
    print(line)
    print(f"Dimming windows per night   : {summarise(col('windows'))}")
    print(f"Total dimming time (min)    : {summarise(col('dimming_min'))}")
    print(f"Peak CRI                    : {summarise(col('peak_cri'))}")
    print(f"Relay transitions (engine)  : {summarise(col('relay_transitions'))}")
    print(f"Relay transitions (naive)   : {summarise(col('naive_transitions'))}")
    print(f"Coverage of true risk steps : {summarise(col('risk_coverage_pct'))} %")
    print(f"CPU time per update (ms)    : mean of means {col('cpu_mean_ms').mean():.4f}, worst p99 {col('cpu_p99_ms').max():.4f}")
    print(f"E2E latency model (ms)      : {summarise(col('e2e_mean_ms'))}")
    print("-" * 92)
    print(f"Outside-pulse onsets        : {summarise(col('outside_pulse_onsets'))}  (nights with >=1: {int((col('outside_pulse_onsets') > 0).sum())}/{n})")
    print(f"Noise-linked onsets (total) : {int(col('noise_linked_onsets').sum())} over {n} nights ({int((col('noise_linked_onsets') > 0).sum())} nights)")
    print(f"Short windows (< 10 min)    : {summarise(col('short_windows'))}  (nights with >=1: {int((col('short_windows') > 0).sum())}/{n})")
    print(f"Scripted noise-burst (min 45) triggered dimming: {int(col('stress_burst_triggered').sum())}/{n} nights")
    print(f"Nights where a migration pulse produced NO dimming (missed): "
          f"{int((col('missed_risk_periods') > 0).sum())}/{n}")
    print(line)
    print(f"Per-seed results written to: {csv_path}")


if __name__ == "__main__":
    main()
