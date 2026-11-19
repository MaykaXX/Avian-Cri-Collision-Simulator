# Bird Collision Risk Engine

A simulation prototype of an edge-to-building control loop that reduces nocturnal bird–building collisions: it fuses four sensor channels into a **Collision Risk Index (CRI)** and, when the risk is confirmed, publishes a command to a Building Management System (BMS) to dim facade lighting and enable acoustic deterrent emitters.

![CRI dynamics and sensor inputs](examples/cri_simulation_graph.png)

> **Scope.** This repository simulates the *decision logic* on **synthetic** sensor data generated from explicit physical models (dBA, kHz, lux, metres, birds/km/h). It is **not** a field validation, and it contains no data showing the effect of dimming or acoustic signals on real birds. All model constants are configurable assumptions, not measurements. See [Limitations](#limitations).

## How it works

```
CRI = w1·D_radar + w2·A_calls + w3·M_weather + w4·L_lux        (weights sum to 1)
```

| Channel | Meaning | Physical model → normalisation to [0, 1] |
|---|---|---|
| `D_radar` (0.15) | Radar-derived migration traffic rate | MTR in birds/km/h, linear, 1000 birds/km/h = 1.0 |
| `A_calls` (0.35) | Edge-confirmed nocturnal flight calls (NFC) | City noise 58 dBA; NFC 62–78 dBA at 2–9 kHz; SNR → classifier confidence; only confidence > 0.88 counts; evidence integrated over 90 s and saturated: `1 − exp(−E/E_ref)` |
| `M_weather` (0.30) | Fog hazard | Visibility in metres, logarithmic (3000 m → 0, 50 m → 1; 180 m ≈ 0.69) |
| `L_lux` (0.20) | Facade illuminance | Lux, linear on 600 lux (450 lux → 0.75). Measured in a **closed loop**: it drops to ≈ 112 lux while 75 % dimming is active |

**Decision logic** (`CollisionRiskEngine`): activate when `CRI ≥ 0.75` for 2 consecutive 10 s steps; release when `CRI < 0.48` for 12 consecutive steps *and* the protection has been active for at least 600 s. This hysteresis is needed because dimming itself lowers `L_lux` and therefore the CRI.

**Noise robustness:** sirens and vehicle pass-bys (Poisson-scheduled, plus one scripted worst-case burst) both mask real calls and are occasionally misclassified as calls. A single channel cannot reach the threshold on its own (even `A_calls = 1` with baseline inputs gives CRI ≈ 0.63), so a spurious acoustic detection must be corroborated by radar, weather and lighting conditions.

**Actuation:** each activation/release produces an MQTT-style JSON command (`facade_dimming_pct = 75`, acoustic emitters at 4–6 kHz) written to `bms_mqtt_log.json`, with a modelled delivery latency (ESP32-S3 inference < 45 ms + log-normal network RTT + BMS processing).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python collision_risk_engine.py --out-dir results          # one simulated night
python monte_carlo.py --runs 100 --out-dir results         # 100 nights, summary + CSV
python -m unittest discover -s tests -v                    # 19 tests
```

Options of `collision_risk_engine.py`: `--seed N` (default 42), `--out-dir DIR`, `--no-show` (do not open the plot window). Requires Python ≥ 3.10 (developed on 3.12; CI runs 3.10 and 3.12).

Outputs: `bms_mqtt_log.json`, `cri_simulation_graph.png` (and `monte_carlo_results.csv` for the multi-seed study). Reference outputs from the default configuration are committed in [`examples/`](examples/).

## Results of the default configuration

**Single night (seed 42, 8 h, 10 s steps)**

| Metric | Value |
|---|---|
| Dimming windows (counted from False→True transitions) | 2 (89 min and 97 min) |
| Total dimming time | 186 min of 480 min (facade at full output 61 % of the night) |
| Peak CRI | 0.774 |
| Relay transitions | 4 with hysteresis vs 558 for a single-threshold controller |
| Max CRI during spurious-detection bursts outside risk windows | 0.605 |
| Modelled MQTT/BMS end-to-end latency (mean, 4 commands) | 194 ms |
| CRI update time on the host CPU (mean / p99) | ≈ 0.001 ms / ≈ 0.003 ms (machine-dependent) |

**100 simulated nights** (fixed scenario; random noise, call arrivals, transients, latencies)

| Metric | Result |
|---|---|
| Dimming windows per night | median 2 (range 2–4) |
| Total dimming time | mean 176 min (range 152–206) |
| Peak CRI | 0.76–0.82 |
| Relay transitions | mean 4.4 vs 524 for the single-threshold controller |
| Scripted noise burst (minute 45) triggered dimming | 0 of 100 nights |
| Migration pulse with no dimming | 0 of 100 nights |
| Actuations starting outside the generated pulses (±10 min) | 35 of 100 nights had one; these occur between the pulses, when radar traffic and fog are at their maximum, so they are not necessarily false alarms |
| Noise-linked onsets (outside-pulse onset after a spurious detection) | 1 in 100 nights |

The CPU time measures the CRI update in CPython/NumPy on the host, **not** on the ESP32-S3 target.

## Project layout

```
collision_risk_engine.py   sensor models, normalisation, CRI engine, BMS/MQTT log, report, plot
monte_carlo.py             multi-seed robustness study
tests/test_engine.py       unit tests (unittest)
examples/                  reference outputs of the default configuration
.github/workflows/ci.yml   runs the tests on every push
```

All configuration lives in frozen dataclasses (`CriWeights`, `SimConfig`, `RadarModel`, `WeatherModel`, `LightingModel`, `AcousticModel`, `EngineConfig`, `ActuationConfig`, `NetworkModel`).

## Limitations

- **Synthetic data only.** The scenario (migration pulses at minutes 165 and 310 of the night, i.e. 23:45 and 02:10 UTC, plus the fog profile and radar envelope) was designed to exercise the logic and was tuned by hand. The number of dimming windows is a property of the scenario, not a finding about real migration.
- **Weights and thresholds are not calibrated** against collision or carcass-survey data.
- **No bird-behaviour model.** The effect of dimming or 4–6 kHz sound on flight paths is not simulated; the repository makes no claim about it.
- **Latencies are modelled** from assumed distributions (log-normal RTT with a 110 ms median, ESP32-S3 inference ≈ 30 ms, BMS processing ≈ 35 ms), not measured on a real link.
- The acoustic classifier is represented by a sigmoid confidence model, not by a trained neural network. The MTR scale (1000 birds/km/h) is an assumption.
- Two metrics are defined by this study (pulse periods, noise-linked onsets) and are explained in the docstring of `monte_carlo.py`.

## Background

Motivation and context: Loss et al. (2014), *The Condor* 116(1); Van Doren et al. (2017), *PNAS* 114(42); Horton et al. (2019), *Frontiers in Ecology and the Environment* 17(4); Boycott et al. (2021), *PLOS ONE* 16(4), e0249826.

## License

MIT — see [LICENSE](LICENSE). Replace `[Your Name]` in the license file with the copyright holder.
