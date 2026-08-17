# xaubot — XAUUSD machine-learning trading system

Production-oriented ML trading system for spot gold: 5m execution timeframe
with 15m/1h/4h/1d context, multi-task neural network, walk-forward validation,
event-driven backtesting, and a risk engine that can say *no*.

**Status: Phase 2 of 16.** Data ingestion and the feature engine work end to
end: 583 point-in-time-verified features across 5m/15m/1h/4h/1d. No labels, no
model, no signals, no execution yet. Live trading is not implemented and stays
that way until the evidence gates in `docs/ARCHITECTURE.md` §11 are met.

Design documents:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full system design, ML
  approach, schema, feature list, label design, validation and backtest
  methodology, risk architecture, roadmap
- [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — bar schema, timestamp
  semantics, and what was verified about the current dataset

---

## Setup

```bash
py -3.13 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Python 3.13 specifically: PyTorch wheel availability for 3.14 is not something
to build on. Copy `.env.example` to `.env` for local paths and (later) broker
credentials — secrets are read from the environment only, never from YAML.

## Usage

Ingest, validate, resample, and store historical bars:

```bash
.venv/Scripts/xaubot.exe data ingest -c config/source_strategy_csv.yaml
```

Inspect what is in the bar store:

```bash
.venv/Scripts/xaubot.exe data summary
```

Build the feature matrix (runs the point-in-time audit before writing):

```bash
.venv/Scripts/xaubot.exe features build -c config/source_strategy_csv.yaml
```

Run the point-in-time audits on their own — the static look-ahead scan plus
replay equivalence:

```bash
.venv/Scripts/xaubot.exe features audit -c config/source_strategy_csv.yaml
```

Resolve and print the effective configuration with its content hash:

```bash
.venv/Scripts/xaubot.exe config-show
```

Point it at a different file without editing YAML:

```bash
.venv/Scripts/xaubot.exe data ingest -s data.source.path=D:/data/xau.csv -s data.source.timestamp_convention=close
```

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

Run only the point-in-time guards — the tests that matter most, since a failure
there invalidates every downstream result:

```bash
.venv/Scripts/python.exe -m pytest -m leakage -v
```

## Design commitments

These are the non-negotiable ones. The reasoning is in `docs/ARCHITECTURE.md`.

1. **Point-in-time correctness is structural.** Bars carry both `open_time` and
   `close_time`; features at time `t` read only rows with `close_time <= t`;
   higher-timeframe context is joined with `merge_asof` on `close_time`; orders
   fill at the *next* bar's open. Enforced by `tests/leakage/`.
2. **The neural network must beat a LightGBM baseline** on out-of-sample
   expectancy after realistic costs, or the report says so and the simpler
   model ships.
3. **Ambiguous outcomes resolve pessimistically.** When a bar touches both stop
   and target, the stop wins — in label generation *and* in the backtester.
4. **Nothing is fabricated.** Missing bars are not forward-filled by default;
   a synthetic bar is an observation the model would learn from as if it were
   real.
5. **The test split is used once.** If a test result motivates a change, that
   change is a new experiment and the old result stays in the record.
6. **Live and backtest features come from one implementation.** The streaming
   driver replays the batch transforms over a trailing buffer, so parity is a
   testable property — `transform(history)[t] == transform(history.as_of(t))[-1]`
   — rather than something maintained by discipline.

## Layout

```
config/     YAML configuration (no secrets)
docs/       Design documents
src/xaubot/
  config/   Pydantic schema, layered loading, content-addressed hashing
  core/     Enums, domain types, time/session/calendar utilities
  data/     Loaders, validators, cleaning, resampling, Parquet store
  features/ Transforms, MTF alignment, manifest, engine, PIT audit
  cli.py    Typer CLI
tests/
  unit/         Per-module tests
  leakage/      Point-in-time guards
  integration/  End-to-end pipeline tests
```

Dependencies flow strictly downward through the layer stack; `features/` may
never import from `models/`, `signals/`, `risk/`, or `backtesting/`.
