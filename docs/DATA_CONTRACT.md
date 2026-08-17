# Data Contract

What the system assumes about input bars, and what it verified about the
dataset currently configured.

---

## 1. Canonical bar schema

Every bar, at every timeframe, carries:

| Column | Type | Meaning |
|---|---|---|
| `open_time` | datetime64[UTC] | Start of the bar's interval (inclusive) |
| `close_time` | datetime64[UTC] | End of the bar's interval (exclusive) — **this is decision time** |
| `open` `high` `low` `close` | float64 | Prices in quote currency |
| `volume` | float64 | Activity proxy (see §4) |

Invariants enforced by `BarFrame` on construction:

- index is tz-aware UTC, unique, monotonically increasing, and equal to `close_time`
- `close_time - open_time == timeframe.duration`
  (exception: daily bars anchored to a local exchange close are 23h/25h on the
  two DST transition days per year — that is correct, not corruption)
- OHLCV columns present and numeric

## 2. The timestamp convention is configuration, never inference

`data.source.timestamp_convention` must be `open` or `close`. Assuming the
wrong one shifts every feature by exactly one bar, which is a look-ahead bias
that produces excellent-looking backtests and no downstream test can detect.

**For the currently configured file, `open` is correct**, evidenced by the
bar distribution: the first bar of each session day opens at exactly 01:00 UTC
and the last bar of each Friday opens at 23:55 UTC. Under a close-stamping
convention those boundaries would be 01:05 and 00:00 instead.

## 3. Execution lag

- Features at decision time `t` may read only rows with `close_time <= t`.
- Orders decided at `t` fill at the **open of the next 5m bar** (`t + 5m`),
  with spread and slippage applied.

There is no path in the system that decides and fills on the same bar.

## 4. Volume semantics

`data.source.volume_semantics` records what the number means. MT5 XAUUSD
"volume" is a **tick count**, not traded contracts — a usable activity proxy,
but broker-specific and not comparable across feeds. Cross-feed robustness
tests should disable volume features rather than assume they transfer.

## 5. The trading calendar is a property of the feed, not of the instrument

Gap detection is meaningless without knowing when the market is supposed to be
open. A naive 24/7 grid reports ~25% of every week as missing and buries the
outages that matter.

Two calendar shapes appear in practice:

- **Broker feeds** (MT5 and friends): anchored to a local exchange close, so
  the schedule shifts by an hour in UTC across DST. `config/base.yaml` models
  this: week opens Sunday 18:00 New York, closes Friday 17:00 New York, daily
  maintenance break 17:00-18:00 New York.
- **Vendor feeds**: often anchored to fixed UTC hours with no DST shift.

Using the wrong one does not corrupt data, but it makes the completeness
number meaningless.

---

## 6. Verified dataset: `D:/Strategy/data/xauusd_5m.csv`

Overlay: `config/source_strategy_csv.yaml` · SHA-256 `4136c8b75b9e55cc…`

| Property | Value |
|---|---|
| Rows in / out | 49,998 / 49,998 (zero dropped) |
| Range | 2025-11-25 08:45 UTC → 2026-08-12 09:15 UTC (~8.5 months) |
| Format | `datetime,open,high,low,close,volume`, ISO-8601 UTC, comma-delimited |
| Timestamp convention | `open` (evidence in §2) |
| Duplicates / null / OHLC-inconsistent / off-grid | 0 |
| **Completeness** | **98.62%** (697 missing of 50,515 expected) |

### Derived schedule

Recovered empirically from the missing-bar distribution, not assumed:

- trading week opens **Monday 01:00 UTC**, closes **Saturday 00:00 UTC**
- daily maintenance break **00:00–01:00 UTC**, fixed year-round (no DST shift)
- full-day holidays: 2025-12-25, 2026-01-01, 2026-04-03 (Good Friday)

This is a **fixed-UTC vendor schedule, not a broker feed.** A broker feed opens
Sunday evening and shifts with DST. Against the default NY-anchored calendar
this file scores 92.14% completeness; against its own schedule, 98.62%.

### Residual gaps are all genuine US half-days

Thanksgiving (2025-11-27), Christmas Eve, New Year's Eve, MLK Day (2026-01-19),
Presidents' Day (2026-02-16), Memorial Day (2026-05-25), Juneteenth
(2026-06-19), July 3rd — plus late opens on 2025-12-26 and 2026-01-02. Early
closes cluster at 21:25 UTC, which is the expected 18:00 New York early close.

### Extreme returns: 86 bars beyond 12 robust sigma

Inspected, and they are real news spikes rather than bad prints:

- robust sigma is 0.0667% per 5m bar; the largest move is +2.87%
  (2026-03-23 13:10, 4282 → 4407)
- all carry elevated volume (2,000–3,400 vs ~700 typical) and wide intrabar
  ranges; **zero** are zero-range prints or pure gaps
- 16 of 86 fall in the 01:00 UTC hour — the first bar after the daily break,
  where an overnight move is compressed into one bar

**Implications for later phases:** the first bar after any gap must be excluded
from label generation and flagged as a feature (`bars_since_gap`), because its
return spans a closed market rather than five minutes of trading.

### Fitness for purpose

Clean enough to build and validate the pipeline on. **Not sufficient to train a
model that should be trusted**: ~8.5 months and 50k bars covers roughly one
volatility regime. The walk-forward schedule in ARCHITECTURE.md §7.2 assumes
2021-2026. Note also that this period contains an extraordinary gold range
(≈4,000 → 5,600 → 4,000), which makes ATR/percentage normalisation of every
price feature mandatory rather than merely advisable.

**Needed before Phase 5 (baselines):** 3+ years of 5m XAUUSD from the venue
intended for execution, so that spread behaviour, session structure, and the
trading calendar in training match the calendar in live trading.
