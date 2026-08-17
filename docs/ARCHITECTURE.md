# XAUUSD ML Trading System — Architecture & Design Specification

**Status:** Design (pre-implementation)
**Instrument:** XAUUSD | **Execution TF:** 5m | **Context TFs:** 15m, 1h, 4h, 1d
**Target platform:** Windows 11, Python 3.13, PyTorch (CUDA 12.x, RTX 2060 6 GB), CPU fallback

---

## 0. Design stance (read this first)

Three assumptions drive every decision below.

1. **The dominant failure mode of this class of project is not a weak model — it is leakage.** A 5m gold model that shows 68% accuracy in a notebook is, in ~95% of cases, leaking through one of six channels (Section 17). The architecture therefore treats point-in-time (PIT) correctness as a *structural* property enforced by the code layout, not as a discipline the developer is asked to remember.

2. **Predictable edge at 5m on XAUUSD is small and intermittent.** Realistic per-trade edge after costs is on the order of 2–8 bps of notional. XAUUSD spread on a retail ECN account is ~15–25 cents (1.5–2.5 pips of $0.01) plus commission ~$7/lot round turn, which on a 100 oz lot is ~$25–35 round-trip cost. The system must therefore be *selective*: `WAIT` is the correct output the large majority of the time, and label design (Section 5) must reflect net-of-cost outcomes, not raw direction.

3. **The neural net must earn its place.** Every NN result is reported alongside four baselines (buy & hold, random with matched trade count, EMA cross, ATR-normalized momentum) and one strong tabular baseline (LightGBM on the same features). If the NN does not beat LightGBM on out-of-sample risk-adjusted expectancy after costs, the system reports that plainly and we ship the gradient-boosted model, or nothing.

---

## 1. Complete Architecture

### 1.1 Layer diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L0  SOURCES         CSV (MT5 export) │ future: broker REST/WS, tick files     │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  RawBar records
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L1  INGESTION      schema coercion → tz normalization → dedupe → gap scan →   │
│     (data/)        monotonic sort → sanity filters → session tagging          │
│                    OUT: canonical 5m bar table (Parquet, immutable, versioned)│
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  BarFrame(5m, canonical)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L2  RESAMPLING     5m → 15m/1h/4h/1d via label='left', closed='left'          │
│     (data/)        every HTF bar carries explicit open_time AND close_time    │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  dict[TF -> BarFrame]
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L3  FEATURE ENGINE Per-TF causal transforms (no centering, no future window)  │
│     (features/)    → AS-OF MERGE onto 5m grid keyed on HTF.close_time <= t    │
│                    → cross-TF derived features → PIT assertion pass           │
│                    OUT: feature matrix X (float32) + feature manifest (hash)  │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  X [T, F]
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L4  LABEL ENGINE   triple-barrier (net of cost) │ horizon returns │ MFE/MAE   │
│     (labels/)      │ regime │ meta-labels │ sample weights (uniqueness decay) │
│                    OUT: Y [T, K] + label_end_index[T]  (needed for purging)   │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  (X, Y, w, t)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L5  DATASET        walk-forward fold generator → purge + embargo →            │
│     (datasets/)    per-fold scaler FIT ON TRAIN ONLY → sequence windowing     │
│                    → torch Dataset/DataLoader (no shuffling across folds)     │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  Fold(train/val/calib/test)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L6  MODELS         registry: lgbm │ mlp │ gru │ lstm │ tcn │ tcn_gru │ xfmr   │
│     (models/)      multi-task heads, shared encoder                          │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L7  TRAINING       deterministic seeding → train loop (AMP) → early stop on   │
│     (training/)    val trading metric → Optuna (val only) → calibration on    │
│                    held-out calib slice → artifact write (never overwrite)    │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  ModelArtifact(version, weights, scaler, manifest, metrics)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L8  INFERENCE      Predictor: X_t → raw logits → calibrated probs → ModelOut  │
│     (inference/)   identical code path offline (backtest) and online (live)   │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  ModelOutput
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L9  SIGNAL ENGINE  gate chain (prob, edge, RR, regime, structure, vol, spread)│
│     (signals/)     → SL placement → TP ladder → confidence 0..100 → reasons   │
│                    OUT: Signal | None                                        │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  Signal
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L10 RISK ENGINE    pre-trade gates + portfolio state machine + position size  │
│     (risk/)        OUT: Order | Rejection(reason)                            │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  Order
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L11 EXECUTION      ExecutionAdapter (ABC)                                     │
│     (execution/)   ├── BacktestBroker  (event-driven, bar-level, pessimistic) │
│                    ├── PaperBroker     (live feed, simulated fills)           │
│                    └── LiveBroker      (NOT IMPLEMENTED until gates pass)     │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  Fill / PositionUpdate
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ L12 ANALYTICS      trade ledger → metrics (global + sliced) → reports         │
│     (evaluation/)                                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ L13 SERVICE+UI     FastAPI (/predict /signal /health /metrics) │ Dash/Plotly  │
│     (service/, dashboard/)                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The single most important architectural rule

**One `FeatureEngine`, two drivers.** The batch driver replays the whole history; the streaming driver feeds one bar at a time. Both call the *same* transform functions. A nightly parity test asserts:

```
batch_features(history)[-1]  ==  streaming_features(replay(history))[-1]   (atol=1e-6)
```

This single test kills the most common live-vs-backtest divergence class. If a feature cannot be computed identically in both modes, it does not enter the feature set.

### 1.3 Timestamp contract (non-negotiable)

Every bar row carries **both** `open_time` and `close_time`, tz-aware UTC.
- A broker CSV timestamp is assumed to be **open time** unless the loader config says otherwise; `close_time = open_time + tf_duration`.
- **Decision time** for a 5m bar is its `close_time`.
- Features at decision time `t` may use only rows with `close_time <= t`.
- **Execution** happens at the *open of the next 5m bar*, i.e. `t + 5m`, with spread and slippage applied. There is always a one-bar decision→fill lag. No exceptions.

### 1.4 Multi-timeframe as-of alignment (worked example)

Decision at `t = 10:15:00` (close of the 10:10–10:15 5m bar).

| TF | Eligible latest bar | Reason |
|---|---|---|
| 15m | 10:00–10:15, close 10:15 | closes exactly at `t` → **allowed** (`allow_exact_matches=True`) |
| 15m | 10:15–10:30 | closes 10:30 > `t` → **forbidden** (this is the classic leak) |
| 1h  | 09:00–10:00, close 10:00 | latest completed |
| 4h  | 08:00–12:00 | **forbidden**; use 04:00–08:00 |
| 1d  | previous day | today's daily bar is incomplete |

Implementation: compute HTF features on the HTF frame, then

```python
pd.merge_asof(
    base_5m.sort_values("close_time"),
    htf_feats.sort_values("close_time"),   # HTF feature row stamped at ITS close_time
    on="close_time", direction="backward", allow_exact_matches=True,
    suffixes=("", f"_{tf}"),
)
```

Intra-bar HTF progress (e.g. "how far into the current 4h bar are we", "current 4h bar's running high") is **allowed and valuable**, but must be built from completed 5m bars only — never read off the HTF frame. These get their own explicit `*_running_*` names so the distinction is visible in the manifest.

---

## 2. Recommended Neural-Network Architecture

### 2.1 Why not a Temporal Fusion Transformer on day one

TFT is ~5–10× the parameters and needs a strong signal to justify variable-selection + interpretable attention. On 5m XAUUSD the label noise is enormous (the Bayes error for 12-bar direction is probably ≥45%). Attention will happily fit noise. Sequencing: **prove the pipeline with cheap models, then escalate.**

### 2.2 Model ladder (implemented in this order)

| # | Model | Role | Params |
|---|---|---|---|
| 0 | `MajorityWait` / `LogReg` on 12 features | sanity floor | ~10² |
| 1 | **LightGBM** on flat features (last bar + rolling aggregates) | the benchmark the NN must beat | — |
| 2 | `MLP` on flattened window | tests whether sequence structure matters at all | ~10⁵ |
| 3 | **`TCN_GRU`** ← *primary recommendation* | production candidate | ~3–8 × 10⁵ |
| 4 | `GRU`, `LSTM`, `TCN` (pure) | ablations | — |
| 5 | `CausalTransformer` | escalation | ~10⁶ |
| 6 | `TFT-lite` | only if 3–5 show real edge | ~10⁶ |

### 2.3 Primary architecture: `TCN_GRU` multi-task

```
INPUTS
  x_seq   [B, L, F_dyn]    L ∈ {48,96,192,288}, F_dyn ≈ 180  (5m features + as-of HTF features)
  x_stat  [B, F_stat]      F_stat ≈ 25  (session one-hot, cyclical time, regime one-hot, day-of-week)

ENCODER
  InputProj:   Linear(F_dyn → d_model=128) + LayerNorm
  TCN stack:   4 residual blocks, dilations [1,2,4,8], kernel 5, CAUSAL padding only
               receptive field = 1 + 2*(5-1)*(1+2+4+8) = 121 bars  ≥ L=96 ✓
               each block: WeightNormConv1d → GELU → Dropout → Conv1d → residual → LN
  GRU:         2 layers, hidden 128, unidirectional (bidirectional = leakage, banned)
  Pooling:     additive attention over the last min(L,32) steps  +  last hidden state
               (concat → 256)  — attention weights are stored for explainability
  FiLM cond.:  static features → (γ, β) → h = γ ⊙ h + β        ← regime conditioning
               This is how Section 8's requirement ("condition predictions on regime")
               is met inside the network rather than bolted on afterwards.

HEADS (all from shared h)
  direction   Linear(256→3)      → logits  P(BUY) P(SELL) P(WAIT)     CE + class weights
  ret_q       Linear(256→3)      → r@[0.1,0.5,0.9] in ATR units       pinball loss
  vol         Linear(256→1)      → log σ_fwd                           Gaussian NLL
  barrier     Linear(256→4)      → P(TP1) P(TP2) P(TP3) P(SL)          BCE (multi-label)
  mfe_mae     Linear(256→2)      → E[MFE], E[MAE] in ATR units         Huber
  regime      Linear(256→7)      → regime logits (auxiliary)           CE, weight 0.1
```

**Total loss** `L = Σ wᵢ·Lᵢ` with weights either fixed (start: 1.0 / 0.5 / 0.2 / 0.7 / 0.2 / 0.1) or learned via Kendall homoscedastic uncertainty weighting (`L = Σ (1/2σᵢ²)Lᵢ + log σᵢ`). Fixed first — learned weighting is another thing that can silently go wrong.

**Design notes**
- Everything is causal. `bidirectional=True`, non-causal conv padding, and `BatchNorm` over the time axis are all banned and asserted against in `tests/test_causality.py` (gradient test: perturb `x[:, t+1:, :]`, assert `∂output/∂x == 0`).
- Targets are expressed in **ATR units**, not price units. Gold at $1,800 and gold at $4,600 must produce the same feature/target distributions.
- Multi-task is not decoration: the barrier and MFE/MAE heads are what let the signal engine size SL/TP and compute expected R without a second model.

### 2.4 Sequence & scale budget

With L=96, F=180, d=128, batch 256: activation memory ≈ 250 MB — comfortable on 6 GB with AMP (`torch.amp.autocast('cuda', dtype=torch.float16)`). ~4 years of 5m data ≈ 300k bars ≈ 1.2 GB float32 for X; keep the full matrix in RAM and window on the fly (`as_strided` view, zero-copy) rather than materializing L× the data.

---

## 3. Database / Schema Design

**Two stores, deliberately.** Bulk time series → **Parquet** (columnar, compressed, memory-mappable, no server). Metadata, experiments, and trades → **SQLite** in dev / **PostgreSQL** in prod, behind one SQLAlchemy layer. **DuckDB** is used for ad-hoc analytical queries directly over the Parquet lake.

### 3.1 Parquet lake layout

```
data/
  raw/           xauusd_5m_2019_2026.csv                  (immutable, hashed)
  canonical/     symbol=XAUUSD/tf=5m/year=2024/month=03/part-0.parquet
  resampled/     symbol=XAUUSD/tf=1h/year=2024/part-0.parquet
  features/      fs_version=fs_2026_08_a/symbol=XAUUSD/year=2024/month=03/*.parquet
  labels/        ls_version=ls_2026_08_a/horizon=12/*.parquet
  datasets/      ds_<hash>/fold_00/{train,val,calib,test}.npz + scaler.pkl
```

`fs_version` is the hash of the resolved feature-config; changing any feature parameter produces a new directory. Feature sets are never mutated in place.

### 3.2 Relational schema (DDL)

```sql
-- ── Reference ────────────────────────────────────────────────────────────────
CREATE TABLE instrument (
  symbol            TEXT PRIMARY KEY,          -- 'XAUUSD'
  contract_size     REAL NOT NULL,             -- 100 (oz per 1.0 lot)
  tick_size         REAL NOT NULL,             -- 0.01
  tick_value        REAL NOT NULL,             -- 1.00 USD per tick per lot
  min_lot           REAL NOT NULL,             -- 0.01
  lot_step          REAL NOT NULL,             -- 0.01
  max_lot           REAL NOT NULL,
  margin_pct        REAL NOT NULL,
  commission_per_lot REAL NOT NULL,            -- USD round turn
  typical_spread    REAL NOT NULL,             -- price units
  quote_ccy         TEXT NOT NULL DEFAULT 'USD'
);

-- ── Data lineage ─────────────────────────────────────────────────────────────
CREATE TABLE dataset_version (
  dataset_id     TEXT PRIMARY KEY,             -- content hash
  symbol         TEXT REFERENCES instrument(symbol),
  source_files   JSON NOT NULL,                -- [{path, sha256, rows}]
  tf_base        TEXT NOT NULL,
  row_count      INTEGER NOT NULL,
  ts_start       TIMESTAMPTZ NOT NULL,
  ts_end         TIMESTAMPTZ NOT NULL,
  gap_report     JSON NOT NULL,                -- {expected, present, missing, weekend_excl}
  fs_version     TEXT,
  ls_version     TEXT,
  created_at     TIMESTAMPTZ NOT NULL
);

CREATE TABLE feature_manifest (
  fs_version   TEXT PRIMARY KEY,
  config_yaml  TEXT NOT NULL,
  features     JSON NOT NULL,   -- [{name, group, tf, dtype, lookback_bars, pit_note}]
  n_features   INTEGER NOT NULL,
  max_lookback INTEGER NOT NULL,  -- drives purge width
  created_at   TIMESTAMPTZ NOT NULL
);

-- ── Experiments ──────────────────────────────────────────────────────────────
CREATE TABLE experiment (
  exp_id       TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  git_sha      TEXT NOT NULL,
  git_dirty    BOOLEAN NOT NULL,
  config_yaml  TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE run (                       -- one walk-forward fold of one model
  run_id       TEXT PRIMARY KEY,
  exp_id       TEXT REFERENCES experiment(exp_id),
  fold_idx     INTEGER NOT NULL,
  model_type   TEXT NOT NULL,
  seed         INTEGER NOT NULL,
  dataset_id   TEXT REFERENCES dataset_version(dataset_id),
  fs_version   TEXT REFERENCES feature_manifest(fs_version),
  hparams      JSON NOT NULL,
  train_start  TIMESTAMPTZ, train_end TIMESTAMPTZ,
  val_start    TIMESTAMPTZ, val_end   TIMESTAMPTZ,
  calib_start  TIMESTAMPTZ, calib_end TIMESTAMPTZ,
  test_start   TIMESTAMPTZ, test_end  TIMESTAMPTZ,
  purge_bars   INTEGER NOT NULL, embargo_bars INTEGER NOT NULL,
  status       TEXT NOT NULL,            -- running|done|failed
  duration_s   REAL, device TEXT,
  artifact_dir TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL,
  UNIQUE(exp_id, fold_idx, model_type, seed)
);

CREATE TABLE run_metric (
  run_id  TEXT REFERENCES run(run_id),
  split   TEXT NOT NULL,                 -- train|val|calib|test
  metric  TEXT NOT NULL,                 -- auc|brier|ece|sharpe|profit_factor|...
  value   REAL NOT NULL,
  epoch   INTEGER,                       -- NULL = final
  PRIMARY KEY (run_id, split, metric, epoch)
);

CREATE TABLE model_artifact (
  model_id     TEXT PRIMARY KEY,
  run_id       TEXT REFERENCES run(run_id),
  version      TEXT NOT NULL,            -- xauusd_5m_tcngru_v0007_fold03
  weights_path TEXT NOT NULL, weights_sha256 TEXT NOT NULL,
  scaler_path  TEXT NOT NULL,
  calibrator_path TEXT,
  onnx_path    TEXT,
  promoted     BOOLEAN NOT NULL DEFAULT FALSE,
  created_at   TIMESTAMPTZ NOT NULL
);

-- ── Trading ──────────────────────────────────────────────────────────────────
CREATE TABLE signal (
  signal_id   TEXT PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL,      -- decision time (bar close)
  symbol      TEXT NOT NULL, model_id TEXT REFERENCES model_artifact(model_id),
  action      TEXT NOT NULL,             -- BUY|SELL|WAIT
  confidence  REAL NOT NULL,
  p_buy REAL, p_sell REAL, p_wait REAL,
  expected_return REAL, expected_vol REAL,
  p_tp1 REAL, p_tp2 REAL, p_tp3 REAL, p_sl REAL,
  regime      TEXT NOT NULL,
  entry REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL,
  risk_reward REAL, atr REAL, session TEXT,
  gates_passed JSON, reject_reason TEXT,
  explain     JSON,                      -- top-k SHAP contributions
  UNIQUE(ts, symbol, model_id)
);

CREATE TABLE trade (
  trade_id     TEXT PRIMARY KEY,
  signal_id    TEXT REFERENCES signal(signal_id),
  mode         TEXT NOT NULL,            -- backtest|paper|live
  session_run  TEXT NOT NULL,
  side         TEXT NOT NULL,
  entry_ts TIMESTAMPTZ, entry_price REAL, requested_price REAL, slippage REAL,
  size_lots REAL, risk_amount REAL, risk_pct REAL,
  sl_price REAL, tp1_price REAL, tp2_price REAL, tp3_price REAL,
  exit_ts TIMESTAMPTZ, exit_price REAL, exit_reason TEXT,  -- SL|TP1..3|TRAIL|TIME|EOD|MANUAL
  gross_pnl REAL, commission REAL, spread_cost REAL, swap REAL, net_pnl REAL,
  r_multiple REAL, mfe_r REAL, mae_r REAL, bars_held INTEGER,
  regime TEXT, session TEXT, hour INTEGER, dow INTEGER, vol_regime TEXT
);

CREATE TABLE equity_point (
  session_run TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
  equity REAL NOT NULL, balance REAL NOT NULL,
  open_risk REAL NOT NULL, drawdown REAL NOT NULL,
  PRIMARY KEY (session_run, ts)
);

CREATE TABLE risk_event (
  id INTEGER PRIMARY KEY, session_run TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
  event TEXT NOT NULL,                   -- DAILY_LOSS_HALT|CONSEC_LOSS_HALT|DD_HALT|...
  detail JSON NOT NULL
);
```

Indices: `trade(entry_ts)`, `trade(session_run, entry_ts)`, `signal(ts)`, `run_metric(run_id, split)`, `equity_point(session_run, ts)`.

---

## 4. Feature List

Grouped, with the PIT rule for each group. Names below are the canonical column names. Every feature exists at 5m; the ones marked **⟳** are additionally computed on 15m/1h/4h/1d and as-of merged with a `_15m`/`_1h`/`_4h`/`_1d` suffix.

### 4.1 Price / candle (all ATR- or ratio-normalized) ⟳
```
open_ret          log(open_t / close_{t-1})
high_ret          log(high_t / close_{t-1})
low_ret           log(low_t  / close_{t-1})
close_ret         log(close_t/ close_{t-1})
body_ret          (close-open)/close
range_ret         (high-low)/close
upper_wick        (high - max(open,close)) / (high-low+eps)
lower_wick        (min(open,close) - low) / (high-low+eps)
body_to_range     |close-open| / (high-low+eps)
close_position    (close-low)/(high-low+eps)                 ∈[0,1]
high_low_range_atr (high-low)/ATR14
open_close_range_atr |close-open|/ATR14
gap_atr           (open_t - close_{t-1})/ATR14
ret_{1,2,3,5,10,20,50}_atr    cumulative return over k bars / (ATR14*sqrt(k))
zscore_close_{20,50,100}      (close - SMA_n)/std_n
```

### 4.2 Volatility ⟳
```
atr_14, atr_50                        Wilder
atr_pct                               atr_14 / close
atr_percentile_{252,1000}             rank of atr_14 in trailing window ∈[0,1]
atr_ratio_fast_slow                   atr_14 / atr_50
realized_vol_{20,50}                  std of log returns * sqrt(bars_per_day)
parkinson_vol_20                      high/low estimator
garman_klass_vol_20
vol_of_vol_50                         std(realized_vol_20)
range_expansion                       range_t / mean(range, 20)
range_compression                     mean(range,5)/mean(range,50)
nr7, wr7                              narrowest/widest range of last 7 (0/1)
bb_width_20                           (upper-lower)/mid
bb_width_percentile_252
squeeze_flag                          bb_width < keltner_width
vol_regime                            categorical from atr_percentile: LOW/NORMAL/HIGH/EXTREME
vol_regime_onehot_{4}
```

### 4.3 Volume ⟳
```
volume_ma_{20,50}
volume_ratio                          volume / volume_ma_20
relative_volume_tod                   volume / median volume at THIS time-of-day over trailing 20 days
                                      (critical: raw volume ratio is dominated by session effects)
volume_zscore_50
volume_expansion                      volume_ma_5 / volume_ma_50
volume_compression                    1 / volume_expansion
buy_pressure                          ((close-low)-(high-close))/(high-low+eps)   ∈[-1,1]
volume_weighted_buy_pressure          buy_pressure * volume_ratio
cvd_proxy_{20,50}                     rolling sum of buy_pressure*volume, z-scored
obv_slope_20                          normalized OBV slope
vwap_session, dist_vwap_atr           session VWAP + distance in ATR
```
> Note: MT5 XAUUSD "volume" is **tick count**, not traded contracts. It is a usable activity proxy but is broker-specific; the feature manifest records `volume_semantics: tick_count` and the config can disable volume features entirely for cross-broker robustness testing.

### 4.4 Trend ⟳
```
ema_{20,50,100,200}                   never fed raw
dist_ema_{20,50,100,200}_atr          (close - ema_n)/atr_14
ema_slope_{20,50,100,200}             (ema_n - ema_n[-5]) / (5*atr_14)
ema_stack_score                       ∈[-1,1]; +1 = 20>50>100>200
ema_20_50_spread_atr, ema_50_200_spread_atr
adx_14, di_plus_14, di_minus_14, di_diff (=DI+ − DI−)/(DI+ + DI−)
adx_slope_5
macd_hist_atr                         MACD histogram / atr_14
rsi_14, rsi_slope_5, rsi_divergence_20
stoch_k_14, stoch_d_3
cci_20_norm, williams_r_14
linreg_slope_{20,50}_atr, linreg_r2_{20,50}
hurst_100                             persistence estimate
efficiency_ratio_20                   |net move| / sum|moves|   (Kaufman)
```

### 4.5 Market structure
```
swing_high_flag, swing_low_flag       fractal, k=2 bars each side
                                      ⚠ PIT: a swing at bar i is only KNOWN at bar i+k.
                                      Implementation shifts confirmation by k. Enforced.
last_swing_high_dist_atr, last_swing_low_dist_atr
last_swing_high_age, last_swing_low_age        (bars)
hh, hl, lh, ll                        booleans on confirmed swings
structure_direction                   ∈{-1,0,+1}
structure_strength                    ∈[0,1] = consecutive consistent swings, decayed
bos_bull, bos_bear                    break of structure (close beyond last confirmed swing)
bos_strength                          break distance / atr_14, clipped to [0,1]
bos_age
mss_bull, mss_bear                    market-structure shift (BOS against prior direction)
mss_strength, mss_age
range_high, range_low, range_width_atr, range_position ∈[0,1]
consolidation_score                   ∈[0,1]
```

### 4.6 Liquidity levels & sweeps
```
prev_day_high/low, prev_week_high/low
asia_high/low, london_high/low, ny_high/low          (previous completed session)
curr_session_high/low                                (running, from completed 5m bars only)
dist_<level>_atr                                     for all 14 levels above
nearest_level_above_atr, nearest_level_below_atr, level_cluster_density
sweep_high, sweep_low                 wick pierces level then closes back inside
sweep_strength                        pierce_depth/atr * close_back_fraction   ∈[0,1]
sweep_age, sweep_level_type           which level was swept (one-hot)
stop_hunt_score                       sweep + immediate opposite displacement  ∈[0,1]
rejection_after_sweep                 ∈[0,1]
equal_highs_count_50, equal_lows_count_50            (liquidity pools)
```

### 4.7 Fair value gaps
```
For up to N=3 nearest unfilled FVGs, each side:
fvg_bull_exists, fvg_bull_size_atr, fvg_bull_dist_atr, fvg_bull_age,
fvg_bull_fill_pct, fvg_bull_strength
fvg_bear_* (same)
fvg_count_bull_50, fvg_count_bear_50, fvg_imbalance
```
FVG detection: 3-bar pattern where `low[i] > high[i-2]` (bullish) — **only confirmed at bar i**, so all FVG features are stamped at `i`, never at `i-1`.

### 4.8 Order blocks
```
ob_bull_exists, ob_bull_dist_atr, ob_bull_size_atr, ob_bull_age,
ob_bull_strength      (displacement after OB / atr, × volume_ratio at OB, clipped)
ob_bull_mitigated, ob_bull_mitigation_pct
ob_bear_* (same)
ob_confluence_fvg     OB overlaps an FVG ∈[0,1]
```

### 4.9 Candle patterns — continuous strength ∈[0,1]
```
engulf_strength         body_t/body_{t-1} ratio × direction agreement × close_position, squashed
pinbar_strength         wick/range × body smallness × location vs level
rejection_strength      wick beyond a liquidity level + close back, ATR-scaled
compression_strength    inside-bar / NR-n depth
breakout_strength       close beyond n-bar range, distance/atr, × volume_ratio
momentum_strength       consecutive same-direction bodies, ATR-weighted
double_retest_strength  two touches of a level within k bars with declining penetration
```
All squashed with `tanh(x/s)` or a min-max over a trailing window (trailing, never global — global min-max is leakage).

### 4.10 Time & session
```
hour, minute, day_of_week, day_of_month, week_of_year
hour_sin, hour_cos, dow_sin, dow_cos, minute_of_day_sin/cos, month_sin/cos
session_onehot        ASIA | LONDON | NY | LONDON_NY_OVERLAP | OFF
session_progress      ∈[0,1]
bars_since_session_open
is_session_first_hour, is_session_last_hour
is_month_end, is_quarter_end, is_dst_shift_week
minutes_to_ny_open, minutes_to_london_open
session_atr_ratio     current session ATR / that session's trailing-20-day ATR
session_range_atr, dist_session_high_atr, dist_session_low_atr
session_trend         session VWAP slope, normalized
```
Session boundaries (UTC, DST-aware via `zoneinfo`): Asia 00:00–08:00, London 07:00–16:00, NY 12:00–21:00, overlap 12:00–16:00. Configurable.

### 4.11 Cross-timeframe derived
```
htf_trend_agreement       Σ sign(ema_stack_score_tf) / 4        ∈[-1,1]
htf_alignment_score       weighted agreement, weights configurable
tf_vol_ratio_5m_1h        atr_pct_5m / atr_pct_1h
dist_1d_high_atr, dist_1d_low_atr, dist_4h_ob_atr
running_4h_position       where in the *incomplete* 4h bar we are, built from 5m ⚠ explicit
mtf_regime_concordance
```

**Count:** ≈ 190 at 5m + ≈ 40 per HTF (subset) × 4 + ≈ 25 static ≈ **370 raw columns**, reduced to ~180–220 after correlation pruning (|ρ|>0.97 cluster → keep highest MI with target, decided **on training folds only**).

---

## 5. Target / Label Design

### 5.1 The primary label: cost-aware triple barrier

For each decision bar `t`, hypothetical entry at `open_{t+1}` (with spread):

```
entry_buy  = open_{t+1} + spread/2 + slippage
sl_dist    = k_sl * ATR14_t                 (k_sl configurable, default 1.5)
tp_dist    = rr * sl_dist                   (rr = 1.0 for the 1R label)
horizon    = H bars                          (H ∈ {3, 6, 12, 24})

Walk bars t+1 .. t+H using HIGH/LOW:
  if low  <= sl  → label = SL_HIT
  if high >= tp  → label = TP_HIT
  ambiguous bar (both touched) → SL_HIT           ← pessimistic, always
  neither by t+H → TIMEOUT, record signed return
```

Labels produced per horizon:
- `y_dir` ∈ {BUY, SELL, WAIT} — BUY iff the long triple barrier hits TP first **and** the short one does not; symmetric for SELL; else WAIT.
- `y_tp1r`, `y_tp15r`, `y_tp2r`, `y_tp3r` — binary "reached +nR before −1R" (Section 11's TP ladder probabilities).
- `y_ret_h` — net log return over H bars **minus round-trip cost**, in ATR units.
- `y_mfe`, `y_mae` — max favorable/adverse excursion in R units over H.
- `y_vol` — realized volatility over the next H bars (ATR units).
- `y_regime` — see 5.3.

**The pessimistic tie-break matters enormously.** With 5m bars you cannot know intrabar path order. Resolving ties in favor of TP inflates measured win rate by 5–12 pp on gold. If tick data is ever added, the backtester upgrades to true path resolution; until then, pessimism is the honest default and the same rule is used in labels *and* backtest.

### 5.2 Why not next-candle direction

Next-bar direction at 5m on gold is ~50.2/49.8 and is entirely inside the spread. Any model that "predicts" it is fitting microstructure noise or leaking. The triple-barrier formulation asks the only question that matters: *given a real stop and a real target, does this trade pay after costs?*

### 5.3 Regime labels

Two sources, both PIT-safe, used as an auxiliary target and as a conditioning input:

1. **Rule-based (primary, computed causally):**
   `TREND_UP`: `adx_14 > 25 AND ema_stack_score > 0.5 AND efficiency_ratio_20 > 0.35`
   `TREND_DOWN`: mirror
   `RANGE`: `adx_14 < 20 AND range_position ∈ [0.2,0.8] AND efficiency_ratio < 0.25`
   `HIGH_VOL` / `LOW_VOL`: `atr_percentile_252 > 0.8` / `< 0.2` (overlay dimension)
   `BREAKOUT`: `breakout_strength > 0.6 AND range_expansion > 1.5`
   `REVERSAL`: `mss_* within 6 bars AND sweep_strength > 0.5`
   Precedence is explicit and configurable; regimes are emitted as a primary label plus a soft one-hot.

2. **Unsupervised (validation/monitoring only):** 3-state Gaussian HMM on `[ret, |ret|, atr_pct, adx]` fitted **on training data only**, used with `filter` (forward algorithm, no smoothing — Viterbi smoothing over the full sequence is leakage).

### 5.4 Meta-labeling (second stage)

Primary model → direction. **Meta-model** → "should we take this signal?" trained only on bars where the primary fires, target = did the trade actually make money after costs. This is the cleanest known mechanism for raising precision and cutting trade count, and it maps directly onto Section 29's "the model must be allowed to say WAIT."

### 5.5 Sample weighting & overlap

Overlapping labels (bar `t` and `t+1` both look forward H bars) break the i.i.d. assumption and inflate effective sample size ~H×.
- **Uniqueness weight:** `u_t = mean over the label's life of 1/(# concurrent labels)`.
- **Time decay:** `w_t = u_t · d(t)`, `d` linear from `decay_floor` (oldest) to 1.0 (newest), `decay_floor` configurable (default 0.5).
- **Class weight:** inverse-frequency, capped at 5×, applied on top.
- Weights feed the loss **and** the sampler.

### 5.6 Label distribution sanity gate

Before any training: assert `WAIT` fraction ∈ [0.5, 0.95]. If BUY+SELL > 50% of bars at H=12 with k_sl=1.5, either the barriers are too tight or costs are being ignored — training is refused with a diagnostic, not run.

---

## 6. Training Methodology

### 6.1 Determinism
`seed_everything(seed)` sets `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, `DataLoader(worker_init_fn=..., generator=...)`. Every run records the seed; every reported result is the **median of 3 seeds** with the spread shown. A single-seed NN result is not a result.

### 6.2 Per-fold pipeline (order is mandatory)

```
1. slice fold indices  (train | purge | val | purge | calib | purge | test | embargo)
2. drop features with >2% NaN on TRAIN; impute rest with TRAIN median (forward-fill first)
3. correlation prune + variance filter — decided on TRAIN only, applied to all splits
4. fit scaler on TRAIN only          RobustScaler (median/IQR) — gold has fat tails
   clip to ±8 IQR (fit-time clip bounds from TRAIN)
5. build sequences (view, not copy); a sequence is valid only if all L bars are in-split
6. train on TRAIN, early-stop on VAL
7. fit calibrator on CALIB (never train, never val, never test)
8. evaluate on TEST — ONCE per fold, at the end. No test-driven iteration.
```

### 6.3 Optimization
- AdamW, `lr` 1e-4–3e-3 (Optuna), cosine schedule with 5% linear warmup, `weight_decay` 1e-6–1e-2.
- Gradient clipping at 1.0. AMP fp16 on CUDA, fp32 on CPU.
- Batch 128–512. Class-balanced sampling **off by default** (it distorts the base rate that calibration depends on); handled by loss weights instead.
- Early stopping patience 15 epochs on the **validation trading metric**, not val loss:
  `score = expectancy_R × sqrt(n_trades)` computed by running the signal engine over validation predictions with fixed default thresholds. Optimizing val cross-entropy optimizes the wrong thing; a model can improve CE while making worse trades.
- EMA of weights (decay 0.999) evaluated alongside raw weights; the better on val is kept.

### 6.4 Calibration (Section 19)
Raw softmax on financial data is badly overconfident. On the **calibration slice**:
- **Temperature scaling** (single parameter, safest) — default.
- **Vector scaling** and **isotonic regression** (per class, one-vs-rest) as alternatives; the one with the lowest val ECE is selected, ties broken toward the simpler method.
- Report **ECE, MCE, Brier, and a reliability diagram per fold**. Confidence is `f(calibrated_p, agreement_terms)` — never the raw softmax, and never rescaled to look better.

Confidence bands (configurable): `<50 NO_TRADE | 50–65 WEAK | 65–75 MODERATE | 75–85 STRONG | 85–100 VERY_STRONG`.

### 6.5 Hyperparameter search
Optuna TPE, MedianPruner, **objective = mean validation trading score across the last K folds** (never test). Search space: `lr, batch_size, lookback ∈ {48,96,192,288}, d_model, n_layers, dropout, weight_decay, n_heads, k_sl, loss weights`. Budget ~60–120 trials/model on this GPU. Study persisted to SQLite so it is resumable.

### 6.6 Experiment tracking
Every run writes `artifacts/<exp>/<run_id>/`: `config.resolved.yaml`, `hparams.json`, `feature_manifest.json`, `scaler.pkl`, `calibrator.pkl`, `model_best.pt`, `model_ema.pt`, `metrics.json`, `predictions_test.parquet`, `env.json` (versions, GPU, git SHA + dirty flag), `train.log`. Run IDs are content-addressed; **writes to an existing run_id raise**. Local MLflow optional, DB-of-record is `run`/`run_metric`.

---

## 7. Walk-Forward Validation Methodology

### 7.1 Splitting

```
      ├──────── TRAIN ────────┤ P ├─ VAL ─┤ P ├CALIB┤ P ├─ TEST ─┤ EMBARGO ┤
                                  ▲                       ▲
                        purge = max_feature_lookback + max_label_horizon
                        embargo = 1 trading day (288 bars)
```

- **Purge** removes bars whose *label window* overlaps the next split. With `max_label_horizon = 24` and `max_feature_lookback = 1000` (the 1000-bar ATR percentile), purge = 1024 bars ≈ 3.6 days. Computed automatically from the feature manifest — never hard-coded.
- **Embargo** additionally removes bars immediately after the test split before the next train window, to prevent serial-correlation bleed when the window rolls.

### 7.2 Fold schedule (default: rolling, 24-month train, 6-month test)

| Fold | Train | Val | Calib | Test |
|---|---|---|---|---|
| 0 | 2021-01→2022-12 | 2023-01→2023-04 | 2023-05→2023-06 | 2023-07→2023-12 |
| 1 | 2021-07→2023-06 | 2023-07→2023-10 | 2023-11→2023-12 | 2024-01→2024-06 |
| 2 | 2022-01→2023-12 | 2024-01→2024-04 | 2024-05→2024-06 | 2024-07→2024-12 |
| 3 | 2022-07→2024-06 | 2024-07→2024-10 | 2024-11→2024-12 | 2025-01→2025-06 |
| 4 | 2023-01→2024-12 | 2025-01→2025-04 | 2025-05→2025-06 | 2025-07→2025-12 |
| 5 | 2023-07→2025-06 | 2025-07→2025-10 | 2025-11→2025-12 | 2026-01→2026-06 |

Both **rolling** (fixed window, adapts to regime change) and **anchored** (expanding, more data) are implemented; rolling is the default because gold's character changed materially in 2024–2025 and stale 2021 data may hurt more than it helps. We run both and report both.

### 7.3 What is reported
Per fold: full metric set (Section 18). Across folds: mean, median, std, **worst fold**, and fraction of folds profitable. A strategy that is profitable in 3/6 folds with a great average is not a strategy — the report makes that visible rather than averaging it away.

### 7.4 The single-use test rule
Test-split results are computed once per configuration and logged immutably. If a test result motivates a change, the changed configuration is a **new experiment** and the old test result stays in the record. The final honest number is the one from the last configuration that was never tuned on test. This is tracked by a `test_evaluations` counter per experiment lineage, surfaced in every report.

### 7.5 Combinatorial purged CV (later)
For hyperparameter stability estimates, CPCV (López de Prado) over 6 groups / 2 test groups gives 15 paths and a distribution of Sharpe, enabling a **deflated Sharpe ratio** that accounts for the number of trials we ran. Phase 10.

---

## 8. Backtesting Methodology

### 8.1 Event loop (bar-level, pessimistic)

```
for each 5m bar t (chronological, no lookahead):
    ── 1. MARK-TO-MARKET open positions using bar t's OHLC
    ── 2. RESOLVE exits, intrabar order = pessimistic:
           gap check: if open_t already beyond SL → fill at open (gap fill), not at SL
           if bar touched both SL and TP → SL wins
           partial TPs then trailing stop updated on bar close only
    ── 3. UPDATE risk state (daily P/L, consecutive losses, drawdown, halts)
    ── 4. COMPUTE features for bar t  (close_time = t_close)   ← uses bars ≤ t only
    ── 5. PREDICT (model), CALIBRATE
    ── 6. SIGNAL ENGINE → Signal | WAIT
    ── 7. RISK ENGINE → Order | Rejection
    ── 8. QUEUE order for execution at open of bar t+1
    ── 9. record equity point
```

Steps 4–8 cannot see bar `t+1`. Step 8's queued order fills at `open_{t+1} ± spread/2 ± slippage`. This ordering is the structural guarantee against lookahead; `tests/test_backtest_causality.py` runs the backtester on a dataset where all bars after index `i` are replaced with NaN and asserts that decisions up to `i` are byte-identical.

### 8.2 Cost model
```
spread:      fixed | session-dependent | ATR-scaled | historical (if bid/ask available)
             default: session table {ASIA: 0.30, LONDON: 0.18, NY: 0.20, OVERLAP: 0.16,
                                     ROLLOVER: 1.20} price units, configurable
slippage:    base_ticks + impact*(size/typical_size) + vol_mult*(atr_pct/median_atr_pct)
             asymmetric: worse on stops (stop-out slippage multiplier, default 1.8)
commission:  $ per lot round turn (instrument table)
swap:        per-lot overnight, triple on Wednesday
```
Every cost component is logged per trade so the report can show **gross vs net** and answer "does the edge survive costs?" quantitatively rather than rhetorically.

### 8.3 Fidelity ladder
1. **Bar-level pessimistic** (ship this) — conservative, fast, honest.
2. **Sub-bar reconstruction** — use 1m bars, if available, to order intrabar events. Removes most tie-break pessimism.
3. **Tick replay** — true path. Only if tick data is obtained.

The report always states which fidelity level produced it.

### 8.4 Baselines (mandatory in every report)
`BuyAndHold` · `RandomEntry` (matched trade count & holding period, 200 seeds → distribution, model must beat the 95th percentile) · `EMA(20/50) cross` · `ATR breakout trend-follow` · `LightGBM-on-same-features`. All run through the **identical** backtester and cost model.

### 8.5 Robustness suite
- Cost sensitivity: 0.5×, 1×, 1.5×, 2× spread — an edge that dies at 1.5× spread is not an edge.
- Threshold sensitivity: heat map of expectancy over `(p_threshold, min_rr)`. A sharp spike = overfit; a broad plateau = real.
- Randomized entry timing ±1 bar; feature-noise injection (σ=0.05); Monte Carlo trade-order shuffle → drawdown distribution.
- Regime slicing: does the edge exist only in one regime or one year?

---

## 9. Risk-Management Architecture

### 9.1 Structure

```
RiskEngine
├── RiskState                (equity, balance, peak, daily P/L, streak, halts, open positions)
├── PreTradeGate chain       (each returns Pass | Reject(code, detail); ALL must pass)
│    1  TradingHaltedGate         daily-loss / DD / consec-loss / kill-switch
│    2  SessionGate               session enabled? news blackout? rollover window?
│    3  SpreadGate                spread <= max_spread_atr * ATR
│    4  VolatilityGate            atr_percentile ∈ [min, max]  (no dead / no chaos)
│    5  MaxOpenTradesGate
│    6  MaxTradesPerSessionGate / MaxTradesPerDayGate
│    7  CorrelationGate           no new same-direction position within N bars
│    8  StopDistanceGate          sl_dist <= max_sl_atr (3.0) * ATR AND >= min_sl_atr (0.8)
│    9  RiskBudgetGate            open_risk + new_risk <= max_total_risk_pct
│   10  CooldownGate              n bars after a loss / after a halt release
├── PositionSizer               risk-based, SL-distance-driven
└── PostTradeUpdater            streaks, daily P/L, halts, profit lock
```

### 9.2 Position sizing (SL-distance driven, never confidence-driven)

```python
risk_amount = equity * risk_pct                      # e.g. 10_000 * 0.01 = 100 USD
sl_ticks    = abs(entry - sl) / tick_size            # e.g. 7.40 / 0.01 = 740
value_per_lot_per_tick = tick_value                  # 1.00 USD
lots_raw    = risk_amount / (sl_ticks * tick_value)  # 100 / 740 = 0.135
lots        = floor_to_step(clamp(lots_raw, min_lot, max_lot), lot_step)   # 0.13
actual_risk = lots * sl_ticks * tick_value           # 96.20 — recomputed, logged
if actual_risk > risk_amount * 1.05: reject
```
Confidence may **scale risk_pct within a narrow configured band** (e.g. `risk_pct ∈ [0.5%, 1.0%]` mapped from confidence 65→90) if `confidence_scaling.enabled` is true, but it is **off by default**, and it can never exceed `max_risk_per_trade`. Confidence's real job is the binary trade/no-trade decision.

### 9.3 Portfolio-level controls
| Control | Default | Action on breach |
|---|---|---|
| `max_risk_per_trade` | 1.0% | reject trade |
| `max_total_open_risk` | 2.0% | reject trade |
| `max_open_trades` | 2 | reject trade |
| `max_daily_loss` | 3.0% | **halt until next trading day** |
| `max_weekly_loss` | 6.0% | halt until next week |
| `max_consecutive_losses` | 4 | halt for `cooldown_bars` (default 288) |
| `max_drawdown` | 10% | **hard stop, requires manual reset** |
| `daily_profit_lock` | +4% | stop trading for the day (protect the win) |
| `dd_risk_scaling` | DD>5% → risk × 0.5 | reduce size |
| `max_trades_per_session` | 3 | reject |
| `max_spread` | 2.5 × median session spread | reject |
| `news_blackout` | ±15 min around high-impact | reject (needs a calendar source) |

Every rejection is logged with its code — the report shows *why* trades did not happen, which is as diagnostic as the trades that did.

### 9.4 SL / TP construction
```
sl_candidates = [
  atr_sl        = k_sl * ATR14                       (k_sl configurable, default 1.5)
  structure_sl  = last confirmed swing ± buffer*ATR
  liquidity_sl  = beyond nearest liquidity level ± buffer
  model_sl      = entry ± E[MAE] * mae_multiplier    (from the MFE/MAE head)
]
sl = select(mode)     # mode: widest | structure_first | atr_only | model_blend
sl = clamp(sl, min_sl_atr*ATR, max_sl_atr*ATR)
if required_sl > max_sl_atr * ATR: REJECT (do not shrink the stop to fit the rule)
```
TP ladder: `TP1=1R (close 50%)`, `TP2=2R (close 30%)`, `TP3=3R or structure target (runner)`, with break-even move after TP1 and ATR trailing (`trail_atr_mult`, default 2.0) after TP2. Structure-based TP overrides an R-target when the nearest opposing liquidity level sits between entry and the R-target — taking 1.7R at real resistance beats missing 2R.

---

## 10. Project Folder Structure

```
D:\AI_BOT\
├── config/
│   ├── base.yaml                 # defaults for everything
│   ├── data.yaml                 # sources, sessions, instrument spec
│   ├── features.yaml             # every feature parameter (drives fs_version hash)
│   ├── labels.yaml               # horizons, barriers, costs used in labeling
│   ├── models/{lgbm,mlp,gru,lstm,tcn,tcn_gru,transformer}.yaml
│   ├── training.yaml             # optimizer, folds, purge/embargo, seeds
│   ├── signals.yaml              # thresholds, confidence bands, gate params
│   ├── risk.yaml                 # all limits from §9.3
│   ├── backtest.yaml             # costs, fidelity, baselines
│   └── live.yaml                 # paper/live, broker adapter, polling
├── src/xaubot/
│   ├── __init__.py
│   ├── config/          schema.py (pydantic), loader.py (yaml+env+CLI override), hashing.py
│   ├── core/            types.py (Bar,Signal,Order,Fill,Trade,ModelOutput), enums.py,
│   │                    time_utils.py, sessions.py, logging.py, seeding.py, errors.py
│   ├── data/            loaders/{csv_loader,base}.py, validators.py, cleaning.py,
│   │                    resample.py, store.py (parquet), calendar.py, quality_report.py
│   ├── features/        base.py (Transform ABC), price.py, volatility.py, volume.py,
│   │                    trend.py, structure.py, liquidity.py, fvg.py, order_blocks.py,
│   │                    patterns.py, time_features.py, mtf.py (as-of merge),
│   │                    engine.py (batch+streaming), manifest.py, pit_audit.py
│   ├── labels/          barriers.py, horizons.py, excursions.py, regime.py,
│   │                    meta_labels.py, weights.py, engine.py
│   ├── datasets/        folds.py (walk-forward+purge+embargo), scaling.py,
│   │                    sequences.py, torch_dataset.py, builder.py
│   ├── models/          base.py, registry.py, heads.py, losses.py,
│   │                    mlp.py, gru.py, lstm.py, tcn.py, tcn_gru.py, transformer.py,
│   │                    lgbm.py, calibration.py
│   ├── training/        trainer.py, loop.py, metrics.py, early_stopping.py,
│   │                    optuna_search.py, experiment.py, artifacts.py
│   ├── inference/       predictor.py, batch_predictor.py, explain.py (SHAP/IG)
│   ├── signals/         engine.py, gates.py, stops.py, targets.py,
│   │                    confidence.py, reasons.py
│   ├── risk/            engine.py, state.py, gates.py, sizing.py, limits.py
│   ├── backtesting/     broker.py, engine.py, costs.py, portfolio.py,
│   │                    walkforward.py, baselines.py, robustness.py
│   ├── evaluation/      metrics.py, slicing.py, reports.py, plots.py, calibration_report.py
│   ├── execution/       base.py (ExecutionAdapter ABC), paper.py, backtest_adapter.py,
│   │                    live_stub.py, order_manager.py
│   ├── live/            feed.py, runner.py (5m scheduler), state_store.py, watchdog.py
│   ├── service/         api.py (FastAPI), schemas.py, deps.py
│   ├── dashboard/       app.py (Dash), components/, callbacks.py
│   ├── db/              models.py (SQLAlchemy), session.py, migrations/ (alembic), repo.py
│   └── cli.py           typer: ingest|features|labels|dataset|train|tune|backtest|
│                               walkforward|report|paper|serve|dashboard
├── scripts/             download_sample.py, run_full_pipeline.ps1, profile_features.py
├── tests/
│   ├── unit/            per module
│   ├── property/        hypothesis: features are shift-invariant, scale-equivariant
│   ├── leakage/         test_no_lookahead.py, test_mtf_alignment.py,
│   │                    test_scaler_fit_scope.py, test_causal_model.py,
│   │                    test_backtest_causality.py, test_purge_embargo.py
│   ├── integration/     test_end_to_end_tiny.py (200 bars, full pipeline, <30s)
│   └── fixtures/        synthetic_bars.py (known-answer generators)
├── notebooks/           01_data_quality, 02_feature_eda, 03_label_analysis,
│                        04_results, 05_shap  (exploration only, never imported)
├── artifacts/           <exp_id>/<run_id>/...        (gitignored)
├── data/                raw/ canonical/ resampled/ features/ labels/ datasets/ (gitignored)
├── docs/                ARCHITECTURE.md (this file), DATA_CONTRACT.md, LEAKAGE_AUDIT.md,
│                        RESULTS.md, RUNBOOK.md
├── .env.example         DB_URL, BROKER_KEY placeholders (never real secrets)
├── pyproject.toml       ruff + mypy(strict) + pytest config
└── README.md
```

**Hard rule:** `src/xaubot/features/` may not import from `models/`, `signals/`, `risk/`, or `backtesting/`. Dependencies flow strictly downward through the layer stack. Enforced by an import-linter rule in CI.

---

## 11. Development Roadmap

Each phase lists deliverable → tests → exit criteria. **We stop at the end of each phase for your review before proceeding.**

| Ph | Component | Deliverables | Exit criteria |
|---|---|---|---|
| **0** | Scaffold | pyproject, config schema (pydantic), logging, seeding, core types, CLI skeleton, CI | `pytest` green, `mypy --strict` clean, `xaubot --help` works |
| **1** | Data ingestion | CSV loader, validators, dedupe, gap report, resampler, Parquet store, quality report | Gap report on your real CSV; resample round-trip test; DST correctness test |
| **2** | Feature engine | All groups §4, manifest, batch+streaming drivers, PIT audit | batch≡streaming parity test; known-answer tests vs hand-computed values; **leakage suite green** |
| **3** | Labels | Triple barrier, horizons, MFE/MAE, regime, weights | Label distribution report; synthetic-path known-answer tests; WAIT fraction gate |
| **4** | Datasets & folds | Walk-forward folds, purge/embargo, per-fold scaling, sequence windowing | `test_purge_embargo.py`; assert no timestamp appears in two splits; scaler-scope test |
| **5** | Baselines | LightGBM + logistic + EMA/random/B&H strategy baselines | Honest baseline numbers on fold 0 — **the bar the NN must clear** |
| **6** | NN baseline | GRU + TCN_GRU, multi-task heads, training loop, calibration | Beats logistic on val AUC/Brier; ECE < 0.05 after calibration; 3-seed stability |
| **7** | Evaluation | Full metric suite, slicing, reports, plots, calibration report | Fold-by-fold report generated end to end |
| **8** | Backtester | Event-driven engine, costs, partials, trailing, portfolio | `test_backtest_causality.py` green; baselines reproduced within tolerance |
| **9** | Signal engine | Gates, SL/TP construction, confidence, reasons | Threshold heat map is a plateau, not a spike |
| **10** | Risk engine | All §9 controls, sizing, halts, state machine | Unit test per control; sizing exactness test; forced-drawdown scenario test |
| **11** | Walk-forward | Full WF harness, robustness suite, deflated Sharpe | **RESULTS.md with the honest verdict**, incl. "NN does not beat LightGBM" if true |
| **12** | Explainability | SHAP (GradientExplainer/DeepLIFT for NN, TreeSHAP for LGBM), attention export, reason strings | Reasons traceable to attributions; no post-hoc storytelling |
| **13** | Inference + API | Predictor, FastAPI `/predict` `/signal` `/health`, §27 response schema | Contract test on the exact JSON shape; latency < 100 ms |
| **14** | Paper trading | Live feed adapter, 5m scheduler, PaperBroker, state persistence, watchdog | 2-week paper run; paper vs backtest divergence < tolerance on identical bars |
| **15** | Dashboard | Dash/Plotly: candles, signal panel, P/L, DD, trades, model metrics | Renders live from paper state |
| **16** | Advanced models | Transformer, TFT-lite, ensembling, meta-labeling stage 2 | Only if Phase 11 showed real edge; same WF protocol |

**Live money is not in this roadmap.** `LiveBroker` remains a stub until Phases 11 and 14 produce evidence, and enabling it will require an explicit config flag plus a documented go/no-go checklist in `RUNBOOK.md`.

---

## 12. Leakage Audit Register

Every item is a test in `tests/leakage/`.

| # | Channel | Concrete failure | Control |
|---|---|---|---|
| 1 | HTF incomplete bars | 5m at 10:15 uses the 10:15–10:30 15m bar | as-of merge on `close_time`, `direction='backward'`; §1.4 test |
| 2 | Centered/future windows | `rolling(center=True)`, `shift(-1)`, `bfill()` | AST scan in CI rejects these in `features/` |
| 3 | Global normalization | scaler fit on full dataset before split | scaler fit only inside `folds.py`; scope test |
| 4 | Swing/fractal confirmation | swing at bar `i` known at bar `i` | confirmation shift by `k`; known-answer test |
| 5 | Label overlap | overlapping H-bar labels leak across the split boundary | purge = lookback + horizon, computed from manifest |
| 6 | Same-bar execution | entry at the close of the decision bar | execution at `open_{t+1}`, enforced in the engine |
| 7 | Optimistic tie-break | both SL and TP touched → count TP | pessimistic in both labels and backtest |
| 8 | Survivorship in levels | prev-day levels computed with today's data | levels built from *completed* sessions only |
| 9 | Feature selection on all data | MI/correlation pruning over the full set | selection inside the train fold only |
| 10 | Test-set tuning | iterate until test looks good | single-use test rule + `test_evaluations` counter in reports |
| 11 | Bidirectional/non-causal nets | `bidirectional=True`, symmetric conv padding | gradient-based causality test |
| 12 | Trailing-window min-max | pattern strength normalized with global min/max | trailing-window normalization only |

---

## 13. Metrics Reported (Section 18 mapping)

**Core:** total trades, win rate, profit factor, net profit, avg R, expectancy (R & $), Sharpe (annualized, per-trade and per-bar), Sortino, Calmar, max drawdown (% & $ & duration), avg win, avg loss, largest win, largest loss, max consecutive wins/losses, recovery factor, exposure %, avg bars held, turnover, cost drag (gross − net).

**Slices:** session (Asia/London/NY/overlap), regime (7), direction (long/short), day of week, hour of day, volatility regime (4), confidence band (5), month, fold.

**ML-side:** accuracy/precision/recall/F1 per class, ROC-AUC, PR-AUC (matters more — classes are imbalanced), Brier, ECE/MCE, reliability diagrams, log loss, and — the one that actually decides things — **expectancy conditional on predicted probability bucket**. If expectancy does not rise monotonically with predicted probability, the model has no usable edge regardless of its AUC.

---

## 14. Configuration & Secrets

Layering: `base.yaml` → environment-specific yaml → `.env` → CLI overrides. Everything is validated by pydantic at startup; unknown keys are errors, not warnings. Secrets (`DB_URL`, broker credentials) come **only** from environment variables; `.env` is gitignored and `.env.example` holds placeholders. The resolved config is hashed and written into every artifact, so any result can be reproduced exactly.

---

## 15. Honest Expectations

What "good" looks like on this problem, so we can recognize success and failure without moving the goalposts afterward:

- Direction accuracy of **52–55%** on the triple-barrier label is *good*. 60%+ on 5m gold means look for the bug first.
- After realistic costs, a per-trade expectancy of **+0.05R to +0.15R** with 1–4 trades/day is a real, tradable result.
- A walk-forward Sharpe of **0.8–1.5** with max DD under 15% would be a genuinely strong outcome.
- The most likely honest outcome is **"edge exists in some regimes/sessions and not others."** The architecture is built to detect and report that, and the signal engine's gates are the mechanism for trading only where the edge lives.
- If Phase 11 shows the NN does not beat LightGBM or the trend baseline net of costs, that will be stated plainly in `RESULTS.md`, and the recommendation will be to ship the simpler model or nothing. No backtest will be adjusted to change that conclusion.
