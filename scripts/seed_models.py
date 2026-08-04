#!/usr/bin/env python3
"""Seed VIT AI model artifacts — runs at Docker build time.

Trains all 16 VIT ensemble models using real historical match data when
available (CSV files in DATA_DIR), falling back to synthetic data otherwise.

Real data schema expected in CSV files:
  home_team, away_team, home_goals, away_goals, actual_outcome (H/D/A),
  league, date, season, B365H, B365D, B365A

Features engineered (10-dim):
  0  home_elo        — ELO rating normalised to [0, 1]
  1  away_elo
  2  home_form       — rolling win-rate over last 5 matches
  3  away_form
  4  prob_home       — vig-free implied probability from Betfair odds
  5  prob_draw
  6  prob_away
  7  home_goals_avg  — rolling avg goals scored, last 5 matches
  8  away_goals_avg
  9  h2h_home_rate   — historical H2H win rate for home team
"""
import os
import sys
import logging
import glob
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from app.services.rating_shim import RatingShim

try:
    from xgboost import XGBClassifier
except ImportError:
    print("xgboost not installed — using GradientBoostingClassifier as fallback.")
    XGBClassifier = GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("seed_models")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
DATA_DIR  = Path(os.getenv("DATA_DIR", "/app/data"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Real data loading ─────────────────────────────────────────────────────────

def _load_csvs() -> list[dict]:
    """Load all CSVs from DATA_DIR and return a list of row dicts."""
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available — falling back to synthetic data")
        return []

    patterns = [
        str(DATA_DIR / "*.csv"),
        str(DATA_DIR / "uploads" / "*.csv"),
        # Vitnetwork uploads path (used during local development)
        "/tmp/vit-core/vit/data/uploads/*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        log.warning("No CSV files found in %s — using synthetic data", DATA_DIR)
        return []

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            required = {"home_team", "away_team", "home_goals", "away_goals",
                        "actual_outcome", "B365H", "B365D", "B365A"}
            if required.issubset(df.columns):
                frames.append(df)
                log.info("Loaded %d rows from %s", len(df), Path(f).name)
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)

    if not frames:
        return []

    import pandas as pd
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    combined = combined.sort_values("date", na_position="last").reset_index(drop=True)
    return combined.to_dict("records")


def _engineer_features(rows: list[dict]):
    """
    Compute 10-dim feature vectors + three target arrays from real match rows.
    Returns (X: float32 ndarray, y3: int ndarray, y_binary: int ndarray)
    """
    elo      = defaultdict(lambda: 1500.0)
    form     = defaultdict(list)   # team -> list of results (1=win,0.5=draw,0=loss)
    goals_sc = defaultdict(list)   # team -> goals scored per match
    h2h: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [wins, total]
    K = 32

    X, y3, y_btts, y_ou = [], [], [], []

    for row in rows:
        h = str(row.get("home_team", ""))
        a = str(row.get("away_team", ""))
        try:
            hg = int(float(row["home_goals"]))
            ag = int(float(row["away_goals"]))
        except (ValueError, KeyError):
            continue
        outcome = str(row.get("actual_outcome", "")).strip().upper()
        if outcome not in ("H", "D", "A"):
            continue

        # Vig-free probabilities from Betfair odds
        try:
            bh = float(row["B365H"]); bd = float(row["B365D"]); ba = float(row["B365A"])
            if bh <= 1 or bd <= 1 or ba <= 1:
                raise ValueError
            s = 1/bh + 1/bd + 1/ba
            ph, pd_, pa = (1/bh)/s, (1/bd)/s, (1/ba)/s
        except (ValueError, ZeroDivisionError, KeyError):
            ph, pd_, pa = 0.45, 0.25, 0.30

        # Rolling features
        hf  = float(np.mean(form[h][-5:]))     if form[h]     else 0.45
        af  = float(np.mean(form[a][-5:]))     if form[a]     else 0.45
        hga = float(np.mean(goals_sc[h][-5:])) if goals_sc[h] else 1.30
        aga = float(np.mean(goals_sc[a][-5:])) if goals_sc[a] else 1.10
        h2h_d = h2h[h][a]
        h2h_rate = h2h_d[0] / h2h_d[1] if h2h_d[1] > 0 else 0.45

        X.append([
            min(elo[h] / 2000.0, 1.5),
            min(elo[a] / 2000.0, 1.5),
            hf, af,
            ph, pd_, pa,
            min(hga / 4.0, 1.0),
            min(aga / 4.0, 1.0),
            h2h_rate,
        ])

        t3 = 0 if outcome == "H" else (1 if outcome == "D" else 2)
        y3.append(t3)
        y_btts.append(1 if hg > 0 and ag > 0 else 0)
        y_ou.append(1 if hg + ag > 2.5 else 0)

        # Update ELO
        exp_h    = 1.0 / (1.0 + 10 ** ((elo[a] - elo[h]) / 400.0))
        actual_h = 1.0 if outcome == "H" else (0.5 if outcome == "D" else 0.0)
        elo[h]  += K * (actual_h - exp_h)
        elo[a]  += K * ((1.0 - actual_h) - (1.0 - exp_h))

        # Update form + goals
        form[h].append(actual_h);      form[a].append(1.0 - actual_h)
        goals_sc[h].append(hg);        goals_sc[a].append(ag)

        # Update H2H
        h2h[h][a][1] += 1
        if outcome == "H":
            h2h[h][a][0] += 1

    if not X:
        return None, None, None

    return (
        np.array(X, dtype=np.float32),
        np.array(y3, dtype=int),
        np.array(y_btts, dtype=int),
    )


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic_data():
    rng = np.random.RandomState(42)
    N   = 2000
    X   = rng.rand(N, 10).astype(np.float32)
    y3  = rng.choice([0, 1, 2], size=N, p=[0.45, 0.25, 0.30])
    y2  = rng.choice([0, 1],    size=N, p=[0.55, 0.45])
    log.warning("Using SYNTHETIC training data — predictions will not reflect real football patterns")
    return X, y3, y2


# ── Model definitions ─────────────────────────────────────────────────────────

def _build_models(xgb_cls):
    _xgb3  = dict(n_estimators=100, max_depth=5, random_state=42, eval_metric="mlogloss")
    _xgb2  = dict(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
    if xgb_cls is not GradientBoostingClassifier:
        if hasattr(xgb_cls, "use_label_encoder"):
            _xgb3["use_label_encoder"] = False
            _xgb2["use_label_encoder"] = False

    models_3class = {
        "xgb_v1":         xgb_cls(**_xgb3) if xgb_cls is not GradientBoostingClassifier
                          else GradientBoostingClassifier(n_estimators=100, random_state=42),
        "rf_v1":          RandomForestClassifier(n_estimators=100, random_state=42),
        "gbm_v1":         GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_v1":    LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "transformer_v1": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42),
        "lstm_v1":        MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=42),
        "bayes_v1":       GaussianNB(),
        "dixon_coles_v1": GaussianNB(var_smoothing=1e-8),
        "market_v1":      GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42),
        "hybrid_v1":      RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "ensemble_v1":    RandomForestClassifier(n_estimators=200, random_state=42),
    }
    models_2class = {
        "btts_v2":          xgb_cls(**_xgb2) if xgb_cls is not GradientBoostingClassifier
                            else GradientBoostingClassifier(n_estimators=100, random_state=42),
        "over_under_v2":    xgb_cls(**_xgb2) if xgb_cls is not GradientBoostingClassifier
                            else GradientBoostingClassifier(n_estimators=100, random_state=42),
        "correct_score_v2": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    models_rating = {
        "elo_v1":     RatingShim(),
        "poisson_v1": RatingShim(),
    }
    return models_3class, models_2class, models_rating


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Try real data first
    rows = _load_csvs()
    X, y3, y2 = _engineer_features(rows) if rows else (None, None, None)

    using_real = X is not None and len(X) >= 100
    if using_real:
        log.info("Training on %d REAL match records across %d features", len(X), X.shape[1])
    else:
        X, y3, y2 = _synthetic_data()

    models_3class, models_2class, models_rating = _build_models(XGBClassifier)

    trained = errors = 0

    for name, clf in models_3class.items():
        try:
            clf.fit(X, y3)
            path = MODEL_DIR / f"{name}.pkl"
            joblib.dump(clf, path)
            log.info("  ✓ %s → %s", name, path)
            trained += 1
        except Exception as e:
            log.error("  ✗ %s: %s", name, e)
            errors += 1

    for name, clf in models_2class.items():
        try:
            clf.fit(X, y2)
            path = MODEL_DIR / f"{name}.pkl"
            joblib.dump(clf, path)
            log.info("  ✓ %s → %s", name, path)
            trained += 1
        except Exception as e:
            log.error("  ✗ %s: %s", name, e)
            errors += 1

    for name, clf in models_rating.items():
        try:
            clf.fit(X, y2.astype(float))
            path = MODEL_DIR / f"{name}.pkl"
            joblib.dump(clf, path)
            log.info("  ✓ %s → %s", name, path)
            trained += 1
        except Exception as e:
            log.error("  ✗ %s: %s", name, e)
            errors += 1

    data_source = "REAL historical data" if using_real else "SYNTHETIC data"
    log.info("\nSeeded %d/16 models using %s to %s", trained, data_source, MODEL_DIR)
    if errors:
        log.error("%d model(s) failed to seed.", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
