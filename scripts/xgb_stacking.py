"""
ICDAR 2026 CircleID — Level-2 XGBoost Stacking Meta-Learner
============================================================
Consumes OOF probabilities from Level-1 models (ConvNeXt-Tiny + SwinV2-Tiny)
and trains an XGBClassifier with engineered meta-features.

Inputs (all auto-detected under /kaggle/input/ or local CWD):
  - oof_train_features.csv   (16 prob cols from Level-1 OOF)
  - test_features.csv        (16 prob cols from Level-1 test inference)
  - train.csv                (original labels: image_id, pen_id, writer_id)

Outputs:
  - xgb_submission.csv       (image_id, pen_id)
  - Console: per-fold + overall OOF Macro F1
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report

# ============================================================================
# CONFIG
# ============================================================================
class CFG:
    SEED = 42
    N_FOLDS = 3                  # 3 folds outperformed 5 folds (0.91806 vs 0.91750)
    NUM_CLASSES = 8
    OUTPUT_DIR = "submissions"

    # --- Optuna auto-tuning ---
    USE_OPTUNA = False           # Disabled — Optuna found worse params (0.91693 vs 0.91806)
    OPTUNA_TRIALS = 80

    # XGBoost hyperparameters — PROVEN BEST (scored 0.91806)
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 8,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
        "verbosity": 0,
    }
    EARLY_STOPPING_ROUNDS = 50

    # Level-1 model names (must match column prefixes in OOF/test CSVs)
    L1_MODELS = ["convnext_tiny", "swinv2_tiny_window8_256"]


# ============================================================================
# DATA LOADING
# ============================================================================
def find_file(filename: str) -> Path:
    """Search for a file in local directories, working dir, and CWD."""
    # Fast checks first
    quick = [
        Path(f"icdar-2026-circleid-pen-classification/{filename}"),
        Path(f"submissions/{filename}"),
        Path(f"outputs/{filename}"),
        Path(f"weights/{filename}"),
        Path(filename),
    ]
    for p in quick:
        if p.exists():
            return p

    # Deep recursive search in the current project directory (skip virtual envs & gits)
    for root, dirs, files in os.walk("."):
        if ".git" in dirs: dirs.remove(".git")
        if "venv" in dirs: dirs.remove("venv")
        if "__pycache__" in dirs: dirs.remove("__pycache__")
        
        if filename in files:
            p = Path(root) / filename
            print(f"  Found {filename} at: {p}")
            return p

    raise FileNotFoundError(
        f"Cannot find {filename}. Make sure the Level-1 outputs "
        f"(oof_train_features.csv, test_features.csv) and Kaggle CSVs are available locally."
    )


# ============================================================================
# META-FEATURE ENGINEERING
# ============================================================================
def build_meta_features(df: pd.DataFrame, prob_cols: list, models: list, num_classes: int) -> pd.DataFrame:
    """Engineer meta-features from Level-1 probability outputs.
    
    Creates 3 categories of features:
      1. Row-wise statistics (max, std, entropy, confidence gap)
      2. Per-class model agreement (ConvNeXt vs SwinV2 disagreement)
      3. Per-model confidence indicators
    """
    X = df[prob_cols].copy()
    probs = X.values  # (N, 16)

    # ── 1. Row-wise statistics across ALL 16 probability columns ─────────
    X["meta_max_prob"] = probs.max(axis=1)
    X["meta_min_prob"] = probs.min(axis=1)
    X["meta_std_prob"] = probs.std(axis=1)
    X["meta_mean_prob"] = probs.mean(axis=1)

    # Entropy: -Σ p·log(p) — measures overall prediction uncertainty
    eps = 1e-10
    X["meta_entropy"] = -(probs * np.log(probs + eps)).sum(axis=1)

    # Confidence gap: Top-1 minus Top-2 probability (hesitation measure)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # descending
    X["meta_top1_prob"] = sorted_probs[:, 0]
    X["meta_top2_prob"] = sorted_probs[:, 1]
    X["meta_confidence_gap"] = sorted_probs[:, 0] - sorted_probs[:, 1]
    X["meta_top1_top3_gap"] = sorted_probs[:, 0] - sorted_probs[:, 2]

    # ── 2. Per-class model agreement (ConvNeXt vs SwinV2 disagreement) ───
    m1, m2 = models[0], models[1]
    for c in range(num_classes):
        col1, col2 = f"{m1}_p{c}", f"{m2}_p{c}"
        X[f"meta_diff_c{c}"] = (df[col1] - df[col2]).abs()     # absolute difference
        X[f"meta_ratio_c{c}"] = df[col1] / (df[col2] + eps)    # ratio (with eps safety)
        X[f"meta_mean_c{c}"] = (df[col1] + df[col2]) / 2.0     # agreement average

    # ── 3. Per-model confidence indicators ───────────────────────────────
    for m in models:
        m_cols = [f"{m}_p{c}" for c in range(num_classes)]
        m_probs = df[m_cols].values
        X[f"meta_{m}_argmax"] = m_probs.argmax(axis=1)
        X[f"meta_{m}_max"] = m_probs.max(axis=1)
        X[f"meta_{m}_std"] = m_probs.std(axis=1)
        X[f"meta_{m}_entropy"] = -(m_probs * np.log(m_probs + eps)).sum(axis=1)

        # Per-model confidence gap
        m_sorted = np.sort(m_probs, axis=1)[:, ::-1]
        X[f"meta_{m}_gap"] = m_sorted[:, 0] - m_sorted[:, 1]

    # Model agreement: do both models predict the same class?
    X["meta_models_agree"] = (
        X[f"meta_{models[0]}_argmax"] == X[f"meta_{models[1]}_argmax"]
    ).astype(int)

    return X


# ============================================================================
# OPTUNA HYPERPARAMETER SEARCH
# ============================================================================
def run_optuna_search(X_train, y_train, groups, feature_cols, n_folds, n_trials=80):
    """Search for optimal XGBoost hyperparameters using Optuna + GroupKFold."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": CFG.NUM_CLASSES,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": CFG.SEED,
            "verbosity": 0,
            # --- Tunable params ---
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": 3000,
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 3),
        }

        gkf = GroupKFold(n_splits=n_folds)
        fold_f1s = []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups):
            X_tr = X_train.iloc[train_idx][feature_cols]
            X_va = X_train.iloc[val_idx][feature_cols]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            clf = xgb.XGBClassifier(early_stopping_rounds=100, **params)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            preds = clf.predict_proba(X_va).argmax(axis=1)
            fold_f1s.append(f1_score(y_va, preds, average="macro", zero_division=0))

        return np.mean(fold_f1s)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n{'=' * 60}")
    print(f"OPTUNA SEARCH COMPLETE ({n_trials} trials)")
    print(f"{'=' * 60}")
    print(f"  Best Macro F1: {study.best_value:.5f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Build final params dict
    best = study.best_params.copy()
    best.update({
        "objective": "multi:softprob",
        "num_class": CFG.NUM_CLASSES,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 3000,
        "random_state": CFG.SEED,
        "verbosity": 0,
    })
    return best, study.best_value


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(CFG.SEED)

    # --- Detect output directory ---
    output_dir = Path(CFG.OUTPUT_DIR)
    if not output_dir.exists():
        output_dir = Path(".")
    print(f"Output directory: {output_dir}")

    # --- Load data ---
    oof_path = find_file("oof_train_features.csv")
    test_path = find_file("test_features.csv")

    print(f"Loading OOF features:  {oof_path}")
    print(f"Loading test features: {test_path}")

    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)

    print(f"  OOF shape:   {oof_df.shape}")
    print(f"  Test shape:  {test_df.shape}")

    # --- Load train labels (both train.csv + additional_train.csv) ---
    train_dfs = []
    for csv_name in ["train.csv", "additional_train.csv", "combined_train.csv"]:
        try:
            p = find_file(csv_name)
            train_dfs.append(pd.read_csv(p))
            print(f"  Loaded {csv_name}: {len(train_dfs[-1])} rows from {p}")
        except FileNotFoundError:
            pass

    if not train_dfs:
        raise FileNotFoundError("Cannot find any train CSV (train.csv, additional_train.csv, or combined_train.csv)")

    train_df = pd.concat(train_dfs, ignore_index=True).drop_duplicates(subset=["image_id"], keep="first")
    print(f"  Combined train labels: {len(train_df)} rows")

    # Handle writer_id = -1 (unknown writer)
    train_df["writer_id"] = train_df["writer_id"].astype(str)
    unknown_mask = train_df["writer_id"].isin(["-1", "-1.0"])
    if unknown_mask.sum() > 0:
        train_df.loc[unknown_mask, "writer_id"] = [
            f"pseudo_{i}" for i in range(unknown_mask.sum())
        ]
        print(f"  Assigned {unknown_mask.sum()} pseudo-writer IDs")

    # --- Data Integration: merge OOF with train labels for pen_id + writer_id ---
    oof_df["image_id"] = oof_df["image_id"].astype(str)
    train_df["image_id"] = train_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)

    merged = oof_df.merge(train_df[["image_id", "pen_id", "writer_id"]], on="image_id", how="left")

    if merged["pen_id"].isna().any():
        n_missing = merged["pen_id"].isna().sum()
        print(f"⚠️  {n_missing} OOF samples could not be matched to train.csv!")
        merged = merged.dropna(subset=["pen_id"])

    print(f"  Merged shape: {merged.shape}")
    print(f"  Pen classes:  {sorted(merged['pen_id'].unique())}")
    print(f"  Writers:      {merged['writer_id'].nunique()}")

    # --- Build pen_id remapping (same as Level-1) ---
    unique_pens = sorted(merged["pen_id"].unique())
    pen_to_idx = {int(p): i for i, p in enumerate(unique_pens)}
    idx_to_pen = {i: int(p) for p, i in pen_to_idx.items()}
    merged["label"] = merged["pen_id"].map(pen_to_idx)
    print(f"  Pen ID mapping: {pen_to_idx}")

    # --- Identify probability columns ---
    prob_cols = []
    for m in CFG.L1_MODELS:
        for c in range(CFG.NUM_CLASSES):
            prob_cols.append(f"{m}_p{c}")

    missing_cols = [c for c in prob_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing probability columns in OOF data: {missing_cols}")

    print(f"\n{'=' * 60}")
    print("META-FEATURE ENGINEERING")
    print(f"{'=' * 60}")

    # --- Engineer meta-features ---
    X_train = build_meta_features(merged, prob_cols, CFG.L1_MODELS, CFG.NUM_CLASSES)
    X_test = build_meta_features(test_df, prob_cols, CFG.L1_MODELS, CFG.NUM_CLASSES)

    y_train = merged["label"].values
    groups = merged["writer_id"].values

    feature_cols = X_train.columns.tolist()
    print(f"  Total meta-features: {len(feature_cols)}")
    print(f"  Feature categories:")
    print(f"    - Raw probabilities:    {len(prob_cols)}")
    print(f"    - Row-wise statistics:  {sum(1 for c in feature_cols if c.startswith('meta_') and '_c' not in c and '_convnext' not in c and '_swinv2' not in c)}")
    print(f"    - Per-class agreement:  {sum(1 for c in feature_cols if '_c' in c)}")
    print(f"    - Per-model indicators: {sum(1 for c in feature_cols if '_convnext' in c or '_swinv2' in c)}")

    # ======================================================================
    # MULTI-CONFIG GRID — each config = (n_folds, xgb_params)
    # ======================================================================
    _BASE = {
        "objective": "multi:softprob", "num_class": 8, "eval_metric": "mlogloss",
        "tree_method": "hist", "random_state": 42, "verbosity": 0,
    }

    PARAM_GRID = {
        # Config A: REIGNING CHAMPION — 4f baseline (scored 0.91861)
        "A_4f_champion": (4, {**_BASE,
            "max_depth": 5, "learning_rate": 0.05, "n_estimators": 2000,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 3, "gamma": 0.1,
        }),
        # Config B: Slightly more subsample + colsample (less dropout)
        "B_4f_less_drop": (4, {**_BASE,
            "max_depth": 5, "learning_rate": 0.05, "n_estimators": 2000,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 3, "gamma": 0.1,
        }),
        # Config C: Slower LR + more trees (finer convergence at 4 folds)
        "C_4f_slow_lr": (4, {**_BASE,
            "max_depth": 5, "learning_rate": 0.04, "n_estimators": 2500,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 3, "gamma": 0.1,
        }),
        # Config D: Mild regularization bump (slightly tighter)
        "D_4f_mild_reg": (4, {**_BASE,
            "max_depth": 5, "learning_rate": 0.05, "n_estimators": 2000,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.2, "reg_lambda": 1.5,
            "min_child_weight": 4, "gamma": 0.15,
        }),
        # Config E: Depth 4 + more colsample (simpler trees, more features)
        "E_4f_depth4": (4, {**_BASE,
            "max_depth": 4, "learning_rate": 0.05, "n_estimators": 2000,
            "subsample": 0.8, "colsample_bytree": 0.85,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 3, "gamma": 0.1,
        }),
    }

    # ======================================================================
    # TRAIN & PREDICT FOR EACH CONFIG
    # ======================================================================
    results = []  # (config_name, oof_f1, test_preds, fold_scores, n_folds)

    for cfg_name, (n_folds, xgb_params) in PARAM_GRID.items():
        print(f"\n{'#' * 70}")
        print(f"# CONFIG: {cfg_name}  ({n_folds}-fold CV)")
        print(f"# max_depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}, "
              f"subsample={xgb_params['subsample']}, colsample={xgb_params['colsample_bytree']}")
        print(f"{'#' * 70}")

        gkf = GroupKFold(n_splits=n_folds)
        oof_preds = np.zeros((len(X_train), CFG.NUM_CLASSES))
        test_preds = np.zeros((len(X_test), CFG.NUM_CLASSES))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr = X_train.iloc[train_idx][feature_cols]
            X_va = X_train.iloc[val_idx][feature_cols]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            clf = xgb.XGBClassifier(
                early_stopping_rounds=CFG.EARLY_STOPPING_ROUNDS,
                **xgb_params,
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            fold_probs = clf.predict_proba(X_va)
            oof_preds[val_idx] = fold_probs
            fold_f1 = f1_score(y_va, fold_probs.argmax(axis=1), average="macro", zero_division=0)
            fold_scores.append(fold_f1)

            test_preds += clf.predict_proba(X_test[feature_cols]) / n_folds

            best_iter = getattr(clf, 'best_iteration', clf.n_estimators)
            print(f"  Fold {fold}: F1={fold_f1:.5f} (iter={best_iter})")

        oof_f1 = f1_score(y_train, oof_preds.argmax(axis=1), average="macro", zero_division=0)
        print(f"  → OOF Macro F1: {oof_f1:.5f}  (per-fold: {[f'{s:.5f}' for s in fold_scores]})")

        results.append((cfg_name, oof_f1, test_preds.copy(), fold_scores, n_folds))

        # --- Save submission for this config ---
        test_labels = test_preds.argmax(axis=1)
        test_pen_ids = [idx_to_pen[l] for l in test_labels]
        sub = pd.DataFrame({"image_id": test_df["image_id"].values, "pen_id": test_pen_ids})
        sub_path = str(output_dir / f"xgb_submission_{cfg_name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f"  💾 Saved: {sub_path}")

    # ======================================================================
    # LEADERBOARD — compare all configs
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("LEADERBOARD — All Configs Ranked by OOF Macro F1")
    print(f"{'=' * 70}")

    results.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, f1, _, folds, nf) in enumerate(results, 1):
        marker = " ⭐ BEST" if rank == 1 else ""
        print(f"  #{rank}  {name:22s}  folds={nf}  OOF F1={f1:.5f}  "
              f"per-fold=[{', '.join(f'{s:.4f}' for s in folds)}]{marker}")

    # --- Copy the best config's submission as the default submission.csv ---
    best_name, best_f1, best_test_preds, _, _ = results[0]
    best_labels = best_test_preds.argmax(axis=1)
    best_pen_ids = [idx_to_pen[l] for l in best_labels]
    best_sub = pd.DataFrame({"image_id": test_df["image_id"].values, "pen_id": best_pen_ids})
    default_sub_path = str(output_dir / "submission.csv")
    best_sub.to_csv(default_sub_path, index=False)
    print(f"\n✅ Best config '{best_name}' (F1={best_f1:.5f}) → {default_sub_path}")

    # --- Save best test probabilities ---
    prob_df = pd.DataFrame(best_test_preds, columns=[f"xgb_p{c}" for c in range(CFG.NUM_CLASSES)])
    prob_df.insert(0, "image_id", test_df["image_id"].values)
    prob_df.to_csv(str(output_dir / "xgb_test_probs.csv"), index=False)

    print(f"\n   Class distribution (best):")
    dist = best_sub["pen_id"].value_counts().sort_index()
    for cls, cnt in dist.items():
        print(f"     pen_{cls}: {cnt} ({cnt/len(best_sub)*100:.1f}%)")


if __name__ == "__main__":
    main()
