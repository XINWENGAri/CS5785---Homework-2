"""CS 5785 Homework 2 programming exercises.

Part 1 fits OLS on a known linear data-generating process and checks
coefficient and R^2 convergence.

Part 2 classifies TOP clinical-trial outcomes from eligibility text
(Fu et al., HINT / Patterns 2022). The public splits are the course
files: phase_{I,II,III}_{train,valid,test}.csv. Test labels are held
out of fitting and model selection; they are used only to compute the
F1 the Kaggle leaderboard would report on these same 3,427 rows.
"""

from __future__ import annotations

import ast
import json
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # notebook cell
    HERE = Path.cwd()
# Data may sit next to this file (submission zip) or one directory up.
if (HERE / "data").exists():
    ROOT = HERE
elif (HERE.parent / "data").exists():
    ROOT = HERE.parent
else:
    ROOT = HERE
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 5785
ALPHA, BETA = 20.0, 0.5
MU_X, SD_X, SD_EPS = 168.0, 30.0, 20.0
# Population variances under the assignment's 1/n definition.
VAR_X = SD_X ** 2
VAR_EPS = SD_EPS ** 2
VAR_Y = (BETA ** 2) * VAR_X + VAR_EPS
R2_LIMIT = 1.0 - VAR_EPS / VAR_Y

# A word/bigram must show up in at least this many distinct training trials.
# M=10 drops the hapax tail (see the vocabulary table in the report) while
# leaving a few thousand features that L2 logistic regression can estimate.
M_UNI = 10
M_BI = 10

# Keep negations and comparison words. Eligibility templates are not signal.
_KEEP = {"no", "not", "nor", "none", "never", "without", "against", "above", "below"}
_TEMPLATE = {
    "inclusion",
    "exclusion",
    "criteria",
    "criterion",
    "eligibility",
    "eligible",
}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - _KEEP) | _TEMPLATE)
TOKEN_PATTERN = r"(?u)\b[a-z][a-z0-9]{1,}\b"


def preprocess(text) -> str:
    """Lowercase, turn punctuation into spaces, keep letter-led tokens."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = text.replace("\u2013", " ").replace("\u2014", " ").replace("\u2212", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def field_to_text(val) -> str:
    """Join list-valued disease/drug cells into a single string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join(str(x) for x in val)
    s = str(val).strip()
    if s.startswith("["):
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            obj = None
        if isinstance(obj, (list, tuple)):
            return " ".join(str(x) for x in obj)
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    if "nctid" in df.columns and "NCTID" not in df.columns:
        rename["nctid"] = "NCTID"
    if "diseases" in df.columns and "diseases_cleaned" not in df.columns:
        rename["diseases"] = "diseases_cleaned"
    if rename:
        df = df.rename(columns=rename)
    return df


def load_split(phase: str, split: str) -> pd.DataFrame:
    path = DATA / f"phase_{phase}_{split}.csv"
    if not path.exists():
        alt = DATA / f"phase{phase[-1].lower()}_{split}.csv"
        path = alt if alt.exists() else path
    df = normalize_columns(pd.read_csv(path))
    df["criteria_raw"] = df["criteria"]
    df["criteria_pp"] = df["criteria"].map(preprocess)
    disease = df["diseases_cleaned"].map(field_to_text).map(preprocess)
    drugs = df["drugs"].map(field_to_text).map(preprocess)
    df["combined_pp"] = (
        df["criteria_pp"].fillna("") + " " + disease + " " + drugs
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def binary_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def make_logit(kind: str) -> LogisticRegression:
    common = dict(C=1.0, max_iter=2000, random_state=SEED, tol=1e-4)
    if kind == "none":
        return LogisticRegression(penalty=None, solver="lbfgs", **common)
    if kind == "l1":
        return LogisticRegression(
            penalty="l1", l1_ratio=1.0, solver="liblinear", **common
        )
    if kind == "l2":
        return LogisticRegression(
            penalty="l2", l1_ratio=0.0, solver="lbfgs", **common
        )
    raise ValueError(kind)


def fit_eval(kind: str, X_train, y_train, X_valid, y_valid) -> dict:
    model = make_logit(kind)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    out = {
        "train": binary_metrics(model, X_train, y_train),
        "valid": binary_metrics(model, X_valid, y_valid),
        "seconds": round(elapsed, 2),
        "n_iter": int(np.max(model.n_iter_)),
        "nonzero": int(np.count_nonzero(model.coef_)),
        "n_coef": int(model.coef_.size),
    }
    return out, model


def vectorize(train_text, valid_text, ngram_range, min_df):
    vec = CountVectorizer(
        binary=True,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=STOP_WORDS,
        token_pattern=TOKEN_PATTERN,
    )
    X_train = vec.fit_transform(train_text)
    X_valid = vec.transform(valid_text)
    return vec, X_train, X_valid


def top_coefficients(vec, model, k=10):
    names = vec.get_feature_names_out()
    coef = model.coef_.ravel()
    order = np.argsort(coef)
    neg_idx = order[:k]
    pos_idx = order[-k:][::-1]
    def pack(idxs):
        return [
            {"token": str(names[i]), "coef": float(coef[i])}
            for i in idxs
        ]
    return pack(pos_idx), pack(neg_idx)


def sample_bigrams(vec, X_train, k=10):
    names = vec.get_feature_names_out()
    dfreq = np.asarray(X_train.sum(axis=0)).ravel()
    idxs = np.argsort(dfreq)[-k:][::-1]
    return [
        {"token": str(names[i]), "n_docs": int(dfreq[i])}
        for i in idxs
    ]


def explore(df_train, df_valid, df_test) -> dict:
    def block(df, has_label=True):
        crit = df["criteria"]
        info = {
            "n": int(len(df)),
            "criteria_missing": int(crit.isna().sum() + (crit.fillna("").astype(str).str.strip() == "").sum()),
        }
        # empty string after fill is the operational missing count; raw NA is separate
        info["criteria_na"] = int(crit.isna().sum())
        info["criteria_blank"] = int((crit.fillna("").astype(str).str.strip() == "").sum())
        if has_label and "label" in df.columns and df["label"].notna().all():
            y = df["label"].astype(int)
            info["n_label_1"] = int((y == 1).sum())
            info["n_label_0"] = int((y == 0).sum())
            info["pct_label_1"] = float((y == 1).mean() * 100)
            info["pct_label_0"] = float((y == 0).mean() * 100)
            info["f1_all_positive"] = float(f1_score(y, np.ones(len(y), dtype=int)))
        return info

    test_has_label = "label" in df_test.columns and df_test["label"].notna().any()
    return {
        "train": block(df_train),
        "valid": block(df_valid),
        "test": block(df_test, has_label=test_has_label),
    }


def run_representation(train_text, y_train, valid_text, y_valid, ngram_range, min_df, kinds):
    vec, X_train, X_valid = vectorize(train_text, valid_text, ngram_range, min_df)
    models = {}
    fitted = {}
    for kind in kinds:
        stats, model = fit_eval(kind, X_train, y_train, X_valid, y_valid)
        models[kind] = stats
        fitted[kind] = model
        print(
            f"    {kind:4s}  train F1 {stats['train']['f1']:.4f}  "
            f"valid F1 {stats['valid']['f1']:.4f}  "
            f"ROC {stats['valid']['roc_auc']:.4f}  "
            f"PR {stats['valid']['pr_auc']:.4f}  "
            f"nonzero {stats['nonzero']}/{stats['n_coef']}  "
            f"({stats['seconds']}s)"
        )
    best = max(kinds, key=lambda k: (models[k]["valid"]["f1"], models[k]["valid"]["pr_auc"]))
    return {
        "vec": vec,
        "X_train": X_train,
        "X_valid": X_valid,
        "models": models,
        "fitted": fitted,
        "best": best,
        "n_features": int(X_train.shape[1]),
    }


def run_phase(phase: str) -> dict:
    print(f"\n======== Phase {phase} ========")
    train = load_split(phase, "train")
    valid = load_split(phase, "valid")
    test = load_split(phase, "test")
    info = explore(train, valid, test)
    print("explore", json.dumps(info))

    y_train = train["label"].astype(int).to_numpy()
    y_valid = valid["label"].astype(int).to_numpy()

    print(f"  unigram criteria M={M_UNI}")
    uni = run_representation(
        train["criteria_pp"], y_train, valid["criteria_pp"], y_valid,
        (1, 1), M_UNI, ["none", "l1", "l2"],
    )
    pos, neg = top_coefficients(uni["vec"], uni["fitted"]["l1"])
    best_penalty = uni["best"]
    print("  best unigram penalty by valid F1:", best_penalty)

    print(f"  bigram criteria M={M_BI} penalty={best_penalty}")
    bi = run_representation(
        train["criteria_pp"], y_train, valid["criteria_pp"], y_valid,
        (2, 2), M_BI, [best_penalty],
    )
    ten = sample_bigrams(bi["vec"], bi["X_train"], 10)

    print(f"  combined unigram M={M_UNI} penalty={best_penalty}")
    c_uni = run_representation(
        train["combined_pp"], y_train, valid["combined_pp"], y_valid,
        (1, 1), M_UNI, [best_penalty],
    )
    print(f"  combined bigram M={M_BI} penalty={best_penalty}")
    c_bi = run_representation(
        train["combined_pp"], y_train, valid["combined_pp"], y_valid,
        (2, 2), M_BI, [best_penalty],
    )
    combined_choice = "unigram" if c_uni["models"][best_penalty]["valid"]["f1"] >= c_bi["models"][best_penalty]["valid"]["f1"] else "bigram"
    print("  selected combined representation:", combined_choice)

    # Refit the selected combined model on train+valid, then score the test set.
    selected = c_uni if combined_choice == "unigram" else c_bi
    ngram = (1, 1) if combined_choice == "unigram" else (2, 2)
    min_df = M_UNI if combined_choice == "unigram" else M_BI
    pooled = pd.concat([train, valid], ignore_index=True)
    y_pool = pooled["label"].astype(int).to_numpy()
    vec = CountVectorizer(
        binary=True,
        min_df=min_df,
        ngram_range=ngram,
        stop_words=STOP_WORDS,
        token_pattern=TOKEN_PATTERN,
    )
    X_pool = vec.fit_transform(pooled["combined_pp"])
    X_test = vec.transform(test["combined_pp"])
    final = make_logit(best_penalty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X_pool, y_pool)
    pred = final.predict(X_test).astype(int)
    prob = final.predict_proba(X_test)[:, 1]
    test_metrics = None
    if "label" in test.columns and test["label"].notna().all():
        y_test = test["label"].astype(int).to_numpy()
        test_metrics = {
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
        }
        print("  held-out test", test_metrics)

    ids = test["NCTID"].astype(str)
    prefix = {"I": "phase1_", "II": "phase2_", "III": "phase3_"}[phase]
    submission = pd.DataFrame({
        "row_id": prefix + ids,
        "label": pred,
    })

    def slim(rep, kinds):
        return {
            "n_features": rep["n_features"],
            "best": rep["best"] if set(kinds) == set(rep["models"]) else kinds[0],
            "models": rep["models"],
        }

    return {
        "explore": info,
        "M_uni": M_UNI,
        "unigram": slim(uni, ["none", "l1", "l2"]),
        "best_penalty": best_penalty,
        "l1_top_positive": pos,
        "l1_top_negative": neg,
        "l1_sparsity": {
            "nonzero": uni["models"]["l1"]["nonzero"],
            "n_coef": uni["models"]["l1"]["n_coef"],
        },
        "M_bi": M_BI,
        "bigram": slim(bi, [best_penalty]),
        "ten_bigrams": ten,
        "combined_unigram": slim(c_uni, [best_penalty]),
        "combined_bigram": slim(c_bi, [best_penalty]),
        "selected_combined": combined_choice,
        "final_n_features": int(X_pool.shape[1]),
        "final_n_train": int(X_pool.shape[0]),
        "test_metrics_not_used_for_selection": test_metrics,
        "submission": submission,
        "y_test": test["label"].astype(int).to_numpy() if "label" in test.columns else None,
    }


def part1() -> dict:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [10 ** k for k in range(2, 7)]:
        x = rng.normal(MU_X, SD_X, size=(n, 1))
        eps = rng.normal(0.0, SD_EPS, size=(n, 1))
        y = ALPHA + BETA * x + eps
        reg = LinearRegression().fit(x, y.ravel())
        rows.append({
            "n": int(n),
            "alpha_hat": float(reg.intercept_),
            "beta_hat": float(reg.coef_[0]),
            "r2": float(reg.score(x, y.ravel())),
        })
        print(f"n={n:<8d} alpha={reg.intercept_:.6f} beta={reg.coef_[0]:.6f} R2={reg.score(x, y.ravel()):.6f}")

    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
    axes[0].plot(ns, [r["alpha_hat"] for r in rows], "o-", color="C0")
    axes[0].axhline(ALPHA, color="C0", ls="--", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("sample size n")
    axes[0].set_ylabel(r"$\hat\alpha$")
    axes[0].set_title(r"Intercept (truth $\alpha=20$)")
    axes[1].plot(ns, [r["beta_hat"] for r in rows], "s-", color="C1")
    axes[1].axhline(BETA, color="C1", ls="--", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("sample size n")
    axes[1].set_ylabel(r"$\hat\beta$")
    axes[1].set_title(r"Slope (truth $\beta=0.5$)")
    axes[2].plot(ns, [r["r2"] for r in rows], "o-", color="C2")
    axes[2].axhline(R2_LIMIT, color="C2", ls="--", lw=0.8, label=rf"limit {R2_LIMIT:.2f}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("sample size n")
    axes[2].set_ylabel(r"$R^2$")
    axes[2].set_ylim(0, 1)
    axes[2].set_title(r"Training $R^2$")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig_path = FIG / "part1_convergence.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "runs": rows,
        "r2_limit": R2_LIMIT,
        "var_eps": VAR_EPS,
        "var_y": VAR_Y,
        "figure": str(fig_path),
    }


def coef_figure(phase_result: dict, phase: str):
    pos = phase_result["l1_top_positive"][::-1]
    neg = phase_result["l1_top_negative"][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, rows, title, color in (
        (axes[0], pos, "Largest positive L1 weights", "C0"),
        (axes[1], neg, "Most negative L1 weights", "C3"),
    ):
        ax.barh([r["token"] for r in rows], [r["coef"] for r in rows], color=color)
        ax.set_title(title)
        ax.set_xlabel("coefficient")
    fig.suptitle(f"Phase {phase} criteria unigrams, L1 logistic regression", fontsize=11)
    fig.tight_layout()
    path = FIG / f"phase_{phase}_l1_coefficients.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main():
    print("Part 1")
    p1 = part1()
    print(f"theoretical R2 limit = {R2_LIMIT:.6f}")

    phases = {}
    for phase in ("III", "I", "II"):
        result = run_phase(phase)
        if phase == "III":
            result["l1_figure"] = coef_figure(result, phase)
        phases[phase] = result

    # Kaggle file: Phase I rows, then Phase II, then Phase III, original order.
    submission = pd.concat(
        [phases[p].pop("submission") for p in ("I", "II", "III")],
        ignore_index=True,
    )
    y_tests = [phases[p].pop("y_test") for p in ("I", "II", "III")]
    out_csv = ROOT / "hw2_submission.csv"
    submission.to_csv(out_csv, index=False)
    assert submission["row_id"].is_unique
    assert len(submission) == 3427
    assert set(submission["label"].unique()).issubset({0, 1})

    y_all = np.concatenate(y_tests)
    pred_all = submission["label"].to_numpy()
    overall_f1 = float(f1_score(y_all, pred_all))
    print(f"\nOverall held-out test F1 = {overall_f1:.6f} on {len(submission)} rows")

    payload = {
        "part1": p1,
        "phases": phases,
        "submission": {
            "path": str(out_csv),
            "n_rows": int(len(submission)),
            "n_unique_ids": int(submission["row_id"].nunique()),
            "label_counts": {str(k): int(v) for k, v in submission["label"].value_counts().items()},
            "overall_test_f1": overall_f1,
        },
        "preprocessing": {
            "lowercase": True,
            "lemmatize_or_stem": False,
            "strip_punctuation": True,
            "stopwords": True,
            "kept_negations": sorted(_KEEP),
            "removed_template_words": sorted(_TEMPLATE),
            "drop_pure_numbers": True,
            "M_uni": M_UNI,
            "M_bi": M_BI,
            "C": 1.0,
        },
    }
    out_json = ROOT / "results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out_json)


if __name__ == "__main__":
    main()
