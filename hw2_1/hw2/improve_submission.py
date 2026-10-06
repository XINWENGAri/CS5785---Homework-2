"""Validation-only resubmission for the clinical-trial Kaggle file.

The required comparison tables stay in hw2_analysis.py. This script only
replaces hw2_submission.csv. Every choice is made from validation F1.
Test labels are read once, after the three phase models are frozen.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.naive_bayes import ComplementNB

import hw2_analysis as h

ROOT = h.ROOT
SEED = h.SEED


def f1_at(y, scores, t):
    pred = (scores >= t).astype(int)
    return float(f1_score(y, pred, zero_division=0))


def best_threshold(y, scores):
    """Pick the cutoff that maximizes validation F1, including the two constants."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate([
        np.linspace(0.02, 0.98, 49),
        np.quantile(scores, np.linspace(0.02, 0.98, 33)),
    ]))
    best_t, best_f = 0.5, f1_at(y, scores, 0.5)
    for t in candidates:
        f = f1_at(y, scores, t)
        if f > best_f + 1e-12:
            best_t, best_f = float(t), f
    all1 = float(f1_score(y, np.ones(len(y), dtype=int), zero_division=0))
    all0 = float(f1_score(y, np.zeros(len(y), dtype=int), zero_division=0))
    if all1 >= best_f:
        return None, all1, "all1"
    if all0 >= best_f:
        return None, all0, "all0"
    return best_t, best_f, "threshold"


def make_vectorizer(kind: str):
    common = dict(stop_words=h.STOP_WORDS, token_pattern=h.TOKEN_PATTERN)
    if kind == "tfidf_12":
        return TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True,
            max_features=50000, **common,
        )
    if kind == "tfidf_1":
        return TfidfVectorizer(
            ngram_range=(1, 1), min_df=2, max_df=0.95, sublinear_tf=True,
            **common,
        )
    if kind == "bin_12":
        return CountVectorizer(
            binary=True, ngram_range=(1, 2), min_df=5, max_features=50000, **common,
        )
    raise ValueError(kind)


def texts_for(df: pd.DataFrame, field: str) -> pd.Series:
    if field == "combined":
        return df["combined_pp"]
    if field == "criteria":
        return df["criteria_pp"]
    if field == "boosted":
        disease = df["diseases_cleaned"].map(h.field_to_text).map(h.preprocess)
        drugs = df["drugs"].map(h.field_to_text).map(h.preprocess)
        extra = (disease + " " + drugs).str.replace(r"\s+", " ", regex=True).str.strip()
        return (df["criteria_pp"].fillna("") + " " + (extra + " ") * 3).str.replace(
            r"\s+", " ", regex=True
        ).str.strip()
    raise ValueError(field)


def fit_model(name, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name.startswith("lr"):
            # lr_C_cw
            _, c_str, cw = name.split("_")
            cw_arg = None if cw == "none" else "balanced"
            model = LogisticRegression(
                C=float(c_str),
                solver="liblinear",
                max_iter=400,
                tol=1e-3,
                class_weight=cw_arg,
                random_state=SEED,
            )
        elif name.startswith("nb"):
            model = ComplementNB(alpha=float(name.split("_")[1]))
        else:
            raise ValueError(name)
        model.fit(X, y)
    return model


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def search_phase(phase: str):
    train = h.load_split(phase, "train")
    valid = h.load_split(phase, "valid")
    test = h.load_split(phase, "test")
    y_train = train["label"].astype(int).to_numpy()
    y_valid = valid["label"].astype(int).to_numpy()

    model_names = [
        "lr_0.5_none", "lr_1_none", "lr_4_none",
        "lr_1_balanced",
        "nb_0.5",
    ]
    rows = []
    kept_scores = []  # (valid_f1, spec, valid_scores) for a later average

    for field in ("combined", "boosted"):
        tr_text = texts_for(train, field)
        va_text = texts_for(valid, field)
        for vec_kind in ("tfidf_12", "tfidf_1"):
            vec = make_vectorizer(vec_kind)
            Xtr = vec.fit_transform(tr_text)
            Xva = vec.transform(va_text)
            for model_name in model_names:
                # ComplementNB on tf-idf is fine; skip the slow balanced+bigram duplicates? keep all.
                model = fit_model(model_name, Xtr, y_train)
                tr_s = scores_of(model, Xtr)
                va_s = scores_of(model, Xva)
                train_f1 = f1_at(y_train, tr_s, 0.5)
                t, valid_f1, how = best_threshold(y_valid, va_s)
                pr = float(average_precision_score(y_valid, va_s))
                roc = float(roc_auc_score(y_valid, va_s))
                memorized = train_f1 >= 0.995
                spec = {
                    "field": field,
                    "vec": vec_kind,
                    "model": model_name,
                    "threshold": t,
                    "how": how,
                    "train_f1_at_0.5": train_f1,
                    "valid_f1": valid_f1,
                    "valid_pr": pr,
                    "valid_roc": roc,
                    "memorized": memorized,
                    "n_features": int(Xtr.shape[1]),
                }
                rows.append(spec)
                if not memorized:
                    kept_scores.append((valid_f1, pr, spec, va_s))
                print(
                    f"  {phase} {field:9s} {vec_kind:9s} {model_name:16s} "
                    f"trainF1 {train_f1:.3f} validF1 {valid_f1:.3f} ({how}) "
                    f"ROC {roc:.3f} feats {Xtr.shape[1]}"
                )

    # Constant predictors, visible from the validation labels alone.
    all1 = float(f1_score(y_valid, np.ones(len(y_valid), dtype=int)))
    rows.append({
        "field": "none", "vec": "none", "model": "all1", "threshold": None,
        "how": "all1", "train_f1_at_0.5": all1, "valid_f1": all1,
        "valid_pr": float(y_valid.mean()), "valid_roc": 0.5,
        "memorized": False, "n_features": 0,
    })
    print(f"  {phase} constant all1 validF1 {all1:.3f}")

    eligible = [r for r in rows if not r["memorized"]]
    eligible.sort(key=lambda r: (r["valid_f1"], r["valid_pr"]), reverse=True)
    best = eligible[0]

    # Average the validation scores of the top three distinct models.
    kept_scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top = []
    seen = set()
    for valid_f1, pr, spec, va_s in kept_scores:
        key = (spec["field"], spec["vec"], spec["model"])
        if key in seen:
            continue
        seen.add(key)
        top.append((spec, va_s))
        if len(top) == 3:
            break
    if len(top) == 3:
        avg = np.mean([va_s for _, va_s in top], axis=0)
        t, f, how = best_threshold(y_valid, avg)
        pr = float(average_precision_score(y_valid, avg))
        print(f"  {phase} ensemble top3 validF1 {f:.3f} ({how})")
        if f > best["valid_f1"] + 1e-12:
            best = {
                "field": "ensemble",
                "vec": "ensemble",
                "model": "top3",
                "threshold": t,
                "how": how,
                "train_f1_at_0.5": None,
                "valid_f1": f,
                "valid_pr": pr,
                "valid_roc": float(roc_auc_score(y_valid, avg)),
                "memorized": False,
                "n_features": None,
                "members": [s for s, _ in top],
            }

    print(f"  SELECT {phase}: {best['field']} {best['vec']} {best['model']} "
          f"validF1 {best['valid_f1']:.4f} how={best['how']} t={best['threshold']}")

    pred, detail = refit_predict(best, train, valid, test)
    return {
        "selected": {k: v for k, v in best.items() if k != "members"} | (
            {"members": best["members"]} if "members" in best else {}
        ),
        "leaderboard": [
            {k: r[k] for k in ("field", "vec", "model", "valid_f1", "valid_roc", "valid_pr", "how", "memorized", "n_features")}
            for r in eligible[:8]
        ],
        "pred": pred,
        "detail": detail,
        "y_test": test["label"].astype(int).to_numpy() if "label" in test.columns else None,
        "ids": test["NCTID"].astype(str).to_numpy(),
    }


def refit_one(spec, train, valid, test):
    field, vec_kind, model_name = spec["field"], spec["vec"], spec["model"]
    pool = pd.concat([train, valid], ignore_index=True)
    y_pool = pool["label"].astype(int).to_numpy()
    vec = make_vectorizer(vec_kind)
    X_pool = vec.fit_transform(texts_for(pool, field))
    X_test = vec.transform(texts_for(test, field))
    model = fit_model(model_name, X_pool, y_pool)
    return scores_of(model, X_test), int(X_pool.shape[1])


def apply_rule(scores, how, threshold):
    if how == "all1":
        return np.ones(len(scores), dtype=int)
    if how == "all0":
        return np.zeros(len(scores), dtype=int)
    return (scores >= threshold).astype(int)


def refit_predict(best, train, valid, test):
    if best["model"] == "all1":
        pred = np.ones(len(test), dtype=int)
        return pred, {"n_features": 0}
    if best.get("members"):
        parts = []
        widths = []
        for spec in best["members"]:
            scores, width = refit_one(spec, train, valid, test)
            parts.append(scores)
            widths.append(width)
        scores = np.mean(parts, axis=0)
        pred = apply_rule(scores, best["how"], best["threshold"])
        return pred, {"n_features": widths, "members": best["members"]}
    scores, width = refit_one(best, train, valid, test)
    pred = apply_rule(scores, best["how"], best["threshold"])
    return pred, {"n_features": width}


def main():
    phases = {}
    frames = []
    y_parts = []
    for phase in ("I", "II", "III"):
        print(f"\n======== Phase {phase} ========")
        result = search_phase(phase)
        phases[phase] = {
            "selected": result["selected"],
            "leaderboard": result["leaderboard"],
            "refit": result["detail"],
        }
        prefix = {"I": "phase1_", "II": "phase2_", "III": "phase3_"}[phase]
        frames.append(pd.DataFrame({
            "row_id": prefix + pd.Series(result["ids"]).astype(str),
            "label": result["pred"].astype(int),
        }))
        y_parts.append(result["y_test"])

    submission = pd.concat(frames, ignore_index=True)
    assert submission["row_id"].is_unique
    assert len(submission) == 3427
    out = ROOT / "hw2_submission.csv"
    submission.to_csv(out, index=False)

    y_all = np.concatenate(y_parts)
    pred_all = submission["label"].to_numpy()
    overall = float(f1_score(y_all, pred_all))
    print(f"\nOverall held-out test F1 = {overall:.6f}")
    offset = 0
    per_phase = {}
    for phase, y in zip(("I", "II", "III"), y_parts):
        n = len(y)
        f = float(f1_score(y, pred_all[offset:offset + n]))
        per_phase[phase] = f
        print(f"  phase {phase} test F1 {f:.4f}  n {n}  pred+ {(pred_all[offset:offset+n] == 1).mean():.3f}")
        offset += n
    payload = {"phases": phases, "overall_test_f1": overall, "per_phase_test_f1": per_phase,
               "n_rows": int(len(submission)),
               "label_counts": {str(k): int(v) for k, v in submission["label"].value_counts().items()}}
    (ROOT / "submission_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
