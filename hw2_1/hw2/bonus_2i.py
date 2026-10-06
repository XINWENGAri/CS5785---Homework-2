"""Programming 2(i): pool the three phases, select on validation F1 only.

Test labels are scored once after the pooling method is frozen.
The Kaggle entry for this file must be named so that it contains "Bonus".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score

import hw2_analysis as h
from improve_submission import (
    apply_rule,
    best_threshold,
    fit_model,
    make_vectorizer,
    scores_of,
)

PHASES = ("I", "II", "III")
TOKEN = {"I": "trialphasei", "II": "trialphaseii", "III": "trialphaseiii"}
PREFIX = {"I": "phase1_", "II": "phase2_", "III": "phase3_"}


def load_all():
    data = {}
    for phase in PHASES:
        data[phase] = {}
        for split in ("train", "valid", "test"):
            df = h.load_split(phase, split).copy()
            df["phase"] = phase
            base = df["combined_pp"].fillna("")
            df["marked"] = (TOKEN[phase] + " " + base).str.replace(r"\s+", " ", regex=True)
            data[phase][split] = df
    return data


def y_of(df):
    return df["label"].astype(int).to_numpy()


def concat(data, splits, phases=PHASES):
    frames = [data[p][s] for p in phases for s in splits]
    return pd.concat(frames, ignore_index=True)


def phase_rules(scores, labels):
    rules = {}
    preds = []
    ys = []
    for phase in PHASES:
        t, f, how = best_threshold(labels[phase], scores[phase])
        rules[phase] = {"threshold": t, "how": how, "valid_f1": f}
        preds.append(apply_rule(scores[phase], how, t))
        ys.append(labels[phase])
    overall = float(f1_score(np.concatenate(ys), np.concatenate(preds)))
    return overall, rules


def eval_separate(data, model_name):
    """Shared vocabulary from every phase; a separate classifier per phase."""
    train = concat(data, ("train",))
    vec = make_vectorizer("tfidf_12")
    vec.fit(train["marked"])
    scores, labels = {}, {}
    models = {}
    for phase in PHASES:
        tr = data[phase]["train"]
        va = data[phase]["valid"]
        model = fit_model(model_name, vec.transform(tr["marked"]), y_of(tr))
        models[phase] = model
        scores[phase] = scores_of(model, vec.transform(va["marked"]))
        labels[phase] = y_of(va)
    overall, rules = phase_rules(scores, labels)
    return {
        "name": f"shared_vocab_separate_{model_name}",
        "kind": "separate",
        "model_name": model_name,
        "overall_valid_f1": overall,
        "rules": rules,
        "vec": vec,
        "models": models,
    }


def eval_pooled(data, model_name, one_hot):
    train = concat(data, ("train",))
    vec = make_vectorizer("tfidf_12")
    X = vec.fit_transform(train["marked"])
    if one_hot:
        X = sparse.hstack([X, phase_one_hot(train["phase"])], format="csr")
    model = fit_model(model_name, X, y_of(train))
    scores, labels = {}, {}
    for phase in PHASES:
        va = data[phase]["valid"]
        Xv = vec.transform(va["marked"])
        if one_hot:
            Xv = sparse.hstack([Xv, phase_one_hot(va["phase"])], format="csr")
        scores[phase] = scores_of(model, Xv)
        labels[phase] = y_of(va)
    overall, rules = phase_rules(scores, labels)
    tag = "onehot" if one_hot else "token"
    return {
        "name": f"pooled_{tag}_{model_name}",
        "kind": "pooled",
        "model_name": model_name,
        "one_hot": one_hot,
        "overall_valid_f1": overall,
        "rules": rules,
        "vec": vec,
        "model": model,
        "valid_scores": scores,
    }


def phase_one_hot(phases: pd.Series):
    idx = phases.map({p: i for i, p in enumerate(PHASES)}).to_numpy()
    rows = np.arange(len(idx))
    mat = sparse.csr_matrix((np.ones(len(idx)), (rows, idx)), shape=(len(idx), 3))
    return mat


def eval_blend(data, left, right):
    """Average two already-scored validation score dicts and retune cutoffs."""
    scores, labels = {}, {}
    for phase in PHASES:
        scores[phase] = 0.5 * (left["valid_scores"][phase] + right["valid_scores"][phase])
        labels[phase] = y_of(data[phase]["valid"])
    overall, rules = phase_rules(scores, labels)
    return {
        "name": f"blend_{left['name']}__{right['name']}",
        "kind": "blend",
        "left": left["name"],
        "right": right["name"],
        "overall_valid_f1": overall,
        "rules": rules,
    }


def refit_predict(data, chosen, registry):
    """Refit the frozen choice on train+valid, then apply validation cutoffs."""
    rules = chosen["rules"]
    if chosen["kind"] == "separate":
        pool = concat(data, ("train", "valid"))
        vec = make_vectorizer("tfidf_12")
        vec.fit(pool["marked"])
        preds = {}
        for phase in PHASES:
            rows = pool[pool["phase"] == phase]
            model = fit_model(chosen["model_name"], vec.transform(rows["marked"]), y_of(rows))
            te = data[phase]["test"]
            scores = scores_of(model, vec.transform(te["marked"]))
            rule = rules[phase]
            preds[phase] = apply_rule(scores, rule["how"], rule["threshold"])
        return preds
    if chosen["kind"] == "pooled":
        pool = concat(data, ("train", "valid"))
        vec = make_vectorizer("tfidf_12")
        X = vec.fit_transform(pool["marked"])
        if chosen["one_hot"]:
            X = sparse.hstack([X, phase_one_hot(pool["phase"])], format="csr")
        model = fit_model(chosen["model_name"], X, y_of(pool))
        preds = {}
        for phase in PHASES:
            te = data[phase]["test"]
            Xt = vec.transform(te["marked"])
            if chosen["one_hot"]:
                Xt = sparse.hstack([Xt, phase_one_hot(te["phase"])], format="csr")
            scores = scores_of(model, Xt)
            rule = rules[phase]
            preds[phase] = apply_rule(scores, rule["how"], rule["threshold"])
        return preds
    left = refit_predict(data, registry[chosen["left"]], registry)
    # Blend needs scores, not hard labels. Recompute properly.
    raise RuntimeError("blend refit is handled in refit_blend")


def score_split(data, spec, split):
    """Return per-phase scores for a fitted spec rebuilt inside the caller."""
    raise NotImplementedError


def refit_blend(data, chosen, registry):
    left_scores = pooled_or_separate_scores(data, registry[chosen["left"]], "test")
    right_scores = pooled_or_separate_scores(data, registry[chosen["right"]], "test")
    preds = {}
    for phase in PHASES:
        scores = 0.5 * (left_scores[phase] + right_scores[phase])
        rule = chosen["rules"][phase]
        preds[phase] = apply_rule(scores, rule["how"], rule["threshold"])
    return preds


def pooled_or_separate_scores(data, spec, split):
    pool = concat(data, ("train", "valid"))
    vec = make_vectorizer("tfidf_12")
    if spec["kind"] == "separate":
        vec.fit(pool["marked"])
        out = {}
        for phase in PHASES:
            rows = pool[pool["phase"] == phase]
            model = fit_model(spec["model_name"], vec.transform(rows["marked"]), y_of(rows))
            te = data[phase][split]
            out[phase] = scores_of(model, vec.transform(te["marked"]))
        return out
    X = vec.fit_transform(pool["marked"])
    if spec["one_hot"]:
        X = sparse.hstack([X, phase_one_hot(pool["phase"])], format="csr")
    model = fit_model(spec["model_name"], X, y_of(pool))
    out = {}
    for phase in PHASES:
        te = data[phase][split]
        Xt = vec.transform(te["marked"])
        if spec["one_hot"]:
            Xt = sparse.hstack([Xt, phase_one_hot(te["phase"])], format="csr")
        out[phase] = scores_of(model, Xt)
    return out


def main():
    data = load_all()
    candidates = []
    scored = []
    for model_name in ("lr_1_none", "lr_4_none"):
        print("fitting", model_name, "separate", flush=True)
        separate = eval_separate(data, model_name)
        separate["valid_scores"] = {
            phase: scores_of(
                separate["models"][phase],
                separate["vec"].transform(data[phase]["valid"]["marked"]),
            )
            for phase in PHASES
        }
        candidates.append(separate)
        print(" ", separate["name"], f"{separate['overall_valid_f1']:.4f}", separate["rules"], flush=True)

        for one_hot in (False, True):
            print("fitting", model_name, "pooled", "one_hot" if one_hot else "token", flush=True)
            pooled = eval_pooled(data, model_name, one_hot)
            candidates.append(pooled)
            scored.append(pooled)
            print(" ", pooled["name"], f"{pooled['overall_valid_f1']:.4f}", pooled["rules"], flush=True)

    # Blend the two pooled models with the best validation scores so far.
    scored.sort(key=lambda item: item["overall_valid_f1"], reverse=True)
    if len(scored) >= 2:
        blend = eval_blend(data, scored[0], scored[1])
        candidates.append(blend)
        print(" ", blend["name"], f"{blend['overall_valid_f1']:.4f}", blend["rules"], flush=True)

    registry = {item["name"]: item for item in candidates}
    chosen = max(candidates, key=lambda item: item["overall_valid_f1"])
    print("SELECT", chosen["name"], f"{chosen['overall_valid_f1']:.4f}", flush=True)

    if chosen["kind"] == "blend":
        preds = refit_blend(data, chosen, registry)
    else:
        preds = refit_predict(data, chosen, registry)

    frames = []
    y_parts = []
    pred_parts = []
    per_phase = {}
    for phase in PHASES:
        ids = data[phase]["test"]["NCTID"].astype(str)
        pred = preds[phase].astype(int)
        frames.append(pd.DataFrame({"row_id": PREFIX[phase] + ids, "label": pred}))
        y = y_of(data[phase]["test"]) if "label" in data[phase]["test"].columns else None
        y_parts.append(y)
        pred_parts.append(pred)
        per_phase[phase] = {
            "valid_f1": chosen["rules"][phase]["valid_f1"],
            "threshold": chosen["rules"][phase]["threshold"],
            "how": chosen["rules"][phase]["how"],
            "test_f1": float(f1_score(y, pred)) if y is not None else None,
            "pred_positive_rate": float(pred.mean()),
        }
        print(f"phase {phase} valid {per_phase[phase]['valid_f1']:.4f} test {per_phase[phase]['test_f1']}", flush=True)

    submission = pd.concat(frames, ignore_index=True)
    assert submission["row_id"].is_unique and len(submission) == 3427
    out = h.ROOT / "hw2_submission_bonus.csv"
    submission.to_csv(out, index=False)
    overall = float(f1_score(np.concatenate(y_parts), np.concatenate(pred_parts)))
    print(f"Overall held-out test F1 = {overall:.6f}", flush=True)
    payload = {
        "selected": chosen["name"],
        "overall_valid_f1": chosen["overall_valid_f1"],
        "overall_test_f1": overall,
        "per_phase": per_phase,
        "candidates": [
            {"name": item["name"], "overall_valid_f1": item["overall_valid_f1"]}
            for item in sorted(candidates, key=lambda item: -item["overall_valid_f1"])
        ],
        "label_counts": {str(k): int(v) for k, v in submission["label"].value_counts().items()},
    }
    (h.ROOT / "bonus_2i.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
