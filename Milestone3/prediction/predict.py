"""Task C: podium prediction with live SHAP (and optional LIME) for the Graph-RAG F1 assistant.

Public API
    prerace_features(race_id, driver_id, constructor_id, grid, quali_pos=None, q1=None, q2=None, q3=None) -> pd.Series
    predict(features, race_id=None, driver_id=None, k=3, lime=False) -> dict
    predict_known(race_id, driver_id, **kw) -> dict        # entry taken from the stored tables
    explain_lime(features, ...) -> dict
    build_background(force=False)                          # one-time, also called lazily
    verify_model()                                         # saved model still reproduces its saved predictions

Nothing here trains anything and nothing is written to the graph.
"""
import functools
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .features import build_modeling_table
except ImportError:                                   # run as a plain script
    from features import build_modeling_table

# ---- locations (override with env vars; clean/ and results/ are the Milestone 1 folders) -----------------------
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("F1_ROOT", HERE.parent))
CLEAN = Path(os.environ.get("F1_CLEAN_DIR", ROOT / "clean"))
RESULTS = Path(os.environ.get("F1_RESULTS_DIR", ROOT / "results"))
BG_FILE = HERE / "shap_background.parquet"            # 100 rows, saved once, reused by every SHAP call
LIME_FILE = HERE / "lime_background.parquet"          # the full reference pool LIME bins on (as in Task 6)
BG_META = HERE / "background_meta.json"               # fill values for LIME + how the sample was drawn

# ---- same constants as the XAI notebook (Task 6) ---------------------------------------------------------------
BACKGROUND_FROM_YEAR = 2014
BACKGROUND_SIZE = 100
N_PERMUTATIONS = 6
LIME_SAMPLES = 5000
BUILDER_TABLES = ["circuits", "constructor_standings", "drivers", "driver_standings", "races",
                  "results", "qualifying", "status", "nationality_country"]


# ---- loading -----------------------------------------------------------------------------------------------------
def _read(directory, name, **csv_kw):
    pq, csv = Path(directory) / f"{name}.parquet", Path(directory) / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():                                  # Task 1 falls back to CSV when pyarrow is missing
        return pd.read_csv(csv, **csv_kw)
    raise FileNotFoundError(f"{name}.parquet/.csv not found in {directory}. Run the Milestone 1 notebook or set F1_CLEAN_DIR.")


@functools.lru_cache(maxsize=1)
def _artifacts():
    """Saved pipeline + threshold + feature list. joblib.load only: the model is never refitted."""
    cfg = json.load(open(CLEAN / "feature_config.json"))
    final = json.load(open(RESULTS / "final_model.json"))
    model = joblib.load(RESULTS / f"{final['final_model']}_model.joblib")
    return dict(model=model, threshold=float(final["threshold"]), key=final["final_model"], label=final.get("label"),
                features=list(cfg["features"]), binary=list(cfg["binary"]), continuous=list(cfg["continuous"]),
                seed=int(cfg.get("seed", 42)))


@functools.lru_cache(maxsize=1)
def _tables():
    t = {n: _read(CLEAN, n) for n in BUILDER_TABLES}
    t["races"]["date"] = pd.to_datetime(t["races"]["date"])          # no-ops for parquet, needed for the CSV fallback
    t["drivers"]["dob"] = pd.to_datetime(t["drivers"]["dob"])
    return t


def _proba(X):
    a = _artifacts()
    return a["model"].predict_proba(pd.DataFrame(X, columns=a["features"]))[:, 1]


# ---- SHAP background, saved once -------------------------------------------------------------------------------
def build_background(force=False):
    """Draw the 100-row SHAP reference with the Task 6 recipe (training rows from 2014 on, seed from feature_config)
    and save it with the LIME reference pool. Does nothing if the files exist, so the background stays fixed."""
    if BG_FILE.exists() and LIME_FILE.exists() and BG_META.exists() and not force:
        return
    a = _artifacts(); F = a["features"]
    mdl = _read(CLEAN, "modeling_table").sort_values(["race_order", "driverId"]).reset_index(drop=True)
    train, val = (mdl[mdl["split"] == s].reset_index(drop=True) for s in ("train", "val"))
    pool = train[train["year"] >= BACKGROUND_FROM_YEAR]
    if len(pool) < 300:                               # same fallback as Task 6
        pool = val
    bg = pool.sample(min(BACKGROUND_SIZE, len(pool)), random_state=a["seed"])[F]
    bg.to_parquet(BG_FILE, index=False)
    pool[F].to_parquet(LIME_FILE, index=False)
    fill = pd.concat([train[a["continuous"]].median(), train[a["binary"]].mode().iloc[0]])   # the pipeline's imputer rule
    json.dump(dict(fill={k: float(v) for k, v in fill.items()}, seed=a["seed"], rows=len(bg),
                   pool_rows=len(pool), pool_years=[int(pool["year"].min()), int(pool["year"].max())]),
              open(BG_META, "w"), indent=2)


@functools.lru_cache(maxsize=1)
def _background():
    build_background()
    return pd.read_parquet(BG_FILE), pd.read_parquet(LIME_FILE), json.load(open(BG_META))


def _shap_explainer(n_perm):
    # built per call on purpose: the explainer's RNG advances with use, so a cached one would explain the same row differently each time
    import shap
    a, bg = _artifacts(), _background()[0]
    masker = shap.maskers.Independent(bg.to_numpy(float), max_samples=len(bg))
    return shap.explainers.Permutation(_proba, masker, feature_names=a["features"], seed=a["seed"]), 2 * len(a["features"]) * n_perm + 1


# ---- pre-race features -------------------------------------------------------------------------------------------
def prerace_features(race_id, driver_id, constructor_id, grid, quali_pos=None, q1=None, q2=None, q3=None):
    """The model's input row for one driver in one race, from build_modeling_table (same leakage rules as training).

    The entry gets a placeholder outcome (positionOrder=99); any stored row of this driver in this race is removed first,
    so the real result cannot reach the features. Form, standings and circuit history only use earlier races/rounds.
    Missing qualifying stays NaN and is filled by the saved pipeline, as for historical rows."""
    F, t = _artifacts()["features"], dict(_tables())
    if int(race_id) not in set(t["races"]["raceId"].astype(int)):
        raise ValueError(f"raceId {race_id} is not in the clean races table")
    res = t["results"]
    res = res[~((res["raceId"].astype(int) == race_id) & (res["driverId"].astype(int) == driver_id))]
    new_id = int(t["results"]["resultId"].max()) + 1
    finished = int(t["status"].loc[t["status"]["status"] == "Finished", "statusId"].iloc[0])
    row = pd.DataFrame([dict(resultId=new_id, raceId=race_id, driverId=driver_id, constructorId=constructor_id,
                             grid=grid, positionOrder=99, statusId=finished)])
    t["results"] = pd.concat([res, row.astype({c: res[c].dtype for c in row.columns if c in res})], ignore_index=True)
    q = t["qualifying"]
    q = q[~((q["raceId"].astype(int) == race_id) & (q["driverId"].astype(int) == driver_id))]
    if quali_pos is not None:
        nan = lambda v: np.nan if v is None else v
        q = pd.concat([q, pd.DataFrame([dict(raceId=race_id, driverId=driver_id, position=quali_pos,
                                             q1=nan(q1), q2=nan(q2), q3=nan(q3))])], ignore_index=True)
    t["qualifying"] = q
    feats, _ = build_modeling_table(t)
    return feats.loc[feats["resultId"] == new_id, F].iloc[0]


def actual_finish(race_id, driver_id):
    """Official finishing position if this driver-race is in the data, else None. Display only, never a model input."""
    r = _tables()["results"]
    hit = r[(r["raceId"].astype(int) == race_id) & (r["driverId"].astype(int) == driver_id)]
    return int(hit["positionOrder"].iloc[0]) if len(hit) else None


# ---- prediction --------------------------------------------------------------------------------------------------
def _to_row(features):
    F = _artifacts()["features"]
    s = features.iloc[0] if isinstance(features, pd.DataFrame) and len(features) == 1 else pd.Series(features)
    absent = [f for f in F if f not in s.index]
    if absent:                                        # a missing VALUE is fine (NaN), a missing KEY is a bug
        raise ValueError(f"features is missing keys: {absent}")
    return s[F].astype(float).to_frame().T


def _top(contrib, row, k, sign):
    c = contrib[contrib * sign > 0].sort_values(ascending=sign < 0).head(k)
    return [dict(feature=f, value=None if pd.isna(row[f].iloc[0]) else float(row[f].iloc[0]), shap=float(v)) for f, v in c.items()]


def predict(features, race_id=None, driver_id=None, k=3, n_perm=N_PERMUTATIONS, lime=False):
    """features: dict / Series / 1-row DataFrame holding the model's features (see prerace_features).

    probability is the saved model's score. The model was trained with up-weighted podiums, so it ranks drivers well
    but is NOT a calibrated chance: compare it with `threshold`, do not read 0.80 as "80 % likely"."""
    a = _artifacts(); F = a["features"]
    x = _to_row(features)
    p = float(_proba(x.to_numpy(float))[0])

    explainer, max_evals = _shap_explainer(n_perm)
    sv = explainer(x.to_numpy(float), max_evals=max_evals, silent=True)         # live, this one row only
    contrib = pd.Series(sv.values[0], index=F)
    base = float(np.ravel(sv.base_values)[0])
    err = abs(base + contrib.sum() - p)
    if err > 1e-6:
        raise RuntimeError(f"SHAP values do not add up to the model score (error {err:.2e})")

    out = dict(probability=p, threshold=a["threshold"], predicted_podium=p >= a["threshold"],
               **{"class": "podium" if p >= a["threshold"] else "not podium"},
               missing_features=[f for f in F if pd.isna(x[f].iloc[0])],
               shap=dict(base_value=base, top_positive=_top(contrib, x, k, +1), top_negative=_top(contrib, x, k, -1)),
               actual_finish=None if race_id is None or driver_id is None else actual_finish(int(race_id), int(driver_id)))
    if lime:
        out["lime"] = explain_lime(x)
    return out


def predict_known(race_id, driver_id, **kw):
    """Predict a driver-race whose grid / qualifying are stored in the tables (past races, tests, the demo)."""
    t = _tables()
    r = t["results"][(t["results"]["raceId"].astype(int) == race_id) & (t["results"]["driverId"].astype(int) == driver_id)]
    if r.empty:
        raise ValueError(f"no stored entry for driver {driver_id} in race {race_id}")
    e = r.iloc[0]
    q = t["qualifying"][(t["qualifying"]["raceId"].astype(int) == race_id) & (t["qualifying"]["driverId"].astype(int) == driver_id)]
    qa = {} if q.empty else dict(quali_pos=int(q.iloc[0]["position"]), q1=q.iloc[0]["q1"], q2=q.iloc[0]["q2"], q3=q.iloc[0]["q3"])
    x = prerace_features(race_id, driver_id, int(e["constructorId"]), int(e["grid"]), **qa)
    return predict(x, race_id, driver_id, **kw)


# ---- LIME (optional, same setup as Task 6) -----------------------------------------------------------------------
def explain_lime(features, num_features=10, num_samples=LIME_SAMPLES, seed=None):
    """LIME needs complete inputs, so missing values are filled with the training median (binary: most frequent),
    the pipeline's own imputer rule; `filled` lists which features that touched. `local_r2` says how far to trust it."""
    from lime.lime_tabular import LimeTabularExplainer
    a = _artifacts(); F = a["features"]
    pool, meta = _background()[1], _background()[2]
    fill = pd.Series(meta["fill"])[F]
    x = _to_row(features).iloc[0]
    explainer = LimeTabularExplainer(pool.fillna(fill).to_numpy(float), feature_names=F,
                                     categorical_features=[F.index(b) for b in a["binary"]],
                                     class_names=["no podium", "podium"], mode="classification",
                                     discretize_continuous=True, random_state=a["seed"] if seed is None else seed)
    e = explainer.explain_instance(x.fillna(fill).to_numpy(float), lambda X: a["model"].predict_proba(pd.DataFrame(X, columns=F)),
                                   num_features=num_features, num_samples=num_samples, labels=(1,))
    rule = {i: d for (d, _), (i, _) in zip(e.as_list(label=1), e.as_map()[1])}
    r2 = e.score.get(1) if isinstance(e.score, dict) else e.score
    return dict(local_r2=None if r2 is None else float(r2), filled=[f for f in F if pd.isna(x[f])],
                weights=[dict(feature=F[i], rule=rule[i], weight=float(w)) for i, w in e.as_map()[1]])


# ---- guard for "never retrain" -----------------------------------------------------------------------------------
def verify_model(n=500, tol=1e-6):
    """Re-score stored val/test rows with the loaded pipeline and compare with the predictions saved at Milestone 1.
    Fails loudly if the library versions changed enough to alter the model."""
    a = _artifacts()
    saved = pd.read_parquet(RESULTS / f"{a['key']}_preds.parquet").head(n)
    mdl = _read(CLEAN, "modeling_table").set_index("resultId")
    diff = np.abs(_proba(mdl.loc[saved["resultId"], a["features"]].to_numpy(float)) - saved["p"].to_numpy())
    assert diff.max() < tol, f"loaded model differs from its saved predictions (max diff {diff.max():.2e})"
    return float(diff.max())


if __name__ == "__main__":
    print("max diff vs saved predictions:", verify_model())
    t = _tables()
    last = int(t["races"].sort_values(["year", "round"])["raceId"].iloc[-1])
    for d in t["results"][t["results"]["raceId"].astype(int) == last].sort_values("grid").query("grid > 0")["driverId"].head(3):
        o = predict_known(last, int(d))
        print(int(d), round(o["probability"], 3), o["class"], "actual:", o["actual_finish"], "| up:",
              [s["feature"] for s in o["shap"]["top_positive"]], "| down:", [s["feature"] for s in o["shap"]["top_negative"]])
