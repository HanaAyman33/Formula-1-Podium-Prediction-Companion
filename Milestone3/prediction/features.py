"""Milestone 1 feature builder, copied unchanged from the notebook (Task 3, cell "Feature builder: build_modeling_table").

Do not edit: the saved model was trained on exactly this feature definition.
"""
import numpy as np
import pandas as pd

FORM_WINDOWS = [3, 5, 10]
FAMILIES = ["drv_podium_rate", "drv_finish_rel", "team_podium_rate", "team_finish_rel"]   # core form (finishing positions)
STATUS_FAMILIES = ["drv_dnf_rate", "team_dnf_rate"]                                        # status-derived: candidate only
ALL_FAMILIES = FAMILIES + STATUS_FAMILIES
ID_COLS = ["resultId", "raceId", "driverId", "constructorId", "circuitId", "year", "round", "race_order"]
TARGET = "podium"
BINARY_FEATURES = ["pole", "pit_lane_start", "front_row", "home_race"]
MAX_GAP_PCT = 20      # qualifying times slower than this (% off the session's fastest) are treated as not representative
def _is_slow(t, fastest):   # the single definition of that rule (also used to count ignored times in the next cell)
    return t > fastest * (1 + MAX_GAP_PCT / 100)

def _roll_past(df, keys, col, w):
    # mean of the PREVIOUS w values within each group: shift(1) removes the current row.
    # Needs `df` sorted chronologically (it is: sorted by race_order).
    return df.groupby(keys)[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

def build_modeling_table(tbls, windows=FORM_WINDOWS):
    # ---- calendar: a strict chronological order of races (year, round) ----------------------
    races = tbls["races"][["raceId", "year", "round", "circuitId", "date"]].copy()
    races = races.sort_values(["year", "round"]).reset_index(drop=True)
    races["race_order"] = np.arange(len(races))
    for c in ["raceId", "year", "round", "circuitId"]:
        races[c] = races[c].astype("int64")
    races["n_rounds"] = races.groupby("year")["round"].transform("max")

    r = tbls["results"][["resultId", "raceId", "driverId", "constructorId", "grid",
                         "positionOrder", "statusId"]].copy()
    for c in ["resultId", "raceId", "driverId", "constructorId", "grid", "positionOrder", "statusId"]:
        r[c] = r[c].astype("int64")
    stt = tbls["status"][["statusId", "status"]].copy(); stt["statusId"] = stt["statusId"].astype("int64")
    r["status_name"] = r["statusId"].map(stt.set_index("statusId")["status"]).fillna("")
    circ = tbls["circuits"][["circuitId", "country"]].rename(columns={"country": "circuit_country"}).copy()
    circ["circuitId"] = circ["circuitId"].astype("int64")
    drv = tbls["drivers"][["driverId", "dob", "nationality"]].copy()
    drv["driverId"] = drv["driverId"].astype("int64")

    df = (r.merge(races, on="raceId", how="left", validate="m:1")
           .merge(circ, on="circuitId", how="left", validate="m:1")
           .merge(drv, on="driverId", how="left", validate="m:1"))
    df = df.sort_values(["race_order", "driverId"]).reset_index(drop=True)

    # ---- target and the outcome helpers (outcomes are only ever used through shift) ------------
    df[TARGET] = (df["positionOrder"] <= 3).astype(int)
    df["field_size"] = df.groupby("raceId")["driverId"].transform("size").astype(float)
    df["_podium"] = df[TARGET].astype(float)
    df["_finish_rel"] = df["positionOrder"] / df["field_size"]     # era-independent finishing position
    # retirement = the car stopped: status other than "Finished" / "+n Laps" / "Lapped" (reference document), DSQ excluded.
    # NOT positionText == "R": a driver who retired after 90% of the distance is classified and has a numeric position.
    _fin = df["status_name"].isin(["Finished", "Lapped"]) | df["status_name"].str.startswith("+")
    df["_dnf"] = (~_fin & (df["status_name"] != "Disqualified")).astype(float)

    # ---- GRID group (known when the grid is set) -----------------------------------------------
    max_grid = df.groupby("raceId")["grid"].transform("max")
    back = max_grid + 1
    df["pit_lane_start"] = (df["grid"] == 0).astype(float)
    df["grid_clean"] = np.where(df["grid"] > 0, df["grid"], np.where(max_grid > 0, back, np.nan)).astype(float)
    df["grid_rel"] = df["grid_clean"] / df["field_size"]
    df["front_row"] = (df["grid_clean"] <= 2).astype(float).where(df["grid_clean"].notna())
    df["pole"] = (df["grid_clean"] == 1).astype(float).where(df["grid_clean"].notna())
    gt = df.groupby(["raceId", "constructorId"])["grid_clean"]
    df["team_best_grid"] = gt.transform("min")
    df["grid_vs_team"] = df["grid_clean"] - gt.transform("mean")

    # ---- QUALIFYING group ----------------------------------------------------------------------
    q = tbls["qualifying"][["raceId", "driverId", "position", "q1", "q2", "q3"]].copy()
    q["raceId"] = q["raceId"].astype("int64"); q["driverId"] = q["driverId"].astype("int64")
    q["quali_pos"] = q["position"].astype(float)
    for c in ["q1", "q2", "q3"]:
        q[c] = q[c].astype(float)
        # a time more than MAX_GAP_PCT % slower than the session's fastest is not a representative lap (deleted, aborted or
        # mistimed lap, e.g. a 126.7 s Q2 "lap" at the 2023 Austrian GP): it is ignored, so the gap falls back to an earlier session
        fastest = q.groupby("raceId")[c].transform("min")     # unaffected by the masking below (the fastest time is never slow)
        q.loc[_is_slow(q[c], fastest), c] = np.nan
        q[f"_gap_{c}"] = 100 * (q[c] / fastest - 1)   # % slower than the session's fastest
    q["q1_gap_pct"] = q["_gap_q1"]
    # gap in the LAST session the driver reached (Q3 -> Q2 -> Q1): comparable across the single-lap (2003-05) and knockout (2006+) formats
    q["quali_gap_pct"] = q["_gap_q3"].fillna(q["_gap_q2"]).fillna(q["_gap_q1"])
    df = df.merge(q[["raceId", "driverId", "quali_pos", "q1_gap_pct", "quali_gap_pct"]],
                  on=["raceId", "driverId"], how="left", validate="1:1")
    df["quali_pos_rel"] = df["quali_pos"] / df["field_size"]
    df["grid_minus_quali"] = np.where((df["grid"] > 0) & df["quali_pos"].notna(), df["grid"] - df["quali_pos"], np.nan)

    # ---- FORM group: only previous races ---------------------------------------------------------
    drv_src = {"drv_podium_rate": "_podium", "drv_finish_rel": "_finish_rel", "drv_dnf_rate": "_dnf"}
    team_src = {"team_podium_rate": "_podium", "team_finish_rel": "_finish_rel", "team_dnf_rate": "_dnf"}
    for fam, col in drv_src.items():
        for w in windows:
            df[f"{fam}_w{w}"] = _roll_past(df, "driverId", col, w)
    team = (df.groupby(["constructorId", "raceId", "race_order"], as_index=False)[["_podium", "_finish_rel", "_dnf"]].mean()
              .sort_values(["constructorId", "race_order"]).reset_index(drop=True))
    tcols = []
    for fam, col in team_src.items():
        for w in windows:
            team[f"{fam}_w{w}"] = _roll_past(team, "constructorId", col, w); tcols.append(f"{fam}_w{w}")
    df = df.merge(team[["constructorId", "raceId"] + tcols], on=["constructorId", "raceId"], how="left", validate="m:1")
    g = df.groupby("driverId")
    df["drv_career_starts"] = g.cumcount().astype(float)
    prev_pod = g["_podium"].cumsum() - df["_podium"]
    df["drv_career_podium_rate"] = prev_pod / df["drv_career_starts"].replace(0, np.nan)

    # ---- STANDINGS group: championship table after the previous round of the same season ---------
    def prior_standing(kind, ent, prefix):
        st = tbls[f"{kind}_standings"][["raceId", ent, "position", "points", "wins"]].copy()
        st = st.merge(races[["raceId", "year", "round"]], on="raceId", how="inner")
        for c in ["raceId", ent]: st[c] = st[c].astype("int64")
        lead = st.groupby("raceId")["points"].transform("max").astype(float)
        st[f"{prefix}_pts_share_prev"] = np.where(lead > 0, st["points"].astype(float) / lead, 0.0)
        st[f"{prefix}_champ_pos_prev"] = st["position"].astype(float)
        st[f"{prefix}_wins_prev"] = st["wins"].astype(float)
        cols = [f"{prefix}_champ_pos_prev", f"{prefix}_pts_share_prev", f"{prefix}_wins_prev"]
        st = st.sort_values("round")
        left = df[["resultId", ent, "year", "round"]].sort_values("round")
        # allow_exact_matches=False => strictly EARLIER rounds: the standing after this race is never used
        m = pd.merge_asof(left, st[[ent, "year", "round"] + cols], on="round", by=[ent, "year"],
                          allow_exact_matches=False, direction="backward")
        has_table = m["year"].isin(set(st["year"]))
        for c in cols[1:]:                       # nothing scored yet this season => 0 (position stays NaN)
            m.loc[has_table & m[c].isna(), c] = 0.0
        return m[["resultId"] + cols]
    df = df.merge(prior_standing("driver", "driverId", "drv"), on="resultId", how="left", validate="1:1")
    df = df.merge(prior_standing("constructor", "constructorId", "con"), on="resultId", how="left", validate="1:1")

    # ---- CIRCUIT HISTORY group: previous races only ----------------------------------------------
    gdc = df.groupby(["driverId", "circuitId"])
    df["drv_circuit_prev_starts"] = gdc.cumcount().astype(float)
    df["drv_circuit_prev_podiums"] = gdc["_podium"].cumsum() - df["_podium"]
    df["_front"] = df["front_row"].fillna(0.0)
    df["_front_pod"] = df["_front"] * df["_podium"]
    cr = (df.groupby(["circuitId", "raceId", "race_order"], as_index=False)[["_front", "_front_pod"]].sum()
            .sort_values(["circuitId", "race_order"]).reset_index(drop=True))
    for c in ["_front", "_front_pod"]:     # only the last 8 races at the circuit: early-era reliability was far lower (reference document)
        cr["past" + c] = cr.groupby("circuitId")[c].transform(lambda s: s.shift(1).rolling(8, min_periods=3).sum())
    cr["circuit_front_row_podium_rate"] = np.where(cr["past_front"] >= 6, cr["past_front_pod"] / cr["past_front"], np.nan)
    df = df.merge(cr[["circuitId", "raceId", "circuit_front_row_podium_rate"]],
                  on=["circuitId", "raceId"], how="left", validate="m:1")

    # ---- CONTEXT group ---------------------------------------------------------------------------
    df["season_progress"] = df["round"] / df["n_rounds"]
    df["age"] = (df["date"] - df["dob"]).dt.days / 365.25
    nm = tbls["nationality_country"][["nationality", "country"]].drop_duplicates().assign(home_race=1.0)
    df = df.merge(nm, left_on=["nationality", "circuit_country"], right_on=["nationality", "country"],
                  how="left", validate="m:1").drop(columns="country")
    df["home_race"] = df["home_race"].fillna(0.0)

    groups = {
        "grid": ["grid_clean", "grid_rel", "pole", "pit_lane_start", "front_row", "team_best_grid", "grid_vs_team"],
        "qualifying": ["quali_pos", "quali_pos_rel", "quali_gap_pct", "q1_gap_pct", "grid_minus_quali"],
        "driver_form": [f"{fam}_w{w}" for fam in ["drv_podium_rate", "drv_finish_rel"] for w in windows]
                       + ["drv_career_starts", "drv_career_podium_rate"],
        "constructor_form": [f"{fam}_w{w}" for fam in ["team_podium_rate", "team_finish_rel"] for w in windows],
        "reliability_status": [f"{fam}_w{w}" for fam in STATUS_FAMILIES for w in windows],
        "standings": ["drv_champ_pos_prev", "drv_pts_share_prev", "drv_wins_prev",
                      "con_champ_pos_prev", "con_pts_share_prev", "con_wins_prev"],
        "circuit_history": ["drv_circuit_prev_starts", "drv_circuit_prev_podiums", "circuit_front_row_podium_rate"],
        "context": ["field_size", "season_progress", "age", "home_race"],
    }
    feats = [f for fs in groups.values() for f in fs]
    out = df[ID_COLS + [TARGET] + feats].copy()
    out[feats] = out[feats].astype(float)
    return out.reset_index(drop=True), groups
