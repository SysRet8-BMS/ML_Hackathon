#!/usr/bin/env python
"""
PathSnap HMM: KNN + gradient-boosting blend for emission and transition models,
with neighbour-context features.

Emission    : P(candidate is the true arc | 12 features)  -> scaled K-Nearest-Neighbours
Transition  : P(pair is a true consecutive match | pair features) -> scaled K-Nearest-Neighbours
              (replaces the hand-set W_DIRECT/W_SAME/W_TURN/BETA heuristic)
Decoding    : vectorised Viterbi in log space, restarted across observation gaps
Evaluation  : GroupKFold by trip. All probabilities are out-of-fold; the three
              decoding knobs (w_trans, miss_lp) are picked on the TRAIN trips of each
              fold only, then applied to the held-out trips.

Run:  python pathsnap_hmm_fixed.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- CONFIG ----------------
EM_CANDIDATES = ["hmm_data_fast/train_hmm_emissions.csv",
                 "hmm_data/train_hmm_emissions.csv"]
TR_CANDIDATES = ["hmm_data_fast/train_hmm_transitions.csv",
                 "hmm_data/train_hmm_transitions.csv"]
MAX_TOP_K = 8          # upper limit; auto-reduced to what the transition file covers
FOLDS = 5
OUT_DIR = Path("hmm_knn_results")
K_NEIGHBORS = 25       # neighbours used for each probability
SMOOTH = 1.0           # Laplace-style smoothing so probabilities are never exactly 0/1
MAX_TRAIN = 100_000    # KNN is slow: cap its training rows (lower = faster)
KNN_SHARE = 0.5        # blend: 0.5 = half KNN, half gradient boosting (1.0 = pure KNN)

EMISSION = [
    "distance_m", "distance_gap_m", "heading_cos", "heading_abs_diff_rad",
    "gps_speed_mps", "time_gap_s", "candidate_rank", "candidate_count",
    "fraction_along_arc", "arc_length_m", "from_node_distance_m",
    "to_node_distance_m",
]
TRANS_BASE = ["gps_step_distance_m", "network_distance_m", "network_gps_error_m",
              "same_arc", "directly_connected", "turn_angle_abs_rad"]
TRANS_EXTRA = ["src_rank", "dst_rank", "src_dist", "dst_dist", "src_hd", "dst_hd"]

# decoding grid: weight on transition log-prob, log-prob for a missing pair
W_TRANS_GRID = [0.5, 1.0, 2.0, 3.0]
MISS_LP_GRID = [-4.0, -7.0, -10.0, -14.0]


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"None of these exist: {paths}. Run prepare_hmm.py first.")


class SmoothedKNN:
    """StandardScaler + distance-weighted KNN with smoothed probabilities."""
    def __init__(self, k=K_NEIGHBORS, alpha=SMOOTH):
        self.k, self.alpha = k, alpha

    def fit(self, X, y):
        self.pipe = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=self.k, weights="distance", n_jobs=-1))
        self.pipe.fit(X, y)
        self.prior = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = self.pipe.predict_proba(X)[:, 1]
        p = (p * self.k + self.alpha * self.prior) / (self.k + self.alpha)
        return np.column_stack([1 - p, p])


class Blend:
    """KNN (subsampled, smoothed) blended with gradient boosting (all rows)."""
    def __init__(self, share=KNN_SHARE):
        self.share = share

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(0)
        idx = np.arange(len(y))
        if len(idx) > MAX_TRAIN:
            idx = rng.choice(idx, MAX_TRAIN, replace=False)
        self.knn = SmoothedKNN().fit(X[idx], y[idx]) if self.share > 0 else None
        self.gb = None
        if self.share < 1.0:
            self.gb = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42).fit(X, y)
        return self

    def predict_proba(self, X):
        if self.knn is None:
            p = self.gb.predict_proba(X)[:, 1]
        else:
            p = self.knn.predict_proba(X)[:, 1]
            if self.gb is not None:
                p = self.share * p + (1 - self.share) * self.gb.predict_proba(X)[:, 1]
        return np.column_stack([1 - p, p])


def make_model(share=KNN_SHARE):
    return Blend(share)


# ---------------- DATA ----------------
def load():
    em = pd.read_csv(first_existing(EM_CANDIDATES))
    tr = pd.read_csv(first_existing(TR_CANDIDATES))

    for c in ("trip_id", "observation_id", "arc_id", "is_match", "candidate_rank"):
        if c not in em.columns:
            raise ValueError(f"emissions missing column {c}")
    need_tr = ["trip_id", "observation_id", "arc_id", "next_observation_id",
               "next_arc_id", "network_gps_error_m", "directly_connected",
               "same_arc", "turn_angle_abs_rad"]
    miss = [c for c in need_tr if c not in tr.columns]
    if miss:
        raise ValueError(f"transitions missing columns {miss}")

    for c in ("trip_id", "observation_id", "arc_id"):
        em[c] = em[c].astype(np.int64)
    for c in ("trip_id", "observation_id", "arc_id", "next_observation_id", "next_arc_id"):
        tr[c] = tr[c].astype(np.int64)
    em["is_match"] = em["is_match"].astype(int)
    em[EMISSION] = em[EMISSION].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    em = em.drop_duplicates(["trip_id", "observation_id", "arc_id"])
    return em, tr


def detect_top_k(em, tr):
    """The transition file only covers candidates up to some rank. Using more
    candidates than that gives every extra state a 'missing transition' penalty."""
    keys = tr[["trip_id", "observation_id", "arc_id"]].drop_duplicates()
    m = em[["trip_id", "observation_id", "arc_id", "candidate_rank"]].merge(keys)
    covered = int(m.candidate_rank.max()) + 1 if len(m) else MAX_TOP_K
    return min(MAX_TOP_K, covered)


CONTEXT_FEATS = ["prev_dist", "next_dist", "d_over_min", "hd_minus_min", "n_within_15m"]


def add_context(em):
    """Features that look at neighbouring observations and rival candidates."""
    keys = ["trip_id", "observation_id", "arc_id"]
    base = em[keys + ["distance_m"]]
    for name, delta in (("prev", 1), ("next", -1)):
        sh = base.copy()
        sh["observation_id"] = sh["observation_id"] + delta
        sh = sh.rename(columns={"distance_m": f"{name}_dist"})
        em = em.merge(sh, on=keys, how="left")
        em[f"{name}_dist"] = em[f"{name}_dist"].fillna(200.0)
    g = em.groupby(["trip_id", "observation_id"])
    em["d_over_min"] = em.distance_m / (g.distance_m.transform("min") + 1.0)
    em["hd_minus_min"] = em.heading_abs_diff_rad - g.heading_abs_diff_rad.transform("min")
    em["n_within_15m"] = (em.distance_m <= 15).groupby(
        [em.trip_id, em.observation_id]).transform("sum")
    for c in CONTEXT_FEATS:
        if c not in EMISSION:
            EMISSION.append(c)
    return em


def prepare(em, tr, top_k):
    em = em[em.candidate_rank < top_k].sort_values(
        ["trip_id", "observation_id", "candidate_rank"]).reset_index(drop=True)
    em["loc"] = em.groupby(["trip_id", "observation_id"]).cumcount()

    em = add_context(em)
    small = em[["trip_id", "observation_id", "arc_id", "is_match", "candidate_rank",
                "distance_m", "heading_abs_diff_rad", "loc"]]
    src = small.rename(columns={"is_match": "src_match", "candidate_rank": "src_rank",
                                "distance_m": "src_dist", "heading_abs_diff_rad": "src_hd",
                                "loc": "src_loc"})
    dst = small.rename(columns={"observation_id": "next_observation_id", "arc_id": "next_arc_id",
                                "is_match": "dst_match", "candidate_rank": "dst_rank",
                                "distance_m": "dst_dist", "heading_abs_diff_rad": "dst_hd",
                                "loc": "dst_loc"})
    tr = tr[tr.next_observation_id == tr.observation_id + 1]
    tr = (tr.merge(src, on=["trip_id", "observation_id", "arc_id"], how="inner")
            .merge(dst, on=["trip_id", "next_observation_id", "next_arc_id"], how="inner"))
    tr = tr.sort_values(["trip_id", "observation_id"]).reset_index(drop=True)
    tr["y"] = ((tr.src_match == 1) & (tr.dst_match == 1)).astype(int)
    feats = [c for c in TRANS_BASE if c in tr.columns] + TRANS_EXTRA
    tr[feats] = tr[feats].astype(np.float32)
    return em, tr, feats


# ---------------- OUT-OF-FOLD PROBABILITIES ----------------
def oof_predict(em, tr, tfeats, fold_of_trip):
    rng = np.random.default_rng(0)
    pe = np.zeros(len(em))
    pt = np.zeros(len(tr))
    em_fold = em.trip_id.map(fold_of_trip).to_numpy()
    tr_fold = tr.trip_id.map(fold_of_trip).to_numpy()
    Xe = em[EMISSION].to_numpy(np.float32)
    Xt = tr[tfeats].to_numpy(np.float32)
    for f in sorted(set(fold_of_trip.values())):
        print(f"  fitting models for fold {f}...", flush=True)
        te, tt = em_fold != f, tr_fold != f
        m = make_model().fit(Xe[te], em.is_match.to_numpy()[te])
        print("    emission KNN+GB predicting...", flush=True)
        pe[~te] = m.predict_proba(Xe[~te])[:, 1]
        m = make_model(0.0).fit(Xt[tt], tr.y.to_numpy()[tt])
        pt[~tt] = m.predict_proba(Xt[~tt])[:, 1]
    return pe, pt


# ---------------- VITERBI ----------------
class TripData:
    def __init__(self, em_idx, tr_idx, em, tr):
        self.em_idx, self.tr_idx = em_idx, tr_idx
        obs = em.observation_id.to_numpy()[em_idx]
        self.uo, self.st, self.ct = np.unique(obs, return_index=True, return_counts=True)
        self.y = em.is_match.to_numpy()[em_idx]
        o1 = tr.observation_id.to_numpy()[tr_idx]
        order = np.argsort(o1, kind="stable")
        self.tr_idx = tr_idx[order]
        self.o1 = o1[order]
        self.src = tr.src_loc.to_numpy()[self.tr_idx]
        self.dst = tr.dst_loc.to_numpy()[self.tr_idx]
        breaks = np.where(np.diff(self.uo) != 1)[0] + 1
        self.bounds = np.r_[0, breaks, len(self.uo)]


def viterbi(td, pe, pt, w_t, miss_lp):
    e = np.log(np.clip(pe[td.em_idx], 1e-9, 1.0))
    tlp = w_t * np.log(np.clip(pt[td.tr_idx], 1e-9, 1.0))
    miss = w_t * miss_lp
    correct = 0
    for left, right in zip(td.bounds[:-1], td.bounds[1:]):
        dp = e[td.st[left]:td.st[left] + td.ct[left]].copy()
        backs = []
        for k in range(left + 1, right):
            lo = np.searchsorted(td.o1, td.uo[k - 1], "left")
            hi = np.searchsorted(td.o1, td.uo[k - 1], "right")
            M = np.full((td.ct[k - 1], td.ct[k]), miss)
            M[td.src[lo:hi], td.dst[lo:hi]] = tlp[lo:hi]
            cand = dp[:, None] + M
            back = cand.argmax(axis=0)
            dp = cand.max(axis=0) + e[td.st[k]:td.st[k] + td.ct[k]]
            backs.append(back)
        s = int(dp.argmax())
        path = [s]
        for back in reversed(backs):
            s = int(back[s])
            path.append(s)
        path.reverse()
        for off, s in enumerate(path):
            correct += int(td.y[td.st[left + off] + s])
    return correct, int(len(td.uo))


# ---------------- MAIN ----------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading...", flush=True)
    em, tr = load()
    top_k = detect_top_k(em, tr)
    print(f"Using TOP_K = {top_k} (limited by transition file coverage)", flush=True)
    em, tr, tfeats = prepare(em, tr, top_k)
    print(f"Emission rows {len(em):,}; transition rows {len(tr):,} "
          f"(positive rate {tr.y.mean():.3f})", flush=True)

    # Upper bound and baseline
    g = em.groupby(["trip_id", "observation_id"]).is_match.max()
    print(f"Candidate coverage (max possible accuracy): {g.mean() * 100:.2f}%", flush=True)

    trips = np.sort(em.trip_id.unique())
    n_splits = min(FOLDS, len(trips))
    if n_splits < 2:
        raise ValueError("Need at least two trips.")
    gkf = GroupKFold(n_splits=n_splits)
    fold_of_trip = {}
    for f, (_, va) in enumerate(gkf.split(trips, groups=trips)):
        for t in trips[va]:
            fold_of_trip[int(t)] = f

    print("Training out-of-fold models...", flush=True)
    pe, pt = oof_predict(em, tr, tfeats, fold_of_trip)

    # Baseline: independent argmax per observation (what a plain ranker does)
    tmp = em.assign(p=pe)
    best = tmp.loc[tmp.groupby(["trip_id", "observation_id"]).p.idxmax()]
    print(f"Baseline (no HMM, argmax per point): {best.is_match.mean() * 100:.2f}%", flush=True)

    em_groups = em.groupby("trip_id").indices
    tr_groups = tr.groupby("trip_id").indices
    empty = np.array([], dtype=int)
    tds = {int(t): TripData(em_groups[t], tr_groups.get(t, empty), em, tr) for t in trips}

    combos = [(w, m) for w in W_TRANS_GRID for m in MISS_LP_GRID]
    correct = np.zeros((len(trips), len(combos)))
    total = np.zeros(len(trips))
    print(f"Decoding {len(trips)} trips x {len(combos)} settings...", flush=True)
    for i, t in enumerate(trips):
        for j, (w, m) in enumerate(combos):
            c, n = viterbi(tds[int(t)], pe, pt, w, m)
            correct[i, j] = c
            total[i] = n

    trip_fold = np.array([fold_of_trip[int(t)] for t in trips])
    fold_acc, per_trip = [], []
    for f in range(n_splits):
        va = trip_fold == f
        tr_sel = ~va
        j = int(np.argmax(correct[tr_sel].sum(0) / total[tr_sel].sum()))   # tuned on train trips
        acc = correct[va, j].sum() / total[va].sum()
        fold_acc.append(acc)
        print(f"Fold {f + 1}/{n_splits}: accuracy = {acc * 100:.2f}%   "
              f"(w_trans={combos[j][0]}, miss_lp={combos[j][1]})", flush=True)
        for i in np.where(va)[0]:
            per_trip.append({"trip_id": int(trips[i]), "fold": f + 1,
                             "observations": int(total[i]),
                             "accuracy": correct[i, j] / total[i]})

    print("\n========== FINAL HMM RESULTS ==========")
    print(f"HMM Accuracy : {np.mean(fold_acc) * 100:.2f}%  (std {np.std(fold_acc) * 100:.2f})")
    pt_df = pd.DataFrame(per_trip).sort_values("accuracy")
    pt_df.to_csv(OUT_DIR / "per_trip_accuracy.csv", index=False)
    print("\nWorst 10 trips:")
    print(pt_df.head(10).to_string(index=False))

    # Final models on all data
    jbest = int(np.argmax(correct.sum(0)))
    print("\nFitting final models on all trips...", flush=True)
    me = make_model().fit(em[EMISSION].to_numpy(np.float32), em.is_match.to_numpy())
    mt = make_model(0.0).fit(tr[tfeats].to_numpy(np.float32), tr.y.to_numpy())
    dump({"emission_model": me, "transition_model": mt, "emission_features": EMISSION,
          "transition_features": tfeats, "top_k": top_k,
          "w_trans": combos[jbest][0], "miss_lp": combos[jbest][1]},
         OUT_DIR / "hmm_models.joblib")
    json.dump({"cv_fold_accuracy": fold_acc, "top_k": top_k,
               "w_trans": combos[jbest][0], "miss_lp": combos[jbest][1]},
              open(OUT_DIR / "summary.json", "w"), indent=2)
    print("Saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
