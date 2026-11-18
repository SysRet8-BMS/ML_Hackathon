#!/usr/bin/env python
"""
PathSnap HMM v3  --  built to push Top-1 observation accuracy past 90%.

What is new compared with pathsnap_hmm_knn.py
---------------------------------------------
1. DIRECTION-AWARE CONTEXT FEATURES (biggest expected gain).
   Roads usually exist as two directed arcs (A->B, B->A) with identical geometry.
   The single-step heading feature cannot separate them when GPS is noisy or slow.
   For every candidate we now measure how far the vehicle moved ALONG that arc over
   +-1,+-2,+-3 observations (positive = moving with the arc's direction) and whether
   the same arc is still close at those neighbouring observations.
2. TWIN-ARC features: how much better is this arc's heading than its reverse twin.
3. SMOOTHED SPEED over a 5-observation window (heading is only trusted when moving).
4. POSTERIOR (forward-backward) DECODING in addition to Viterbi. Posterior decoding
   maximises the expected number of correct points, which is exactly Top-1 accuracy.
5. ERROR BREAKDOWN so you can see what is still wrong:
      candidate_miss  -> true arc was never a candidate (fix candidate_features.py)
      near_tie        -> chosen arc within 2 m of the true arc (parallel / twin / junction)
      other           -> genuinely wrong choice
Inference rule respected: only GPS + network data are used, never the .route file.

Run:  python pathsnap_hmm_v3.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- CONFIG ----------------
EM_CANDIDATES = ["hmm_data_fast/train_hmm_emissions.csv", "hmm_data/train_hmm_emissions.csv"]
TR_CANDIDATES = ["hmm_data_fast/train_hmm_transitions.csv", "hmm_data/train_hmm_transitions.csv"]
MAX_TOP_K = 8
FOLDS = 5
OUT_DIR = Path("hmm_v3_results")
KNN_SHARE = 0.0        # 0.0 = gradient boosting only (fast). e.g. 0.3 blends in KNN for emissions
K_NEIGHBORS = 25
MAX_KNN_TRAIN = 100_000

EMISSION = [
    "distance_m", "distance_gap_m", "heading_cos", "heading_abs_diff_rad",
    "gps_speed_mps", "time_gap_s", "candidate_rank", "candidate_count",
    "fraction_along_arc", "arc_length_m", "from_node_distance_m", "to_node_distance_m",
]
TRANS_BASE = ["gps_step_distance_m", "network_distance_m", "network_gps_error_m",
              "same_arc", "directly_connected", "turn_angle_abs_rad"]
TRANS_EXTRA = ["src_rank", "dst_rank", "src_dist", "dst_dist", "src_hd", "dst_hd"]

W_TRANS_GRID = [0.5, 1.0, 2.0]
MISS_LP_GRID = [-5.0, -9.0, -14.0]
MODES = ["viterbi", "posterior"]


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"None of these exist: {paths}. Run prepare_hmm.py first.")


# ---------------- MODELS ----------------
class Model:
    """Gradient boosting, optionally blended with a smoothed KNN."""
    def __init__(self, knn_share=0.0):
        self.share = knn_share

    def fit(self, X, y):
        y = np.asarray(y)
        self.knn = None
        if self.share > 0:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            rng = np.random.default_rng(0)
            idx = np.arange(len(y))
            if len(idx) > MAX_KNN_TRAIN:
                idx = rng.choice(idx, MAX_KNN_TRAIN, replace=False)
            self.prior = float(y[idx].mean())
            self.knn = make_pipeline(StandardScaler(), KNeighborsClassifier(
                n_neighbors=K_NEIGHBORS, weights="distance", n_jobs=-1)).fit(X[idx], y[idx])
        self.gb = None
        if self.share < 1.0:
            self.gb = HistGradientBoostingClassifier(
                max_iter=600, learning_rate=0.06, max_leaf_nodes=48, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                random_state=42).fit(X, y)
        return self

    def predict_proba(self, X):
        p = 0.0
        if self.knn is not None:
            pk = self.knn.predict_proba(X)[:, 1]
            pk = (pk * K_NEIGHBORS + self.prior) / (K_NEIGHBORS + 1.0)
            p = p + self.share * pk
        if self.gb is not None:
            p = p + (1 - self.share) * self.gb.predict_proba(X)[:, 1]
        return np.column_stack([1 - p, p])


# ---------------- DATA ----------------
def load():
    em = pd.read_csv(first_existing(EM_CANDIDATES))
    tr = pd.read_csv(first_existing(TR_CANDIDATES))
    for c in ("trip_id", "observation_id", "arc_id", "is_match", "candidate_rank"):
        if c not in em.columns:
            raise ValueError(f"emissions missing column {c}")
    need_tr = ["trip_id", "observation_id", "arc_id", "next_observation_id", "next_arc_id",
               "network_gps_error_m", "directly_connected", "same_arc", "turn_angle_abs_rad"]
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
    keys = tr[["trip_id", "observation_id", "arc_id"]].drop_duplicates()
    m = em[["trip_id", "observation_id", "arc_id", "candidate_rank"]].merge(keys)
    covered = int(m.candidate_rank.max()) + 1 if len(m) else MAX_TOP_K
    return min(MAX_TOP_K, covered)


CONTEXT_FEATS = []


def add_context(em):
    """Context features from neighbouring observations, rival candidates and twin arcs.
    Uses only GPS/network-derived columns (inference-legal). `em` must be sorted by
    trip_id, observation_id, candidate_rank with a unique index."""
    keys = ["trip_id", "observation_id", "arc_id"]
    em = em.copy()
    em["pos_m"] = em.fraction_along_arc * em.arc_length_m
    base = em[keys + ["distance_m", "pos_m"]]
    new = []
    for k in (1, 2, 3):
        for side, delta in (("p", k), ("n", -k)):          # p = past, n = next
            sh = base.copy()
            sh["observation_id"] = sh["observation_id"] + delta
            sh = sh.rename(columns={"distance_m": f"{side}{k}_dist", "pos_m": f"{side}{k}_pos"})
            em = em.merge(sh, on=keys, how="left")
            em[f"{side}{k}_dist"] = em[f"{side}{k}_dist"].fillna(200.0)
            if side == "p":      # progress made along this arc since k steps ago
                em[f"{side}{k}_prog"] = (em.pos_m - em[f"{side}{k}_pos"]).fillna(0.0)
            else:                # progress that will be made along this arc in k steps
                em[f"{side}{k}_prog"] = (em[f"{side}{k}_pos"] - em.pos_m).fillna(0.0)
            em = em.drop(columns=[f"{side}{k}_pos"])
            new += [f"{side}{k}_dist", f"{side}{k}_prog"]
    # total signed progress along the arc over +-3 steps: strong direction signal
    em["prog_total3"] = em.p3_prog + em.n3_prog
    em["prog_total1"] = em.p1_prog + em.n1_prog
    new += ["prog_total3", "prog_total1"]

    g = em.groupby(["trip_id", "observation_id"])
    em["d_over_min"] = em.distance_m / (g.distance_m.transform("min") + 1.0)
    em["hd_minus_min"] = em.heading_abs_diff_rad - g.heading_abs_diff_rad.transform("min")
    em["n_within_15m"] = (em.distance_m <= 15).groupby(
        [em.trip_id, em.observation_id]).transform("sum")
    new += ["d_over_min", "hd_minus_min", "n_within_15m"]

    # twin (reverse-direction) arcs: same distance/length, from/to swapped
    lo = np.minimum(em.from_node_distance_m, em.to_node_distance_m).round(0)
    hi = np.maximum(em.from_node_distance_m, em.to_node_distance_m).round(0)
    tk = pd.DataFrame({"t": em.trip_id, "o": em.observation_id,
                       "d": em.distance_m.round(0), "l": em.arc_length_m.round(0),
                       "lo": lo, "hi": hi})
    tg = em.groupby([tk.t, tk.o, tk.d, tk.l, tk.lo, tk.hi]).heading_abs_diff_rad
    em["n_twin"] = tg.transform("size") - 1
    em["hd_minus_twin_min"] = em.heading_abs_diff_rad - tg.transform("min")
    new += ["n_twin", "hd_minus_twin_min"]

    # smoothed speed (per observation, centred 5-window), merged back
    obs = em.groupby(["trip_id", "observation_id"], sort=True).gps_speed_mps.first().reset_index()
    obs["speed_smooth"] = obs.groupby("trip_id").gps_speed_mps.transform(
        lambda s: s.rolling(5, center=True, min_periods=1).mean())
    em = em.merge(obs[["trip_id", "observation_id", "speed_smooth"]],
                  on=["trip_id", "observation_id"], how="left")
    new += ["speed_smooth"]

    em = em.drop(columns=["pos_m"])
    for c in new:
        if c not in CONTEXT_FEATS:
            CONTEXT_FEATS.append(c)
    return em


def prepare(em, tr, top_k):
    em = em[em.candidate_rank < top_k].sort_values(
        ["trip_id", "observation_id", "candidate_rank"]).reset_index(drop=True)
    em["loc"] = em.groupby(["trip_id", "observation_id"]).cumcount()
    em = add_context(em)
    feats_em = EMISSION + CONTEXT_FEATS
    em[feats_em] = em[feats_em].astype(np.float32)

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
    tfeats = [c for c in TRANS_BASE if c in tr.columns] + TRANS_EXTRA
    tr[tfeats] = tr[tfeats].astype(np.float32)
    return em, tr, feats_em, tfeats


# ---------------- OUT-OF-FOLD PROBABILITIES ----------------
def oof_predict(em, tr, efeats, tfeats, fold_of_trip):
    pe = np.zeros(len(em))
    pt = np.zeros(len(tr))
    em_fold = em.trip_id.map(fold_of_trip).to_numpy()
    tr_fold = tr.trip_id.map(fold_of_trip).to_numpy()
    Xe = em[efeats].to_numpy(np.float32)
    Xt = tr[tfeats].to_numpy(np.float32)
    ye, yt = em.is_match.to_numpy(), tr.y.to_numpy()
    for f in sorted(set(fold_of_trip.values())):
        print(f"  fitting models for fold {f + 1}...", flush=True)
        te, tt = em_fold != f, tr_fold != f
        m = Model(KNN_SHARE).fit(Xe[te], ye[te])
        pe[~te] = m.predict_proba(Xe[~te])[:, 1]
        m = Model(0.0).fit(Xt[tt], yt[tt])
        pt[~tt] = m.predict_proba(Xt[~tt])[:, 1]
    return pe, pt


# ---------------- DECODING ----------------
def lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class TripData:
    def __init__(self, em_idx, tr_idx, em, tr):
        self.em_idx = em_idx
        obs = em.observation_id.to_numpy()[em_idx]
        self.uo, self.st, self.ct = np.unique(obs, return_index=True, return_counts=True)
        self.y = em.is_match.to_numpy()[em_idx]
        self.d = em.distance_m.to_numpy()[em_idx]
        o1 = tr.observation_id.to_numpy()[tr_idx]
        order = np.argsort(o1, kind="stable")
        self.tr_idx = tr_idx[order]
        self.o1 = o1[order]
        self.src = tr.src_loc.to_numpy()[self.tr_idx]
        self.dst = tr.dst_loc.to_numpy()[self.tr_idx]
        breaks = np.where(np.diff(self.uo) != 1)[0] + 1
        self.bounds = np.r_[0, breaks, len(self.uo)]


def decode(td, pe, pt, w_t, miss_lp, mode):
    """Return, for each observation, the chosen row (index local to this trip)."""
    e = np.log(np.clip(pe[td.em_idx], 1e-9, 1.0))
    tlp = w_t * np.log(np.clip(pt[td.tr_idx], 1e-9, 1.0))
    miss = w_t * miss_lp
    chosen = np.zeros(len(td.uo), dtype=int)
    for left, right in zip(td.bounds[:-1], td.bounds[1:]):
        Ms, es = [], []
        for k in range(left, right):
            es.append(e[td.st[k]:td.st[k] + td.ct[k]])
        for k in range(left + 1, right):
            lo = np.searchsorted(td.o1, td.uo[k - 1], "left")
            hi = np.searchsorted(td.o1, td.uo[k - 1], "right")
            M = np.full((td.ct[k - 1], td.ct[k]), miss)
            M[td.src[lo:hi], td.dst[lo:hi]] = tlp[lo:hi]
            Ms.append(M)
        n = right - left
        if mode == "viterbi":
            dp = es[0].copy()
            backs = []
            for t in range(1, n):
                cand = dp[:, None] + Ms[t - 1]
                backs.append(cand.argmax(axis=0))
                dp = cand.max(axis=0) + es[t]
            s = int(dp.argmax())
            path = [s]
            for back in reversed(backs):
                s = int(back[s])
                path.append(s)
            path.reverse()
        else:  # posterior: forward-backward, pick the most probable state per step
            alpha = [es[0].copy()]
            for t in range(1, n):
                alpha.append(lse(alpha[-1][:, None] + Ms[t - 1], 0) + es[t])
            beta = [np.zeros_like(es[-1])]
            for t in range(n - 2, -1, -1):
                beta.append(lse(Ms[t] + (es[t + 1] + beta[-1])[None, :], 1))
            beta.reverse()
            path = [int(np.argmax(alpha[t] + beta[t])) for t in range(n)]
        for off, s in enumerate(path):
            chosen[left + off] = td.st[left + off] + s
    return chosen


# ---------------- MAIN ----------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading...", flush=True)
    em, tr = load()
    top_k = detect_top_k(em, tr)
    print(f"Using TOP_K = {top_k} (limited by transition file coverage)", flush=True)
    print("Building context features...", flush=True)
    em, tr, efeats, tfeats = prepare(em, tr, top_k)
    print(f"Emission rows {len(em):,} ({len(efeats)} features); transition rows {len(tr):,} "
          f"(positive rate {tr.y.mean():.3f})", flush=True)

    cov = em.groupby(["trip_id", "observation_id"]).is_match.max().mean()
    print(f"Candidate coverage (max possible accuracy): {cov * 100:.2f}%", flush=True)

    trips = np.sort(em.trip_id.unique())
    n_splits = min(FOLDS, len(trips))
    if n_splits < 2:
        raise ValueError("Need at least two trips.")
    fold_of_trip = {}
    for f, (_, va) in enumerate(GroupKFold(n_splits=n_splits).split(trips, groups=trips)):
        for t in trips[va]:
            fold_of_trip[int(t)] = f

    print("Training out-of-fold models...", flush=True)
    pe, pt = oof_predict(em, tr, efeats, tfeats, fold_of_trip)

    tmp = em.assign(p=pe)
    best = tmp.loc[tmp.groupby(["trip_id", "observation_id"]).p.idxmax()]
    print(f"Baseline (no HMM, argmax per point): {best.is_match.mean() * 100:.2f}%", flush=True)

    em_groups = em.groupby("trip_id").indices
    tr_groups = tr.groupby("trip_id").indices
    empty = np.array([], dtype=int)
    tds = {int(t): TripData(em_groups[t], tr_groups.get(t, empty), em, tr) for t in trips}

    combos = [(w, m, md) for md in MODES for w in W_TRANS_GRID for m in MISS_LP_GRID]
    correct = np.zeros((len(trips), len(combos)))
    total = np.zeros(len(trips))
    print(f"Decoding {len(trips)} trips x {len(combos)} settings...", flush=True)
    for i, t in enumerate(trips):
        td = tds[int(t)]
        total[i] = len(td.uo)
        for j, (w, m, md) in enumerate(combos):
            rows = decode(td, pe, pt, w, m, md)
            correct[i, j] = td.y[rows].sum()

    trip_fold = np.array([fold_of_trip[int(t)] for t in trips])
    fold_acc, per_trip = [], []
    err = {"candidate_miss": 0, "near_tie": 0, "other": 0}
    for f in range(n_splits):
        va = trip_fold == f
        j = int(np.argmax(correct[~va].sum(0) / total[~va].sum()))   # tuned on train trips only
        acc = correct[va, j].sum() / total[va].sum()
        fold_acc.append(acc)
        w, m, md = combos[j]
        print(f"Fold {f + 1}/{n_splits}: accuracy = {acc * 100:.2f}%   "
              f"(decoder={md}, w_trans={w}, miss_lp={m})", flush=True)
        for i in np.where(va)[0]:
            td = tds[int(trips[i])]
            per_trip.append({"trip_id": int(trips[i]), "fold": f + 1,
                             "observations": int(total[i]),
                             "accuracy": correct[i, j] / total[i]})
            rows = decode(td, pe, pt, w, m, md)
            for k, r in enumerate(rows):
                if td.y[r] == 1:
                    continue
                grp = slice(td.st[k], td.st[k] + td.ct[k])
                if td.y[grp].sum() == 0:
                    err["candidate_miss"] += 1
                else:
                    true_d = td.d[grp][td.y[grp] == 1][0]
                    err["near_tie" if abs(td.d[r] - true_d) <= 2.0 else "other"] += 1

    print("\n========== FINAL HMM RESULTS ==========")
    print(f"HMM Accuracy : {np.mean(fold_acc) * 100:.2f}%  (std {np.std(fold_acc) * 100:.2f})")
    n_err = max(sum(err.values()), 1)
    print("\nWhere the remaining errors come from:")
    for k, v in err.items():
        print(f"  {k:15s} {v:8d}  ({v / n_err * 100:5.1f}% of errors)")
    pt_df = pd.DataFrame(per_trip).sort_values("accuracy")
    pt_df.to_csv(OUT_DIR / "per_trip_accuracy.csv", index=False)
    print("\nWorst 10 trips:")
    print(pt_df.head(10).to_string(index=False))

    jbest = int(np.argmax(correct.sum(0)))
    print("\nFitting final models on all trips...", flush=True)
    me = Model(KNN_SHARE).fit(em[efeats].to_numpy(np.float32), em.is_match.to_numpy())
    mt = Model(0.0).fit(tr[tfeats].to_numpy(np.float32), tr.y.to_numpy())
    w, m, md = combos[jbest]
    dump({"emission_model": me, "transition_model": mt, "emission_features": efeats,
          "transition_features": tfeats, "top_k": top_k,
          "w_trans": w, "miss_lp": m, "decoder": md}, OUT_DIR / "hmm_models.joblib")
    json.dump({"cv_fold_accuracy": fold_acc, "top_k": top_k, "w_trans": w, "miss_lp": m,
               "decoder": md, "error_breakdown": err},
              open(OUT_DIR / "summary.json", "w"), indent=2)
    print("Saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
