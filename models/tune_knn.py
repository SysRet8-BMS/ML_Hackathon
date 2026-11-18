"""Provisional KNN Bayesian tuning; student test is evaluated after selection."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.mlp import DEFAULT_FEATURES, group_folds
from models.logistic_regression import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='train_candidates.csv')
    ap.add_argument('--student-test', default='student_test_candidates.csv')
    ap.add_argument('--out', type=Path, default=Path('results/knn_tuning'))
    ap.add_argument('--trials', type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'study.sqlite3').exists():
        raise ValueError('Choose a new output directory to preserve the previous study')
    if args.trials < 6:
        raise ValueError('Use at least six trials, including adaptive TPE trials')
    train = pd.read_csv(args.train, dtype={'trip_id': str, 'observation_id': str})
    features = list(DEFAULT_FEATURES)
    x = train[features].to_numpy(float)
    y = train.is_match.to_numpy()
    groups = train.trip_id.to_numpy()
    if len(set(groups)) != 80 or set(y) != {0, 1} or not np.isfinite(x).all():
        raise ValueError('Expected 80 training trips with finite features and binary labels')
    folds = list(group_folds(groups, 5))
    manifest = [{'fold': i+1, 'validation_trips': sorted(set(groups[vi]))}
                for i, (_, vi) in enumerate(folds)]
    (args.out / 'folds.json').write_text(json.dumps(manifest, indent=2))
    def make(params):
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            **params, algorithm='kd_tree', leaf_size=40, n_jobs=-1))
    baseline = {'n_neighbors': 15, 'weights': 'distance', 'p': 2}
    def objective(trial):
        params = {'n_neighbors': trial.suggest_int('n_neighbors', 5, 45, step=2),
                  'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
                  'p': trial.suggest_categorical('p', [1, 2])}
        records = []
        for i, (ti, vi) in enumerate(folds):
            model = make(params)
            start = time.perf_counter()
            model.fit(x[ti], y[ti])
            fit = time.perf_counter()-start
            start = time.perf_counter()
            pred = model.predict_proba(x[vi])[:, 1]
            records.append({'fold': i+1, 'macro_f1': float(f1_score(y[vi], pred >= .5, average='macro')),
                            'fit_seconds': fit, 'inference_seconds': time.perf_counter()-start})
            trial.set_user_attr('fold_metrics', records)
        return float(np.mean([r['macro_f1'] for r in records]))
    study = optuna.create_study(direction='maximize',
        storage='sqlite:///' + str((args.out / 'study.sqlite3').resolve()),
        sampler=optuna.samplers.TPESampler(seed=17, n_startup_trials=4))
    study.enqueue_trial(baseline)
    start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials,
                   callbacks=[lambda s,t: s.trials_dataframe().to_csv(args.out / 'trials.csv', index=False)])
    search_seconds = time.perf_counter()-start
    # Only now load the development evaluation partition.
    test = pd.read_csv(args.student_test, dtype={'trip_id': str, 'observation_id': str})
    if len(set(test.trip_id)) != 10 or set(test.trip_id) & set(groups):
        raise ValueError('Expected ten disjoint student-test trips')
    xt, yt = test[features].to_numpy(float), test.is_match.to_numpy()
    if not np.isfinite(xt).all() or not set(yt) <= {0, 1}:
        raise ValueError('Invalid student-test features or labels')
    rows = test.to_dict('records')
    results = {}
    for name, trial in [('baseline', study.trials[0]), ('tuned', study.best_trial)]:
        model = make(trial.params)
        start = time.perf_counter()
        model.fit(x, y)
        fit = time.perf_counter()-start
        start = time.perf_counter()
        prob = model.predict_proba(xt)[:, 1]
        infer = time.perf_counter()-start
        results[name] = {'config': trial.params, 'cv_macro_f1': trial.value,
                         'student_test': metrics(yt, prob, rows),
                         'fit_seconds': fit, 'inference_seconds': infer}
        if name == 'tuned':
            np.savez_compressed(args.out / 'student_probabilities.npz', probability=prob)
            joblib.dump({'model': model, 'features': features}, args.out / 'knn_tuned.joblib.gz', compress=('gzip', 3))
    results.update(search_seconds=search_seconds, trials=len(study.trials), seed=17,
        selection_status='provisional pending five-model comparison',
        versions={'python': platform.python_version(), 'sklearn': sklearn.__version__, 'optuna': optuna.__version__},
        inputs={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.train, args.student_test]})
    (args.out / 'comparison.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
