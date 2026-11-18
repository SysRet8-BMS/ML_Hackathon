"""Distance-weighted KNN candidate classifier with trip-grouped evaluation.

Accepts the same raw trip directories as the MLP (`--train-root` and
`--student-root`) or optional prebuilt candidate CSVs. All scaling is fitted
inside each training fold and on training trips only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from models.mlp import DEFAULT_FEATURES, ID_COLUMNS, LABEL_COLUMN, group_folds, load_csv, load_raw, scores
except ModuleNotFoundError:  # Colab flat-upload fallback.
    from mlp import DEFAULT_FEATURES, ID_COLUMNS, LABEL_COLUMN, group_folds, load_csv, load_raw, scores


def main():
    ap=argparse.ArgumentParser()
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--train-root',type=Path,help='Raw training directory, e.g. datasets/train')
    source.add_argument('--train',type=Path,help='Optional candidate feature CSV')
    ap.add_argument('--student-root',type=Path,help='Raw student-test directory')
    ap.add_argument('--student-test',type=Path,help='Optional student-test candidate CSV')
    ap.add_argument('--out',type=Path,default=Path('results/knn'))
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--neighbors',type=int,default=15)
    ap.add_argument('--weights',choices=['distance','uniform'],default='distance')
    ap.add_argument('--leaf-size',type=int,default=40)
    ap.add_argument('--features',default=','.join(DEFAULT_FEATURES),help='Comma-separated feature columns; defaults to the shared compact set')
    args=ap.parse_args()
    feature_columns=[c.strip() for c in args.features.split(',') if c.strip()] if args.features else None

    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as e:
        raise SystemExit('KNN needs scikit-learn and joblib; install requirements.txt') from e

    if args.train_root:
        if not args.student_root: ap.error('--student-root is required with --train-root')
        train_rows,features,x,y,train_summaries=load_raw(args.train_root,feature_columns)
        test_rows,test_features,xt,yt,test_summaries=load_raw(args.student_root,feature_columns)
    else:
        if not args.student_test: ap.error('--student-test is required with --train')
        train_rows,features,x,y=load_csv(args.train,feature_columns)
        test_rows,test_features,xt,yt=load_csv(args.student_test,feature_columns)
        train_summaries=test_summaries=None
    if features != test_features: raise ValueError('Train/student feature columns must match')
    if len(set(y)) < 2: raise ValueError('Training set must contain both match and non-match candidates')

    groups=[r['trip_id'] for r in train_rows]
    fold_metrics=[]
    for fold,(train_idx,val_idx) in enumerate(group_folds(groups,args.folds),1):
        k=min(args.neighbors,len(train_idx))
        model=make_pipeline(StandardScaler(),KNeighborsClassifier(
            n_neighbors=k,weights=args.weights,algorithm='kd_tree',leaf_size=args.leaf_size,n_jobs=-1))
        start=time.perf_counter();model.fit(x[train_idx],y[train_idx]);fit_s=time.perf_counter()-start
        start=time.perf_counter();p=model.predict_proba(x[val_idx]);infer_s=time.perf_counter()-start
        classes=model[-1].classes_;pos=int(np.flatnonzero(classes==1)[0])
        fold_metrics.append({'fold':fold,'validation_trips':len(set(groups[i] for i in val_idx)),
            'fit_seconds':fit_s,'inference_seconds':infer_s,
            **scores(y[val_idx],p[:,pos],[train_rows[i] for i in val_idx])})

    final=make_pipeline(StandardScaler(),KNeighborsClassifier(
        n_neighbors=min(args.neighbors,len(y)),weights=args.weights,algorithm='kd_tree',
        leaf_size=args.leaf_size,n_jobs=-1))
    start=time.perf_counter();final.fit(x,y);fit_s=time.perf_counter()-start
    start=time.perf_counter();probs=final.predict_proba(xt);infer_s=time.perf_counter()-start
    classes=final[-1].classes_;pos=int(np.flatnonzero(classes==1)[0])
    result={'model':'distance_weighted_knn','features':features,
        'config':{'n_neighbors':args.neighbors,'weights':args.weights,'algorithm':'kd_tree',
                  'leaf_size':args.leaf_size,'scaler':'StandardScaler fit within fold'},
        'cv_folds':fold_metrics,
        'cv_macro_f1_mean':float(np.mean([r['candidate_macro_f1'] for r in fold_metrics])),
        'cv_macro_f1_std':float(np.std([r['candidate_macro_f1'] for r in fold_metrics])),
        'student_test':scores(yt,probs[:,pos],test_rows),
        'fit_seconds_full_train':fit_s,'student_inference_seconds':infer_s}
    if train_summaries is not None:
        n=sum(s['observations'] for s in train_summaries);miss=sum(s['candidate_misses'] for s in train_summaries)
        nt=sum(s['observations'] for s in test_summaries);mt=sum(s['candidate_misses'] for s in test_summaries)
        result['candidate_generation']={'train':train_summaries,'student_test':test_summaries,
            'train_candidate_miss_rate':miss/max(n,1),'student_test_candidate_miss_rate':mt/max(nt,1)}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.out/'fold_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fold_metrics[0]));w.writeheader();w.writerows(fold_metrics)
    joblib.dump({'model':final,'features':features},args.out/'knn_pipeline.joblib')
    print(f'Saved KNN experiment outputs to {args.out}')


if __name__=='__main__': main()
