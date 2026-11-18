"""Generate report tables and SVG plots from completed, real tuning outputs."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import optuna


def chart(path, title, labels, series, maximum):
    colors = ['#2563eb', '#059669']
    height = 115 + 90 * len(labels)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="24" y="32" font-size="22" font-family="sans-serif">{escape(title)}</text>']
    for j, (name, _) in enumerate(series):
        parts.append(f'<text x="{250+j*230}" y="64" fill="{colors[j]}" font-size="16" font-family="sans-serif">{escape(name)}</text>')
    for i, label in enumerate(labels):
        y = 100 + i * 90
        parts.append(f'<text x="24" y="{y+18}" font-size="16" font-family="sans-serif">{escape(label)}</text>')
        for j, (_, values) in enumerate(series):
            v = values[i]
            w = 500*v/maximum
            parts.append(f'<rect x="250" y="{y+j*29}" width="{w:.2f}" height="22" fill="{colors[j]}"/>')
            parts.append(f'<text x="{260+w:.2f}" y="{y+j*29+17}" font-size="15" font-family="sans-serif">{v:.4f}</text>')
    path.write_text('\n'.join(parts)+ '\n</svg>')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('results/knn_tuning'))
    out = ap.parse_args().out
    result = json.loads((out/'comparison.json').read_text())
    study = optuna.load_study(study_name=None, storage='sqlite:///'+str((out/'study.sqlite3').resolve()))
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert len(study.trials) == result['trials']
    for t in study.trials:
        assert len(t.user_attrs['fold_metrics']) == 5
        assert abs(np.mean([f['macro_f1'] for f in t.user_attrs['fold_metrics']])-t.value) < 1e-12
    b, t = result['baseline'], result['tuned']
    assert t['config'] == study.best_params
    columns = [('CV Macro F1', b['cv_macro_f1'], t['cv_macro_f1']),
               *[(label, b['student_test'][key], t['student_test'][key]) for label,key in [
                   ('Student candidate accuracy','candidate_accuracy'), ('Student Macro F1','candidate_macro_f1'),
                   ('Student Top 1 accuracy','observation_top1_accuracy'), ('Student log loss','log_loss'),
                   ('Student conditional Top 1','observation_top1_accuracy_when_reference_candidate_present'),
                   ('Candidate miss rate','candidate_miss_rate')]],
               ('Full training seconds',b['fit_seconds'],t['fit_seconds']),
               ('Student inference seconds',b['inference_seconds'],t['inference_seconds'])]
    with (out/'baseline_vs_tuned.csv').open('w', newline='') as f:
        writer=csv.writer(f,lineterminator='\n'); writer.writerow(['metric','baseline','tuned','difference'])
        writer.writerows((label,a,z,z-a) for label,a,z in columns)
    chart(out/'score_comparison.svg', 'Baseline and tuned KNN scores',
          ['CV Macro F1','Student Macro F1','Student Top 1'],
          [('Baseline',[b['cv_macro_f1'],b['student_test']['candidate_macro_f1'],b['student_test']['observation_top1_accuracy']]),
           ('Tuned',[t['cv_macro_f1'],t['student_test']['candidate_macro_f1'],t['student_test']['observation_top1_accuracy']])],1)
    chart(out/'cost_comparison.svg','Probability loss and inference cost', ['Log loss','Inference seconds'],
          [('Baseline',[b['student_test']['log_loss'],b['inference_seconds']]),
           ('Tuned',[t['student_test']['log_loss'],t['inference_seconds']])],
          max(b['inference_seconds'],t['inference_seconds'],1)*1.1)
    chart(out/'search_history.svg','Trial mean CV Macro F1', [f'Trial {q.number}' for q in study.trials],
          [('CV score',[q.value for q in study.trials])],1)
    folds = json.loads((out/'folds.json').read_text())
    ids=[tid for f in folds for tid in f['validation_trips']]
    assert len(ids)==len(set(ids))==80
    report = ['# Bayesian hyperparameter optimisation of KNN',
        'This completed experiment tunes the current KNN candidate. Final team model selection remains pending the required five-model comparison; the available XGBoost log does not contain student-test results, and Random Forest results are absent.',
        '## Model choice and objective',
        'KNN had the highest saved mean CV Macro F1 among the three comparable completed baselines (0.8086). The objective was mean candidate Macro F1 over five folds of whole training trips. Macro F1 weights match and non-match classes equally. Top-1 accuracy and log loss remain separate route-matching and probability-quality checks.',
        '## Search space and method',
        'Optuna TPE with seed 17 evaluated '+str(result['trials'])+' trials. Trial 0 reproduced the baseline; four startup trials were followed by adaptive TPE proposals. All trials used the full training data and five fixed trip-grouped folds. Each fold fitted its own StandardScaler. No evaluator data was read.',
        '| Parameter | Search | Reason |','|---|---|---|',
        '| Neighbours | Odd integers 5 to 45 | Local decisions versus smoothing |',
        '| Weights | distance or uniform | Influence of nearby candidates |',
        '| Minkowski p | 1 or 2 | Manhattan versus Euclidean geometry |',
        'KD-tree, leaf size 40, the five shared input features, and classification threshold 0.5 were fixed. Input files exactly match the team compressed feature cache. Baseline and tuned evaluations used the same folds, inputs and machine.',
        '## Selected configuration',
        'Baseline: `'+json.dumps(b['config'])+'`. Best trial: '+str(study.best_trial.number)+'. Selected: `'+json.dumps(t['config'])+'`.',
        '## Measured results',
        '| Measure | Baseline | Tuned | Difference |','|---|---:|---:|---:|',
        *[f'| {label} | {a:.6f} | {z:.6f} | {z-a:+.6f} |' for label,a,z in columns],
        'Scores are fractions; a difference of 0.01 is one percentage point. Lower log loss is better. Timing covers one fit/predict measurement per model and excludes CSV loading, candidate generation and HMM decoding.',
        'Both runs use scikit-learn log_loss. The older team KNN scorer adds 1e-9 inside logarithms, which gives different penalties for exact zero/one probabilities. Compare the paired values here rather than mixing old and new log-loss results. Log loss reflects overall probability quality and does not by itself prove calibration.',
        'The CV-winning configuration was fixed before loading student-test data. Both models were refitted on all 80 training trips and evaluated on the same ten disjoint student-test trips.',
        'Search wall time: '+f"{result['search_seconds']:.2f}"+' seconds, excluding final refits and student-test evaluation.',
        '## Interpretation and limits',
        f"CV Macro F1 changed by {(t['cv_macro_f1']-b['cv_macro_f1'])*100:+.3f} percentage points; student Macro F1 changed by {(t['student_test']['candidate_macro_f1']-b['student_test']['candidate_macro_f1'])*100:+.3f} points and Top-1 by {(t['student_test']['observation_top1_accuracy']-b['student_test']['observation_top1_accuracy'])*100:+.3f} points. These are different objectives and improvements in one do not establish improvement in all.",
        'The best CV score is a model-selection score, not an unbiased nested-CV estimate. Ten trials are a limited search, not proof of a global optimum. Candidate misses are measured from supplied is_match labels and depend on the shared alignment assumptions. Tuning cannot recover reference arcs absent from candidate sets. Route continuity and HMM ablation require the later decoder experiment.',
        'The winning neighbour count is the upper search boundary (45), so a larger search could change the preferred value. No additional search was chosen based on the student-test results. The measured inference increase is a deployment tradeoff; this result does not establish KNN as the final system model.',
        '## Reproduction and evidence',
        'Run `.venv/bin/python hyperparameter_optimisation/tune_knn.py --trials 10 --out results/knn_tuning_new`, then `.venv/bin/python hyperparameter_optimisation/summarize_knn_tuning.py --out results/knn_tuning_new`. Existing study directories are protected against overwrite.',
        '`study.sqlite3` retains the full search, `trials.csv` stores trial parameters/status/duration, `folds.json` stores trip assignments, `comparison.json` stores measured results, input SHA256 hashes and versions, and `knn_tuned.joblib.gz` stores the fitted scaler and classifier. SVG charts and baseline_vs_tuned.csv are generated from those saved results.',
        'Use this section in the single consolidated team PDF; it is not a replacement for the required five-model comparison. AI assistance was used to implement and document the experiment; team members should verify and explain it under the event policy.']
    rendered = '\n\n'.join(report)+'\n'
    rendered = rendered.replace('|\n\n|', '|\n|')
    (out/'TUNING_REPORT.md').write_text(rendered)
    print('Validated all trial objectives, fold assignments and winning parameters; generated report and plots.')


if __name__=='__main__':
    main()
