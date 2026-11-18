# PathSnap — Intelligent Road Matching Using Machine Learning

## 1. Project Overview

PathSnap is a machine-learning-based road matching system that maps GPS trajectory observations to the most likely road segments in a road network.

The project explores supervised classification and sequence-based inference to improve road-segment identification, trajectory consistency and route reconstruction.

The system is designed around a shared candidate-generation and feature-engineering pipeline, multiple machine-learning classifiers, and a planned Hidden Markov Model (HMM) with Viterbi decoding for sequential route reconstruction.

## 2. Project Objectives

- Match GPS observations to the most likely road-network arcs.
- Generate candidate road segments using spatial proximity.
- Engineer spatial, directional and movement-related features.
- Compare multiple machine-learning classification approaches.
- Evaluate generalisation using trip-grouped cross-validation.
- Evaluate development performance on the 10 student-testing trips.
- Improve route continuity through sequence-based inference.
- Provide route visualisation and support eventual API/UI integration.

## 3. System Architecture

The intended processing pipeline is:

1. **Input data:** GPS trajectories and road-network geometry.
2. **Candidate generation:** Identify nearby road-segment candidates for each GPS observation.
3. **Feature engineering:** Calculate distances, heading similarity, GPS speed and candidate rank.
4. **Classification:** Predict the probability that each candidate is the correct road segment.
5. **Sequence inference:** Use HMM/Viterbi decoding to obtain a temporally consistent road sequence.
6. **Evaluation:** Compare candidate classification and reconstructed routes using appropriate metrics.
7. **Visualisation:** Display GPS trajectories, road networks and predicted routes.

## 4. Machine Learning Models

The repository includes the following model implementations:

| Model | Implementation |
|---|---|
| Multi-Layer Perceptron (MLP) | `models/mlp.py` |
| K-Nearest Neighbours (KNN) | `models/knn.py` |
| XGBoost | `models/train_xgboost.py` |
| Random Forest | `models/RF.py` |
| Logistic Regression | `models/logistic_regression.py` |

The models use candidate-level features and are intended to follow a common trip-grouped validation and evaluation protocol.

## 5. Feature Engineering

The compact shared feature set includes:

- `distance_m`: Distance between the GPS observation and candidate road segment.
- `distance_gap_m`: Distance difference relative to the nearest candidate.
- `heading_cos`: Cosine similarity between GPS heading and road-segment direction.
- `gps_speed_mps`: GPS-derived movement speed.
- `candidate_rank`: Candidate's proximity ranking.

The feature-generation pipeline is implemented in the candidate-processing modules.

## 6. Validation and Evaluation

The project uses trip-grouped cross-validation to reduce leakage between training and validation trips.

The evaluation framework is intended to report:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Log loss, where applicable
- Observation-level Top-1 accuracy
- Candidate coverage and miss rate
- Route continuity and sequence-level performance
- Training and inference time

The 80 training trips are used for model development and grouped validation. The 10 student-testing trips are reserved for development comparison.

Evaluator-testing data must remain isolated from model selection, tuning and development decisions.

## 7. HMM and Viterbi Route Reconstruction

The intended sequence-inference stage uses an HMM:

- Hidden states represent candidate road segments.
- Emission probabilities are derived from classifier predictions.
- Transition probabilities represent road-network connectivity and movement plausibility.
- Viterbi decoding identifies the most likely sequence of road segments.

The decoder must explicitly handle unreachable transitions, same-segment transitions, U-turns, missing candidates and large timestamp gaps.

HMM/Viterbi implementation and evaluation status should be updated here after verification of the corresponding source code and experiment results.

## 8. Repository Structure

```text
ML_Hackathon/
│
├── models/
│   ├── mlp.py
│   ├── knn.py
│   ├── RF.py
│   ├── logistic_regression.py
│   └── train_xgboost.py
│
├── data_processing/
│   ├── candidate_features.py
│   └── eda.py
│
├── colab_results/
│   ├── features/
│   ├── mlp/
│   ├── knn/
│   └── rf_results/
│
├── results/
│   └── logistic_regression/
│
├── docs/
├── visualizations/
├── requirements.txt
└── README.md
```

Additional experiment directories and HMM artifacts may be added as their implementations are verified.

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/SysRet8-BMS/ML_Hackathon.git
cd ML_Hackathon
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 10. Experiment Reproducibility

Each model should retain:

- Training configuration and random seed.
- Cross-validation fold metrics.
- Student-test evaluation metrics.
- Predictions and model artifacts.
- Feature importance or model-specific diagnostics.
- Training and inference timing.

All published metrics must be generated from actual experiment runs. Dataset files, evaluator labels, credentials and oversized intermediate files must not be committed inadvertently.

## 11. Project Deliverables

The intended final deliverables include:

- Exploratory data analysis and feature engineering.
- Five-model comparative evaluation.
- Model selection and hyperparameter optimisation.
- HMM/Viterbi route reconstruction.
- Route continuity analysis.
- Folium-based route visualisation.
- API and user-interface integration.
- Consolidated technical report.

## 12. Repository

GitHub: https://github.com/SysRet8-BMS/ML_Hackathon

**Project:** PathSnap — Intelligent Road Matching Using Machine Learning.
