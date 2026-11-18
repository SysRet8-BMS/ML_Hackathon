# HMM and Viterbi

Source code for route reconstruction and sequence decoding.

## Status: not yet implemented

The current API decodes each GPS observation independently and reports
`decoder: "classifier_only"`. No HMM transition model or Viterbi decoding exists
in this repository yet.

## What belongs here

1. **Emissions** — per-observation candidate probabilities from the trained
   classifier (already available from the model pipeline).
2. **Transition model** — probability of moving from one route arc to another,
   based on road-network connectivity and distance.
3. **Viterbi decoder** — sequence-level most-likely route instead of
   per-point argmax.
4. **Ablation** — classifier-only vs classifier + Viterbi, using identical
   trip-grouped folds.
5. **Sequence-level metrics** — route continuity, connected-component validity,
   and end-to-end route accuracy.

Wire the decoder into `api/service.py` and change the reported `decoder` field
once it produces predictions.
