# HMM-Based Road Matching

## Overview
This module uses a Hidden Markov Model (HMM) and the Viterbi algorithm to identify the most likely sequence of road segments for a GPS trajectory.

## Approach
- **States:** Candidate road segments for each GPS observation.
- **Emissions:** Likelihood of a GPS observation matching a candidate segment.
- **Transitions:** Likelihood of moving between consecutive road segments.
- **Viterbi:** Finds the most probable sequence of road segments.

## Workflow
1. Load GPS observations and candidate road segments.
2. Calculate emission and transition probabilities.
3. Apply Viterbi decoding.
4. Generate the most likely road-segment sequence.
5. Evaluate the predicted route.

## Output
The predicted sequence of road segments representing the most likely route for each GPS trajectory.

## Purpose
To improve road-matching accuracy by considering the continuity of a complete GPS trajectory rather than matching each GPS point independently.
