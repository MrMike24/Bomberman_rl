# Model v2 Scientific Experiment Notes & Execution Log

**Project**: Machine Learning Essentials — Bomberman RL Final Project  
**Log Timestamp**: September 2026  

---

## Log Entry 1: Baseline Preservation & Codebase Audit
- Created complete directory backups in `backup_baseline/` covering `project_qlearning`, `project_dqn`, existing weights, `report_data/`, and `experiments/`.
- Verified that baseline models and training pipelines remain intact and recoverable.
- Validated current test suite: 5/5 unit tests passed.

---

## Log Entry 2: Feature Engineering & Architecture Design
- **Q-Learning v2 (`project_qlearning_v2`)**:
  - Implemented 8-element discrete state tuple with true BFS distance discretization (`dist_target_bucket`), dead-end corridor detection (`in_dead_end`), prioritized targets (`dir_primary_target`), and bomb feasibility.
  - Resolved reward shaping distance bug by implementing true BFS path length differentials rather than directional action code comparisons.
- **Double DQN v2 (`project_dqn_v2`)**:
  - Engineered 38-dimensional float32 normalized spatial feature representation.
  - Implemented Double DQN target decoupled loss formulation: $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta}(s', a'))$.
  - Implemented Polyak soft target updates ($\tau = 0.005$) replacing unstable 500-step periodic hard updates.
  - Integrated Layer Normalization (`nn.LayerNorm`), Huber Loss (`SmoothL1Loss`), and AdamW optimizer.

---

## Log Entry 3: Comprehensive Unit Testing Suite
- Extended `tests/test_features.py` and `tests/run_tests.py` with 9 unit tests:
  1. `test_bomb_danger_map` -> PASS
  2. `test_bfs_pathfinding` -> PASS
  3. `test_can_escape_bomb` -> PASS
  4. `test_qlearning_feature_extraction` -> PASS
  5. `test_dqn_feature_extraction` -> PASS
  6. `test_qlearning_v2_feature_extraction` -> PASS
  7. `test_dqn_v2_feature_extraction` -> PASS
  8. `test_dqn_v2_double_dqn_step` -> PASS
  9. `test_callbacks_act_all_models` -> PASS
- Result: 9/9 unit tests passed cleanly.

---

## Log Entry 4: Action-Decision CPU Latency Benchmarking
- Benchmarked 1,000 actual `callbacks.act()` invocations on realistic game states:
  - `project_dqn_v2`: Mean **2.47 ms** | Median **2.38 ms** | P95 **2.87 ms** | P99 **3.85 ms** | Max **5.48 ms** (Limit 500 ms: **PASS**).
  - `project_qlearning_v2`: Mean **1.82 ms** | Median **1.75 ms** | P95 **2.09 ms** | P99 **2.99 ms** | Max **4.24 ms** (Limit 500 ms: **PASS**).

---

## Log Entry 5: Model Training Execution
- **`project_qlearning_v2`**:
  - Stage 1: `loot-crate` (100 rounds) -> completed.
  - Stage 2: `classic` arena against opponents (150 rounds) -> completed.
  - Saved model to `agent_code/project_qlearning_v2/model_data/model.pkl`.
- **`project_dqn_v2`**:
  - Stage 1: `loot-crate` (100 rounds) -> completed.
  - Stage 2: `classic` arena against opponents (150 rounds) -> completed.
  - Saved model to `agent_code/project_dqn_v2/model_data/model.pt`.

---

## Log Entry 6: Multi-Spawn Starting Position Evaluation
- Evaluated `project_dqn_v2` across all 4 arena spawn coordinates $(1,1), (1,15), (15,1), (15,15)$ and uniform random permutation (25 rounds each, 125 rounds total):
  - Position 1 $(1, 1)$: Avg Score 1.00, Survival 12.0%, Kills 0.08
  - Position 2 $(1, 15)$: Avg Score 0.80, Survival 12.0%, Kills 0.04
  - Position 3 $(15, 1)$: Avg Score 1.08, Survival 8.0%, Kills 0.08
  - Position 4 $(15, 15)$: Avg Score 0.28, Survival 0.0%, Kills 0.00
  - Random Position: Avg Score 0.96, Survival 12.0%, Kills 0.08
- Results saved to `report_data/random_position_results.csv` and `report_data/random_position_results.json`.
- Plot generated at `report_data/evaluation_plots/starting_position_performance.png`.

---

## Log Entry 7: Fair Head-to-Head Model Comparison
- Evaluated 4 models on Classic scenario against 3 rule-based opponents across 50 rounds (seeds 42 and 101):
  - **Baseline Q-Learning**: Avg Score 1.82, Survival 54.0%, Kills 0.12, Suicides 0.36
  - **Q-Learning v2**: Avg Score 0.56, Survival 20.0%, Kills 0.06, Suicides 0.48
  - **Baseline DQN**: Avg Score 0.70, Survival 2.0%, Kills 0.04, Suicides 0.66
  - **Double DQN v2**: Avg Score 0.98, Survival 20.0%, Kills 0.06, Suicides 0.42
- Results saved to `report_data/model_v2_comparison.csv` and `report_data/model_v2_comparison.json`.
- Comparison plot generated at `report_data/evaluation_plots/model_v2_comparison_chart.png`.

---

## Log Entry 8: Google Colab GPU Pipeline
- Prepared `colab_train_dqn_v2.ipynb` containing dependency installation, CUDA detection, GPU tensor placement, Double DQN training curriculum, evaluation, and checkpoint download.
- Local training was executed using single-threaded CPU mode; the notebook is fully ready for extended GPU training in Colab.
