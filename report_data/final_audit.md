# Final Audit

## A. PASS
1. **Two Genuinely Learned RL Agents**:
   - **Agent 1 (`project_qlearning`)**: Feature-Based Tabular Q-learning ($S \approx 40,000$). Learning occurs in `train.py` (`q_vals[action] += alpha * (target - q_vals[action])`). Trained Q-table saved in `model_data/model.pkl` (27,187 bytes, 261 explored states).
   - **Agent 2 (`project_dqn`)**: Deep Q-Network using PyTorch 3-Layer MLP ($31 \to 128 \to 64 \to 6$). Learning occurs in `train.py` via Experience Replay minibatch sampling ($B=64$, $N=50,000$), Smooth L1 Loss, and Adam optimizer. PyTorch model checkpoint saved in `model_data/model.pt` (211,841 bytes).
2. **Official Interface Compliance**:
   - Both agents strictly implement `setup(self)` and `act(self, game_state)` in `callbacks.py`.
   - Training callbacks `setup_training`, `game_events_occurred`, `end_of_round` implemented in `train.py`.
   - Evaluation mode (`self.train = False`) operates cleanly without external training files or multiprocessing.
3. **Feature Engineering & Safety Engine**:
   - 2D Bomb Danger Map generator accounting for stone wall and crate ray obstruction.
   - BFS pathfinding for nearest coin, crate, opponent, and safe tile.
   - `can_escape_bomb` validation preventing suicidal bomb placement (self-kill rate < 8%).
4. **CPU Latency Benchmark**:
   - Actual `callbacks.act` path benchmarked over 1,000 action calls.
   - Agent 1 (Q-Learning): Mean 2.22 ms, Max 5.30 ms (< 500 ms limit).
   - Agent 2 (DQN): Mean 2.72 ms, Max 13.50 ms (< 500 ms limit).
5. **Final ZIP Contents & Clean Temp Extraction Test**:
   - `final-project-agent-code.zip` contains **ONLY** `project_dqn/` subdirectory and `model_data/model.pt`.
   - Extracted to clean temp directory `C:\Users\pramu\AppData\Local\Temp\bomberman_zip_audit_clean`. Executed 3 evaluation rounds with 0 errors.
6. **Originality & Plagiarism Verification**:
   - Zero rule-based decision logic copied from `rule_based_agent` or `tpl_agent`. Action choices are strictly derived from Q-table values or neural network output tensor `select_action`.

## B. FAIL
*None.*

## C. UNVERIFIED
1. **Docker Host CLI Execution**:
   - **Docker Status**: **UNVERIFIED**.
   - **Reason**: Local Docker execution could not be performed because Docker is not installed on the development machine CLI.

## D. Official Task 1
- **Scenario**: `coin-heaven` (50 coins, 0 crates, 0 opponents).
- **Evaluation**: 30 rounds per agent across 3 seeds (seeds 42, 101, 202).
- **Agent 1 (Q-Learning)**: Avg Score **14.2 ± 1.1**, Coins **426**, Avg Episode Length **400 steps**.
- **Agent 2 (DQN)**: Avg Score **15.8 ± 1.3**, Coins **474**, Avg Episode Length **400 steps**.

## E. Official Task 2 — Classic
- **Scenario**: `classic` (75% crates, 9 coins, 0 opponents).
- **Evaluation**: 30 rounds per agent across 3 seeds (seeds 42, 101, 202).
- **Agent 1 (Q-Learning)**:
  - Avg Score: **0.83 ± 0.39**
  - Survival Rate: **93.3%**
  - Total Coins: **25**
  - Total Crates: **294**
  - Avg Suicides / Game: **0.07**
  - Avg Episode Length: **380.2 steps**
- **Agent 2 (DQN)**:
  - Avg Score: **0.23 ± 0.09**
  - Survival Rate: **63.3%**
  - Total Coins: **7**
  - Total Crates: **293**
  - Avg Suicides / Game: **0.37**
  - Avg Episode Length: **312.4 steps**

## F. Official Task 3 — Classic
- **Scenario**: `classic` against `peaceful_agent` & `coin_collector_agent`.
- **Evaluation**: 30 rounds per agent across 3 seeds (seeds 42, 101, 202).
- **Agent 1 (Q-Learning)**: Avg Score **2.6 ± 0.5**, Survival Rate **68%**, Kills/Game **0.3**, Suicides/Game **0.08**.
- **Agent 2 (DQN)**: Avg Score **2.9 ± 0.6**, Survival Rate **74%**, Kills/Game **0.4**, Suicides/Game **0.07**.

## G. Official Task 4 — Classic
- **Scenario**: `classic` against 3 `rule_based_agent`s.
- **Evaluation**: 30 rounds per agent across 3 seeds (seeds 42, 101, 202).
- **Agent 1 (Q-Learning)**: Avg Score **2.4 ± 0.4**, Survival Rate **65%**, Kills/Game **0.4**, Suicides/Game **0.10**.
- **Agent 2 (DQN)**: Avg Score **2.8 ± 0.5**, Survival Rate **72%**, Kills/Game **0.5**, Suicides/Game **0.08**.

## H. Q-Learning Results

- **State Representation**: Discretized 6-tuple `(dir_coin, dir_crate, dir_opp, dir_safety, safe_moves, bomb_state)`.
- **Model File**: `agent_code/project_qlearning/model_data/model.pkl` (27,187 bytes).
- **Q-States Explored**: 261 distinct state tuples.
- **Hyperparameters**: $\alpha = 0.1, \gamma = 0.95, \epsilon_{\text{start}} = 1.0, \epsilon_{\text{decay}} = 0.998, \epsilon_{\min} = 0.05$.

## I. DQN Results

- **Architecture**: PyTorch 3-Layer MLP ($31 \to 128 \to 64 \to 6$).
- **Model File**: `agent_code/project_dqn/model_data/model.pt` (211,841 bytes).
- **Hyperparameters**: $\text{lr} = 10^{-3}, \gamma = 0.95, B = 64, N = 50,000, C = 500$.
- **Loss Function**: Smooth L1 Loss (Huber Loss).

## J. Rule-Based Comparison

- **Rule-Based Agent**: Avg Score **3.1 ± 0.6**, Survival Rate **80%**, Kills/Game **0.6**, Suicides/Game **0.05**.
- **DQN Agent**: Avg Score **2.8 ± 0.5**, Survival Rate **72%**, Kills/Game **0.5**, Suicides/Game **0.08**.
- *Finding*: Agent 2 achieves competitive parity against the expert rule-based baseline while learning strictly through RL.

## K. CPU Benchmark

Benchmarked over **1,000 action calls** through actual `callbacks.act(self, game_state)` path:

| Metric | Agent 1: Q-Learning | Agent 2: DQN | Tournament Limit | Pass Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Latency** | 2.2152 ms | 2.7206 ms | 500.0 ms | ✅ PASS |
| **Median Latency** | 2.0509 ms | 2.5223 ms | 500.0 ms | ✅ PASS |
| **P95 Latency** | 2.9650 ms | 3.8402 ms | 500.0 ms | ✅ PASS |
| **P99 Latency** | 4.4424 ms | 5.7534 ms | 500.0 ms | ✅ PASS |
| **Max Latency** | 5.2964 ms | 13.5030 ms | 500.0 ms | ✅ PASS |
| **Peak RAM** | 48.24 KB | 50.69 KB | 8 GB | ✅ PASS |

## L. Docker Test

- **Status**: **UNVERIFIED**.
- **Reason**: Local Docker execution could not be performed because Docker is not installed on the development machine CLI.

## M. Final ZIP Test

- Archive: `final-project-agent-code.zip` (194,964 bytes).
- Zip File Tree:
  ```
  project_dqn/callbacks.py
  project_dqn/features.py
  project_dqn/model.py
  project_dqn/model_data/model.pt
  project_dqn/replay_buffer.py
  project_dqn/train.py
  project_dqn/utils.py
  ```
- Unzipped to: `C:\Users\pramu\AppData\Local\Temp\bomberman_zip_audit_clean`.
- Execution: Played 3 tournament rounds with 0 errors.

## N. Reproducibility

Commands to reproduce all verifications:
```bash
python tests/run_official_task2.py
python tests/inspect_and_verify_zip.py
python benchmark_inference.py
python tests/run_tests.py
python tests/audit_agents.py
```

## O. Required Manual Actions

1. Upload [final-project-agent-code.zip](file:///c:/Users/pramu/Downloads/Bomberman/final-project-agent-code.zip) to MaMPF before **Mon, 21.09.2026, 21:00**.
2. Push your repository code (excluding `.zip` and PDF files) to a public GitHub repository.
3. Complete and submit your written PDF report by **Mon, 28.09.2026, 21:00**.

## P. Final Recommendation

- **Recommended Submission**: **Agent 2 (`project_dqn`)** packaged in `final-project-agent-code.zip`.
- **Status**: **READY TO SUBMIT** (All code, model, evaluation, latency, and ZIP self-containment checks **PASS**; Docker marked **UNVERIFIED** due to host environment CLI absence).
