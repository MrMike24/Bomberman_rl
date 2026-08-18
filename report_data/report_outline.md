# University Report Outline & File Mapping

**Course**: Machine Learning Essentials Summer Semester 2026  
**Project**: Reinforcement Learning for Bomberman  

This outline provides the structural mapping and empirical evidence references required for writing the 4,000-word final scientific report.

---

## 1. Introduction
- **Discussion**: Problem definition of multi-agent grid games, game rules, discrete action space, survival and tactical bombing challenges.
- **Author Assignment**: [Team Member Name]
- **Supporting Figures & Files**:
  - `README.md` (Game mechanics & specifications)
  - `report_data/hyperparameters.json` (Environment settings)

## 2. Background
- **Discussion**: Theoretical foundations of Reinforcement Learning, MDP formulation, Tabular Q-Learning vs Deep Q-Networks, Bellman Optimality Equation, Experience Replay, and Target Networks.
- **Author Assignment**: [Team Member Name]
- **Supporting Literature & References**:
  - Mnih et al. (2015) *Human-level control through deep reinforcement learning*
  - Watkins & Dayan (1992) *Q-learning*

## 3. Project Planning
- **Discussion**: Team organization, task distribution, hardware allocation (CPU training), curriculum progression strategy (Task 1 to Task 4).
- **Author Assignment**: [Team Member Name]
- **Supporting Documents**:
  - `implementation_plan.md`
  - `report_data/experiment_notes.md`

## 4. Methods
- **Discussion**:
  - **Feature Engineering Engine**: 2D Bomb Danger Map raycasting, BFS pathfinding to nearest coin/crate/opponent, action safety indicators, and `can_escape_bomb` escape path validator.
  - **Agent 1 (Feature-Based Q-Learning)**: Discretized 6-tuple state space ($S \approx 40,000$).
  - **Agent 2 (Deep Q-Network)**: PyTorch 3-Layer MLP ($31 \to 128 \to 64 \to 6$), Smooth L1 Loss, Adam optimizer.
- **Author Assignment**: [Team Member Name]
- **Supporting Files**:
  - `agent_code/project_qlearning/features.py`
  - `agent_code/project_dqn/features.py`
  - `agent_code/project_dqn/model.py`

## 5. Training
- **Discussion**:
  - **Reward Shaping**: Sparse vs Dense Nav vs Combative Survival profiles (`profile_combative_survival`).
  - **Experience Replay Memory**: Capacity $N=50,000$, minibatch size $B=64$.
  - **Target Network**: Hard updates every $C=500$ steps.
  - **Epsilon Schedule**: Decay rate $\gamma_{\epsilon} = 0.998$, $\epsilon_{\min} = 0.05$.
- **Author Assignment**: [Team Member Name]
- **Supporting Files**:
  - `agent_code/project_qlearning/train.py`
  - `agent_code/project_dqn/train.py`
  - `report_data/training_curves/reward_vs_task.png`

## 6. Experiments and Results (Crucial Chapter)
- **Discussion**:
  - **Task 1 (`coin-heaven`)**: Navigation & coin collection (Q: 14.2, DQN: 15.8 coins).
  - **Official Task 2 (`classic`, 0 opponents)**: Crate clearing & bomb survival (Q-Learning score 0.83, 93.3% survival vs DQN score 0.23, 63.3% survival).
  - **Task 3 (`classic` vs passive agents)**: Hunting `peaceful_agent` & `coin_collector_agent` (Q: 2.6, DQN: 2.9).
  - **Task 4 (`classic` vs 3 `rule_based_agent`s)**: Competitive battle (DQN: 2.8 score, 72% survival, 0.5 kills/game vs Q-Learning: 2.4 score, 65% survival).
  - **CPU Action Latency Benchmark**: DQN mean latency 2.72 ms, max 13.50 ms (< 500 ms limit).
- **Author Assignment**: [Team Member Name]
- **Supporting Figures & Tables**:
  - `report_data/experiment_summary.csv`
  - `report_data/official_task2_classic.json`
  - `report_data/large_evaluation_results.csv`
  - `report_data/evaluation_plots/model_comparison_bar.png`
  - `report_data/final_audit.md`

## 7. Conclusion
- **Discussion**: Summary of findings, trade-offs between Tabular Q-learning and DQN across task complexities, future directions (Prioritized Experience Replay, Attention mechanisms).
- **Author Assignment**: [Team Member Name]
