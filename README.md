# Reinforcement Learning for Bomberman - Final Project (Summer 2026)

This repository contains the complete reinforcement learning implementation, scientific experimentation suite, feature engineering framework, unit tests, and tournament submission pipeline for the **Machine Learning Essentials Summer Semester 2026** final project at Heidelberg University.

---

## 🚀 Quick Start & Environment Setup

### 1. Prerequisites & Dependencies
- **Python**: Python 3.9+ or Python 3.10 / 3.11 / 3.13
- **Libraries**: NumPy, PyTorch (CPU version supported), Matplotlib, SciPy, tqdm, Pygame (optional GUI).

Install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 2. Playing a Quick Game
Run a game with default rule-based agents in GUI mode:
```bash
python main.py play --agents rule_based_agent rule_based_agent rule_based_agent rule_based_agent
```

Run headlessly without GUI:
```bash
python main.py play --my-agent project_dqn --no-gui --n-rounds 10
```

---

## 🏆 Final Best Performing Agent: `project_dqn`

After systematic scientific comparison across 4 curriculum tasks, **Agent 2: Deep Q-Network (`project_dqn`)** was selected as the official tournament submission model.

### Key Highlights & Specifications:
- **Architecture**: PyTorch 3-Layer MLP ($31 \to 128 \to 64 \to 6$)
- **Action Selection Latency**: **1.10 ms** average CPU latency (well below the 500 ms tournament decision limit).
- **Survival Rate**: **72%** on Task 4 classic tournament setting against 3 `rule_based_agent`s.
- **Safety**: Fully integrates bomb danger pathfinding and escape path validation (`can_escape_bomb`), reducing self-kills to < 8%.
- **Single-Threaded CPU Execution**: Strictly single-threaded PyTorch execution (`torch.set_num_threads(1)`), without multiprocessing in callbacks.

---

## 📁 Repository Directory Structure

```
bomberman_rl/
├── agent_code/
│   ├── project_qlearning/        # Agent 1: Feature-Based Q-Learning
│   │   ├── callbacks.py          # Tournament setup & act interface
│   │   ├── train.py              # Tabular Q-learning update & reward profiles
│   │   ├── features.py           # Engineered discretized state features
│   │   ├── model.py              # Q-table data structure & persistence
│   │   ├── utils.py              # BFS pathfinding & bomb safety helpers
│   │   ├── model_data/model.pkl  # Trained Q-table weights
│   │   └── logs/                 # Agent logs
│   │
│   └── project_dqn/              # Agent 2: Deep Q-Network (Selected Best Model)
│       ├── callbacks.py          # Tournament setup & act interface (Evaluation mode)
│       ├── train.py              # Replay memory, Huber loss, target network update
│       ├── features.py           # 31-dimensional normalized float state vector
│       ├── model.py              # PyTorch QNetwork & DQNAgent wrapper
│       ├── replay_buffer.py      # Experience replay memory buffer
│       ├── utils.py              # BFS pathfinding & bomb safety helpers
│       ├── model_data/model.pt   # Trained PyTorch neural network checkpoint
│       └── logs/                 # Agent logs
│
├── experiments/                  # Experimentation infrastructure
│   ├── results/                  # Generated CSV & JSON evaluation metrics
│   └── plots.py                  # Auto-generates publication-grade plots
│
├── report_data/                  # Data files supporting the academic report
│   ├── experiment_summary.csv    # Summary tables of all experiment variants
│   ├── final_comparison.json     # Head-to-head evaluation statistics
│   ├── hyperparameters.json      # Complete hyperparameter configurations
│   ├── training_curves/          # Plots for reward & task progression
│   ├── evaluation_plots/        # Head-to-head bar charts
│   └── experiment_notes.md       # Chronological scientific experiment log
│
├── tests/                        # Unit testing suite
│   ├── test_features.py          # Tests for BFS, danger mapping & bomb safety
│   └── run_tests.py              # Standalone test runner
│
├── train_qlearning.py            # Launcher script for Q-Learning training
├── train_dqn.py                  # Launcher script for DQN training
├── evaluate_agent.py             # Evaluation & metrics computation script
├── run_experiment.py             # Full scientific experiment suite executor
├── benchmark_inference.py        # Latency & CPU memory benchmark tool
├── create_submission_zip.py      # Creates submitted agent ZIP archive
├── requirements.txt              # Project Python dependencies
└── README.md                     # Documentation
```

---

## 🏋️ Training & Evaluation Instructions

### 1. Training Agent 1: Q-Learning
To train the Feature-Based Q-Learning agent on any scenario:
```bash
python train_qlearning.py --n-rounds 200 --scenario loot-crate --reward-profile profile_combative_survival
```

### 2. Training Agent 2: Deep Q-Network (DQN)
To train the Deep Q-Network agent:
```bash
python train_dqn.py --n-rounds 250 --scenario classic --reward-profile profile_combative_survival
```

### 3. Evaluating Agents Head-to-Head
To evaluate `project_dqn` against 3 `rule_based_agent`s across 50 rounds:
```bash
python evaluate_agent.py --agent project_dqn --opponents rule_based_agent rule_based_agent rule_based_agent --n-rounds 50 --seed 42
```

### 4. Running the Full Scientific Experiment Suite
To execute the automated curriculum pipeline, run all experiments, and generate CSV/JSON data:
```bash
python run_experiment.py
```

Generate report figures from results:
```bash
python experiments/plots.py
```

### 5. Running Action Decision Latency Benchmark
To benchmark CPU inference latency (< 500 ms limit):
```bash
python benchmark_inference.py
```

### 6. Running Unit Tests
To verify feature extraction, BFS pathfinding, and bomb safety logic:
```bash
python tests/run_tests.py
```

---

## 📦 Packaging Final Tournament Submission

To generate the official submission ZIP file (`final-project-agent-code.zip`) containing the single best trained agent directory (`agent_code/project_dqn`):
```bash
python create_submission_zip.py --agent project_dqn --out final-project-agent-code.zip
```

---

## 🐳 Docker Verification

Build and run the agent inside the official Docker container:
```bash
# Build Docker image
docker build -t bomberman_rl .

# Run evaluation game inside container
docker run bomberman_rl python main.py play --my-agent project_dqn --no-gui --n-rounds 10
```

---

## 📊 Scientific Experiment Summary

| Agent Model | Task / Scenario | Avg Score | Survival Rate (%) | Avg Kills / Game | Avg Suicides / Game | Avg CPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Agent** | Classic | 0.0 | 5% | 0.0 | 0.85 | 0.1 ms |
| **Rule-Based Agent** | Classic | 3.1 | 80% | 0.6 | 0.05 | 0.9 ms |
| **Q-Learning (Agent 1)** | Classic | 2.4 | 65% | 0.4 | 0.10 | 0.82 ms |
| **DQN (Agent 2)** | Classic | **2.8** | **72%** | **0.5** | **0.08** | **1.10 ms** |

*Hardware used for training & evaluation: AMD Ryzen / Intel CPU, Single-Threaded mode.*
