import argparse
import csv
import json
import os
import sys
import numpy as np

from train_qlearning import train_qlearning
from train_dqn import train_dqn
from evaluate_agent import evaluate_agent


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'results')
REPORT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'report_data')


def run_full_experiment_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORT_DATA_DIR, exist_ok=True)

    summary_rows = []

    print("\n=======================================================")
    print("STARTING SCIENTIFIC EXPERIMENT SUITE & TASK CURRICULUM")
    print("=======================================================\n")

    # ---------------------------------------------------------
    # STAGE 1: TASK 1 CURRICULUM (Navigation / Coin Heaven)
    # ---------------------------------------------------------
    print("\n>>> STAGE 1: TASK 1 CURRICULUM (Navigation) <<<")
    train_qlearning(n_rounds=150, scenario="coin-heaven", opponents=[], reward_profile="profile_dense_nav", seed=42)
    train_dqn(n_rounds=150, scenario="coin-heaven", opponents=[], reward_profile="profile_dense_nav", seed=42)

    eval_q_t1 = evaluate_agent("project_qlearning", opponents=["random_agent"], scenario="coin-heaven", n_rounds=20, seed=101)
    eval_dqn_t1 = evaluate_agent("project_dqn", opponents=["random_agent"], scenario="coin-heaven", n_rounds=20, seed=101)

    summary_rows.append({"experiment": "Task1_Navigation", "agent": "Q-Learning", **eval_q_t1})
    summary_rows.append({"experiment": "Task1_Navigation", "agent": "DQN", **eval_dqn_t1})

    # ---------------------------------------------------------
    # STAGE 2: TASK 2 CURRICULUM (Loot Crate & Bomb Survival)
    # ---------------------------------------------------------
    print("\n>>> STAGE 2: TASK 2 CURRICULUM (Crates & Bomb Survival) <<<")
    train_qlearning(n_rounds=250, scenario="loot-crate", opponents=[], reward_profile="profile_combative_survival", seed=42)
    train_dqn(n_rounds=250, scenario="loot-crate", opponents=[], reward_profile="profile_combative_survival", seed=42)

    eval_q_t2 = evaluate_agent("project_qlearning", opponents=["random_agent"], scenario="loot-crate", n_rounds=20, seed=202)
    eval_dqn_t2 = evaluate_agent("project_dqn", opponents=["random_agent"], scenario="loot-crate", n_rounds=20, seed=202)

    summary_rows.append({"experiment": "Task2_Crates", "agent": "Q-Learning", **eval_q_t2})
    summary_rows.append({"experiment": "Task2_Crates", "agent": "DQN", **eval_dqn_t2})

    # ---------------------------------------------------------
    # STAGE 3: TASK 3 & 4 CURRICULUM (Opponents & Rule-Based Agent)
    # ---------------------------------------------------------
    print("\n>>> STAGE 3: TASK 3 & 4 CURRICULUM (Competitive Combat) <<<")
    train_qlearning(n_rounds=300, scenario="classic", opponents=["peaceful_agent", "coin_collector_agent", "rule_based_agent"], reward_profile="profile_combative_survival", seed=42)
    train_dqn(n_rounds=300, scenario="classic", opponents=["peaceful_agent", "coin_collector_agent", "rule_based_agent"], reward_profile="profile_combative_survival", seed=42)

    eval_q_t4 = evaluate_agent("project_qlearning", opponents=["rule_based_agent", "rule_based_agent", "rule_based_agent"], scenario="classic", n_rounds=30, seed=303)
    eval_dqn_t4 = evaluate_agent("project_dqn", opponents=["rule_based_agent", "rule_based_agent", "rule_based_agent"], scenario="classic", n_rounds=30, seed=303)
    eval_rb_t4 = evaluate_agent("rule_based_agent", opponents=["rule_based_agent", "rule_based_agent", "rule_based_agent"], scenario="classic", n_rounds=30, seed=303)

    summary_rows.append({"experiment": "Task4_Tournament", "agent": "Q-Learning", **eval_q_t4})
    summary_rows.append({"experiment": "Task4_Tournament", "agent": "DQN", **eval_dqn_t4})
    summary_rows.append({"experiment": "Task4_Tournament", "agent": "RuleBasedBaseline", **eval_rb_t4})

    # ---------------------------------------------------------
    # EXPERIMENT 4: REWARD SHAPING COMPARISON
    # ---------------------------------------------------------
    print("\n>>> EXPERIMENT 4: REWARD SHAPING COMPARISON <<<")
    for profile in ['profile_sparse', 'profile_dense_nav', 'profile_combative_survival']:
        train_qlearning(n_rounds=100, scenario="loot-crate", opponents=[], reward_profile=profile, seed=42)
        res = evaluate_agent("project_qlearning", opponents=["random_agent"], scenario="loot-crate", n_rounds=15, seed=404)
        summary_rows.append({"experiment": f"RewardShaping_{profile}", "agent": "Q-Learning", **res})

    # Save CSV summaries
    csv_file = os.path.join(REPORT_DATA_DIR, 'experiment_summary.csv')
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(csv_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_rows)

    # Save JSON summary
    json_file = os.path.join(REPORT_DATA_DIR, 'final_comparison.json')
    with open(json_file, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved complete experiment metrics to {csv_file} and {json_file}")
    print("=======================================================\n")


if __name__ == '__main__':
    run_full_experiment_suite()
