import csv
import json
import os
import sys
import numpy as np

# Ensure root is in python path
sys.path.insert(0, os.path.abspath('.'))

import settings as s
from evaluate_agent import evaluate_agent


def run_official_task2_evaluations():
    out_dir = os.path.join('report_data')
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 101, 202]
    rounds_per_seed = 10  # 30 rounds total per agent
    scenario = "classic"  # Official Task 2 scenario (75% crates, 9 coins, 0 opponents)
    opponents = []

    task2_results = []

    print("=================================================================")
    print("RUNNING OFFICIAL TASK 2 EVALUATION — CLASSIC (0 Opponents)")
    print("=================================================================\n")

    for agent_name in ["project_qlearning", "project_dqn"]:
        seed_scores = []
        seed_survivals = []
        seed_kills = []
        seed_suicides = []
        seed_coins = []
        seed_crates = []
        seed_steps = []

        for seed in seeds:
            res = evaluate_agent(agent_name, opponents=opponents, scenario=scenario, n_rounds=rounds_per_seed, seed=seed)
            seed_scores.append(res['avg_score'])
            seed_survivals.append(res['survival_rate'])
            seed_kills.append(res['avg_kills_per_game'])
            seed_suicides.append(res['avg_suicides_per_game'])
            seed_coins.append(res['total_coins'])
            seed_crates.append(res['total_crates'])
            seed_steps.append(res['avg_steps'])

        metrics = {
            "experiment": "OFFICIAL TASK 2 — CLASSIC",
            "scenario": scenario,
            "agent": agent_name,
            "avg_score": float(np.mean(seed_scores)),
            "std_score": float(np.std(seed_scores)),
            "survival_rate": float(np.mean(seed_survivals)),
            "avg_kills_per_game": float(np.mean(seed_kills)),
            "avg_suicides_per_game": float(np.mean(seed_suicides)),
            "total_coins": int(np.sum(seed_coins)),
            "total_crates": int(np.sum(seed_crates)),
            "avg_episode_length": float(np.mean(seed_steps))
        }

        task2_results.append(metrics)
        print(f"[OFFICIAL TASK 2 — CLASSIC] {agent_name:22s} -> Score: {metrics['avg_score']:.2f} ± {metrics['std_score']:.2f} | Survival: {metrics['survival_rate']*100:.1f}% | Coins: {metrics['total_coins']} | Crates: {metrics['total_crates']} | Suicides/Game: {metrics['avg_suicides_per_game']:.2f}")

    # Save to JSON
    json_path = os.path.join(out_dir, 'official_task2_classic.json')
    with open(json_path, 'w') as f:
        json.dump(task2_results, f, indent=2)

    # Append/Write to CSV
    csv_file = os.path.join(out_dir, 'experiment_summary.csv')
    if task2_results:
        keys = list(task2_results[0].keys())
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerows(task2_results)

    print(f"\nSaved Official Task 2 Classic results to {json_path} and {csv_file}")
    return task2_results


if __name__ == '__main__':
    run_official_task2_evaluations()
