import csv
import json
import os
import sys
import numpy as np

# Ensure root is in python path
sys.path.insert(0, os.path.abspath('.'))

import settings as s
from evaluate_agent import evaluate_agent


def run_comprehensive_evaluations():
    out_dir = os.path.join('report_data')
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    seeds = [42, 101, 202]
    rounds_per_seed = 10  # 10 * 3 seeds = 30 total rounds per experiment

    print("=================================================================")
    print("STARTING LARGE-SCALE MULTI-SEED FINAL EVALUATION SUITE (~100 GAMES PER SPEC)")
    print("=================================================================\n")

    experiments = [
        # TASK 1: Navigation on coin-heaven (no crates, no opponents)
        ("Task1_Navigation", "coin-heaven", [], ["project_qlearning", "project_dqn"]),

        # TASK 2: Crates & Bomb Survival on loot-crate (75% crates, 0 opponents)
        ("Task2_Crates_Survival", "loot-crate", [], ["project_qlearning", "project_dqn"]),

        # TASK 3: Hunting Passive Opponents on classic (crates, peaceful + coin_collector)
        ("Task3_Passive_Opponents", "classic", ["peaceful_agent", "coin_collector_agent"], ["project_qlearning", "project_dqn"]),

        # TASK 4: Tournament Competitive Battle on classic vs 3 rule_based_agents
        ("Task4_Tournament_Battle", "classic", ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
         ["project_qlearning", "project_dqn", "rule_based_agent", "random_agent"])
    ]

    for exp_name, scenario, opps, agent_list in experiments:
        print(f"\n>>> Running Experiment Suite: {exp_name} (Scenario: {scenario}) <<<")
        for agent_name in agent_list:
            seed_scores = []
            seed_survivals = []
            seed_kills = []
            seed_suicides = []
            seed_coins = []
            seed_crates = []
            seed_steps = []
            seed_invalids = []

            for seed in seeds:
                res = evaluate_agent(agent_name, opponents=opps, scenario=scenario, n_rounds=rounds_per_seed, seed=seed)
                seed_scores.append(res['avg_score'])
                seed_survivals.append(res['survival_rate'])
                seed_kills.append(res['avg_kills_per_game'])
                seed_suicides.append(res['avg_suicides_per_game'])
                seed_coins.append(res['total_coins'])
                seed_crates.append(res['total_crates'])
                seed_steps.append(res['avg_steps'])
                seed_invalids.append(res['total_invalid'])

            combined_metrics = {
                "experiment": exp_name,
                "scenario": scenario,
                "agent": agent_name,
                "total_games": rounds_per_seed * len(seeds),
                "seeds": seeds,
                "avg_score": float(np.mean(seed_scores)),
                "std_score": float(np.std(seed_scores)),
                "survival_rate": float(np.mean(seed_survivals)),
                "avg_kills_per_game": float(np.mean(seed_kills)),
                "avg_suicides_per_game": float(np.mean(seed_suicides)),
                "total_coins": int(np.sum(seed_coins)),
                "total_crates": int(np.sum(seed_crates)),
                "avg_episode_length": float(np.mean(seed_steps)),
                "total_invalid_actions": int(np.sum(seed_invalids))
            }

            all_results.append(combined_metrics)
            print(f"[{exp_name}] {agent_name:22s} -> Score: {combined_metrics['avg_score']:.2f} ± {combined_metrics['std_score']:.2f} | Survival: {combined_metrics['survival_rate']*100:.1f}% | Kills: {combined_metrics['avg_kills_per_game']:.2f} | Suicides: {combined_metrics['avg_suicides_per_game']:.2f}")

    # Save to JSON
    json_path = os.path.join(out_dir, 'large_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save to CSV
    csv_path = os.path.join(out_dir, 'large_evaluation_results.csv')
    if all_results:
        keys = list(all_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    print("\n=================================================================")
    print(f"LARGE-SCALE EVALUATIONS COMPLETE. Saved to {csv_path} and {json_path}")
    print("=================================================================\n")


if __name__ == '__main__':
    run_comprehensive_evaluations()
