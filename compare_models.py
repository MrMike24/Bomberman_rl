import argparse
import csv
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from evaluate_agent import evaluate_agent


def run_fair_comparison(n_rounds=30, seeds=[42, 101]):
    models = [
        ("project_qlearning", "Baseline Q-Learning (Model A)"),
        ("project_qlearning_v2", "Q-Learning v2 (Model A v2)"),
        ("project_dqn", "Baseline DQN (Model B)"),
        ("project_dqn_v2", "Double DQN v2 (Model B v2)")
    ]

    opponents = ["rule_based_agent", "rule_based_agent", "rule_based_agent"]
    scenario = "classic"

    all_results = []

    print("\n=======================================================")
    print("STARTING FAIR SCIENTIFIC HEAD-TO-HEAD MODEL COMPARISON")
    print(f"Scenario: {scenario} | Rounds per seed: {n_rounds} | Seeds: {seeds}")
    print("=======================================================\n")

    for agent_folder, display_name in models:
        scores_combined = []
        survived_combined = 0
        kills_combined = 0
        suicides_combined = 0
        coins_combined = 0
        crates_combined = 0
        steps_combined = []
        total_rounds = 0

        for seed in seeds:
            print(f"Evaluating {display_name} (Seed {seed})...")
            res = evaluate_agent(
                agent_name=agent_folder,
                opponents=opponents,
                scenario=scenario,
                n_rounds=n_rounds,
                seed=seed
            )
            total_rounds += n_rounds
            survived_combined += int(res['survival_rate'] * n_rounds)
            kills_combined += res['total_kills']
            suicides_combined += res['total_suicides']
            coins_combined += res['total_coins']
            crates_combined += res['total_crates']
            scores_combined.append(res['avg_score'])
            steps_combined.append(res['avg_steps'])

        avg_score = float(np.mean(scores_combined))
        std_score = float(np.std(scores_combined))
        survival_rate = float(survived_combined / total_rounds)
        avg_kills = float(kills_combined / total_rounds)
        avg_suicides = float(suicides_combined / total_rounds)
        avg_coins = float(coins_combined / total_rounds)
        avg_crates = float(crates_combined / total_rounds)
        avg_steps = float(np.mean(steps_combined))

        model_summary = {
            "agent_folder": agent_folder,
            "model_name": display_name,
            "scenario": scenario,
            "total_rounds": total_rounds,
            "seeds": seeds,
            "avg_score": avg_score,
            "std_score": std_score,
            "survival_rate": survival_rate,
            "avg_kills_per_game": avg_kills,
            "avg_suicides_per_game": avg_suicides,
            "avg_coins_per_game": avg_coins,
            "avg_crates_per_game": avg_crates,
            "avg_episode_length": avg_steps
        }
        all_results.append(model_summary)
        print(f"==> {display_name}: Score={avg_score:.2f} | Survival={survival_rate*100:.1f}% | Kills={avg_kills:.2f} | Suicides={avg_suicides:.2f}\n")

    # Save to CSV
    csv_path = os.path.join("report_data", "model_v2_comparison.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    # Save to JSON
    json_path = os.path.join("report_data", "model_v2_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved comparison metrics to {csv_path} and {json_path}")

    # Generate comparison plot
    plot_dir = os.path.join("report_data", "evaluation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "model_v2_comparison_chart.png")

    names = [r['model_name'] for r in all_results]
    scores = [r['avg_score'] for r in all_results]
    survivals = [r['survival_rate'] * 100 for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart 1: Average Score
    colors1 = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']
    bars1 = ax1.bar(names, scores, color=colors1, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Average Score / Game', fontsize=11, fontweight='bold')
    ax1.set_title('Average Tournament Score Across Models', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    # Bar chart 2: Survival Rate
    bars2 = ax2.bar(names, survivals, color=colors1, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Survival Rate Against 3 Rule-Based Agents', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax2.set_ylim(0, 100)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated comparison chart at {plot_path}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fair Comparison of RL Models")
    parser.add_argument("--n-rounds", type=int, default=25, help="Rounds per seed")
    args = parser.parse_args()

    run_fair_comparison(n_rounds=args.n_rounds, seeds=[42, 101])
