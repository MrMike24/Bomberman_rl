import argparse
import csv
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

import settings as s
from environment import BombeRLeWorld, WorldArgs
from items import Coin, Explosion, Bomb


class PositionControlledWorld(BombeRLeWorld):
    """
    Subclass of BombeRLeWorld for scientific robustness evaluation from controlled spawn positions.
    Does not modify or break the official environment.
    """
    def __init__(self, args, agents, fixed_spawn_index=None):
        super().__init__(args, agents)
        self.fixed_spawn_index = fixed_spawn_index  # 0, 1, 2, 3 or None for random

    def build_arena(self):
        arena, coins, active_agents = super().build_arena()

        # The 4 official spawn coordinates
        official_spawns = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]

        if self.fixed_spawn_index is not None and 0 <= self.fixed_spawn_index < 4:
            target_spawn = official_spawns[self.fixed_spawn_index]
            remaining_spawns = [pos for pos in official_spawns if pos != target_spawn]
            self.rng.shuffle(remaining_spawns)

            # Assign target agent (index 0) to fixed spawn
            if len(self.agents) > 0:
                self.agents[0].x, self.agents[0].y = target_spawn

            # Assign remaining agents to other spawns
            for agent, spawn in zip(self.agents[1:], remaining_spawns):
                agent.x, agent.y = spawn

        return arena, coins, active_agents


class DummyArgs:
    def __init__(self, agent_name, opponents, scenario, n_rounds, seed):
        self.command_name = 'play'
        self.my_agent = None
        self.agents = [agent_name] + list(opponents)[:s.MAX_AGENTS - 1]
        self.train = 0
        self.continue_without_training = True
        self.scenario = scenario
        self.seed = seed
        self.n_rounds = n_rounds
        self.save_replay = False
        self.match_name = None
        self.silence_errors = True
        self.skip_frames = True
        self.no_gui = True
        self.turn_based = False
        self.update_interval = 0.0
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        self.save_stats = False
        self.make_video = False


def evaluate_spawn_position(agent_name, opponents, scenario="classic", n_rounds=25, seed=42, spawn_idx=None):
    args = DummyArgs(agent_name, opponents, scenario, n_rounds, seed)
    agents_spec = [(name, False) for name in args.agents]

    world = PositionControlledWorld(args, agents_spec, fixed_spawn_index=spawn_idx)
    target_agent = world.agents[0]

    scores = []
    survived = 0
    kills = 0
    suicides = 0
    coins = 0
    crates = 0
    steps_list = []

    for round_idx in range(n_rounds):
        world.new_round()
        while world.running:
            world.do_step(None)

        scores.append(target_agent.score)
        steps_list.append(world.step)

        if not target_agent.dead:
            survived += 1

        kills += target_agent.statistics['kills']
        suicides += target_agent.statistics['suicides']
        coins += target_agent.statistics['coins']
        crates += target_agent.statistics['crates']

    world.end()

    spawn_name = f"Position_{spawn_idx + 1}" if spawn_idx is not None else "Random_Position"
    coords = str([(1, 1), (1, 15), (15, 1), (15, 15)][spawn_idx]) if spawn_idx is not None else "Any"

    return {
        "agent": agent_name,
        "configuration": spawn_name,
        "spawn_coords": coords,
        "n_rounds": n_rounds,
        "seed": seed,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "survival_rate": float(survived / n_rounds),
        "avg_kills_per_game": float(kills / n_rounds),
        "avg_suicides_per_game": float(suicides / n_rounds),
        "total_coins": int(coins),
        "total_crates": int(crates),
        "avg_steps": float(np.mean(steps_list))
    }


def run_random_position_evaluation(agent_name="project_dqn", n_rounds_per_config=25, seed=42):
    opponents = ["rule_based_agent", "rule_based_agent", "rule_based_agent"]
    results = []

    configs = [
        ("Position 1 (Top-Left: 1,1)", 0),
        ("Position 2 (Bottom-Left: 1,15)", 1),
        ("Position 3 (Top-Right: 15,1)", 2),
        ("Position 4 (Bottom-Right: 15,15)", 3),
        ("Random Position (Permutation)", None)
    ]

    print(f"\n=======================================================")
    print(f"RUNNING STARTING POSITIONS EVALUATION FOR {agent_name}")
    print(f"=======================================================")

    for label, spawn_idx in configs:
        print(f"Evaluating {label} ({n_rounds_per_config} rounds)...")
        res = evaluate_spawn_position(agent_name, opponents, scenario="classic", 
                                      n_rounds=n_rounds_per_config, seed=seed, spawn_idx=spawn_idx)
        results.append(res)
        print(f"  -> Avg Score: {res['avg_score']:.2f} | Survival: {res['survival_rate']*100:.1f}% | Kills: {res['avg_kills_per_game']:.2f}")

    # Save to CSV
    csv_path = os.path.join("report_data", "random_position_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Save to JSON
    json_path = os.path.join("report_data", "random_position_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {csv_path} and {json_path}")

    # Plot performance across spawn positions
    plot_dir = os.path.join("report_data", "evaluation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "starting_position_performance.png")

    labels = ["Pos 1 (1,1)", "Pos 2 (1,15)", "Pos 3 (15,1)", "Pos 4 (15,15)", "Random"]
    scores = [r['avg_score'] for r in results]
    survivals = [r['survival_rate'] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.35

    color = '#1f77b4'
    rects1 = ax1.bar(x - width/2, scores, width, label='Average Score', color=color, alpha=0.85, edgecolor='black')
    ax1.set_ylabel('Average Score', color=color, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, max(scores + [3.5]) * 1.3)

    ax2 = ax1.twinx()
    color = '#2ca02c'
    rects2 = ax2.bar(x + width/2, survivals, width, label='Survival Rate (%)', color=color, alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Survival Rate (%)', color=color, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    plt.title(f"Agent Performance Across Starting Spawn Positions ({agent_name})", fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated starting position performance plot at {plot_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Bomberman Starting Positions")
    parser.add_argument("--agent", type=str, default="project_dqn", help="Agent name to evaluate")
    parser.add_argument("--n-rounds", type=int, default=20, help="Rounds per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    run_random_position_evaluation(args.agent, args.n_rounds, args.seed)
