import argparse
import json
import os
import numpy as np
import settings as s
from main import main as run_main
from environment import BombeRLeWorld


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


def evaluate_agent(agent_name, opponents=None, scenario="classic", n_rounds=50, seed=42):
    if opponents is None:
        opponents = ["rule_based_agent", "rule_based_agent", "rule_based_agent"]

    args = DummyArgs(agent_name, opponents, scenario, n_rounds, seed)
    agents_spec = [(name, False) for name in args.agents]

    world = BombeRLeWorld(args, agents_spec)

    target_agent = world.agents[0]

    scores = []
    survived = 0
    kills = 0
    suicides = 0
    deaths = 0
    coins = 0
    crates = 0
    invalid_actions = 0
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
        invalid_actions += target_agent.statistics['invalid']

    world.end()

    metrics = {
        "agent_name": agent_name,
        "scenario": scenario,
        "n_rounds": n_rounds,
        "seed": seed,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "max_score": float(np.max(scores)),
        "survival_rate": float(survived / n_rounds),
        "avg_steps": float(np.mean(steps_list)),
        "total_kills": int(kills),
        "avg_kills_per_game": float(kills / n_rounds),
        "total_suicides": int(suicides),
        "avg_suicides_per_game": float(suicides / n_rounds),
        "total_coins": int(coins),
        "total_crates": int(crates),
        "total_invalid": int(invalid_actions)
    }

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Bomberman Agent")
    parser.add_argument("--agent", type=str, default="project_qlearning", help="Agent folder name")
    parser.add_argument("--opponents", nargs="+", default=["rule_based_agent", "rule_based_agent", "rule_based_agent"], help="Opponent agent names")
    parser.add_argument("--scenario", type=str, default="classic", choices=list(s.SCENARIOS.keys()), help="Scenario name")
    parser.add_argument("--n-rounds", type=int, default=50, help="Number of evaluation rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="JSON output file path")

    args = parser.parse_args()
    results = evaluate_agent(args.agent, args.opponents, args.scenario, args.n_rounds, args.seed)

    print("\n================ EVALUATION RESULTS ================")
    for k, v in results.items():
        print(f"{k:22s}: {v}")
    print("===================================================\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
