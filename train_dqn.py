import argparse
import os
import sys
import settings as s
from main import main as run_main


def train_dqn(n_rounds=100, scenario="coin-heaven", opponents=None, reward_profile="profile_combative_survival", seed=None):
    if opponents is None:
        opponents = ["peaceful_agent", "coin_collector_agent", "rule_based_agent"]

    os.makedirs("logs", exist_ok=True)
    agent_name = os.environ.get("AGENT_NAME", "project_dqn_v2" if os.path.exists(os.path.join("agent_code", "project_dqn_v2")) else "project_dqn")
    cmd_args = [
        "play",
        "--agents", agent_name, *opponents[:s.MAX_AGENTS - 1],
        "--train", "1",
        "--scenario", scenario,
        "--n-rounds", str(n_rounds),
        "--no-gui"
    ]

    if seed is not None:
        cmd_args.extend(["--seed", str(seed)])

    print(f"--- Training DQN ({agent_name}) | Scenario: {scenario} | Rounds: {n_rounds} | Profile: {reward_profile} ---")
    run_main(cmd_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Deep Q-Network Agent")
    parser.add_argument("--n-rounds", type=int, default=100, help="Number of training rounds")
    parser.add_argument("--scenario", type=str, default="classic", choices=list(s.SCENARIOS.keys()), help="Scenario name")
    parser.add_argument("--reward-profile", type=str, default="profile_combative_survival", help="Reward profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_dqn(n_rounds=args.n_rounds, scenario=args.scenario, reward_profile=args.reward_profile, seed=args.seed)
