import os
import sys
import time
import tracemalloc
import numpy as np

# Ensure root is in python path
sys.path.insert(0, os.path.abspath('.'))

import agent_code.project_dqn.callbacks as dqn1_callbacks
import agent_code.project_dqn_v2.callbacks as dqn2_callbacks
import agent_code.project_qlearning.callbacks as q1_callbacks
import agent_code.project_qlearning_v2.callbacks as q2_callbacks


class AgentSelf:
    def __init__(self, train=False):
        self.train = train
        self.logger = None


def create_realistic_game_state():
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    for x in range(2, 16, 2):
        for y in range(2, 16, 2):
            field[x, y] = -1
            
    field[3, 3] = 1
    field[3, 4] = 1
    field[5, 2] = 1

    explosion_map = np.zeros((17, 17), dtype=int)
    bombs = [((5, 5), 2), ((8, 8), 4)]
    coins = [(2, 1), (5, 1), (12, 12)]

    game_state = {
        'round': 1,
        'step': 42,
        'field': field,
        'explosion_map': explosion_map,
        'bombs': bombs,
        'coins': coins,
        'self': ('agent_test', 10, True, (1, 1)),
        'others': [('enemy_1', 5, True, (15, 15)), ('enemy_2', 2, False, (1, 15))],
        'user_input': None
    }
    return game_state


def benchmark_callbacks_act(agent_module, name, n_calls=1000):
    print(f"\n=======================================================")
    print(f"BENCHMARKING ACTUAL callbacks.act() FOR {name}")
    print(f"=======================================================")

    agent_self = AgentSelf(train=False)
    agent_module.setup(agent_self)

    game_state = create_realistic_game_state()

    # Warmup 20 iterations
    for _ in range(20):
        agent_module.act(agent_self, game_state)

    tracemalloc.start()
    latencies = []

    for _ in range(n_calls):
        t0 = time.perf_counter()
        act_res = agent_module.act(agent_self, game_state)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat_arr = np.array(latencies)
    mean_lat = np.mean(lat_arr)
    median_lat = np.median(lat_arr)
    p95_lat = np.percentile(lat_arr, 95)
    p99_lat = np.percentile(lat_arr, 99)
    max_lat = np.max(lat_arr)

    print(f"Action Calls Benchmark : {n_calls} calls")
    print(f"Mean Latency           : {mean_lat:.4f} ms")
    print(f"Median Latency         : {median_lat:.4f} ms")
    print(f"P95 Latency            : {p95_lat:.4f} ms")
    print(f"P99 Latency            : {p99_lat:.4f} ms")
    print(f"Max Latency            : {max_lat:.4f} ms")
    print(f"Peak Memory Allocated  : {peak_mem / 1024.0:.2f} KB")
    print(f"Tournament Limit (500ms): Pass = {max_lat < 500.0}")
    print("=======================================================\n")

    return {
        "agent": name,
        "n_calls": n_calls,
        "mean_ms": float(mean_lat),
        "median_ms": float(median_lat),
        "p95_ms": float(p95_lat),
        "p99_ms": float(p99_lat),
        "max_ms": float(max_lat),
        "peak_mem_kb": float(peak_mem / 1024.0),
        "pass_limit": bool(max_lat < 500.0)
    }


def main():
    bench_dqn2 = benchmark_callbacks_act(dqn2_callbacks, "Agent 2 (v2): project_dqn_v2")
    bench_dqn1 = benchmark_callbacks_act(dqn1_callbacks, "Agent 2 (Baseline): project_dqn")
    bench_q2 = benchmark_callbacks_act(q2_callbacks, "Agent 1 (v2): project_qlearning_v2")
    bench_q1 = benchmark_callbacks_act(q1_callbacks, "Agent 1 (Baseline): project_qlearning")


if __name__ == '__main__':
    main()
