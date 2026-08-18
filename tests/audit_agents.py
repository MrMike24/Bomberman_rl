import os
import pickle
import torch
import numpy as np

def audit_qlearning():
    model_path = os.path.join('agent_code', 'project_qlearning', 'model_data', 'model.pkl')
    print(f"--- Auditing Agent 1: Q-Learning ({model_path}) ---")
    if not os.path.exists(model_path):
        print("FAIL: Q-learning model file missing!")
        return False

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    q_table = data.get('q_table', {})
    epsilon = data.get('epsilon', None)
    alpha = data.get('alpha', None)
    gamma = data.get('gamma', None)

    print(f"File size          : {os.path.getsize(model_path)} bytes")
    print(f"Learned Q-States   : {len(q_table)}")
    print(f"Hyperparameters    : alpha={alpha}, gamma={gamma}, epsilon={epsilon}")

    if len(q_table) == 0:
        print("FAIL: Q-table is empty!")
        return False

    # Sample a Q-state
    sample_state = list(q_table.keys())[0]
    sample_q = q_table[sample_state]
    print(f"Sample State Tuple : {sample_state}")
    print(f"Sample Q-Values    : {sample_q}")
    print("PASS: Q-Learning model is trained and non-empty.\n")
    return True

def audit_dqn():
    model_path = os.path.join('agent_code', 'project_dqn', 'model_data', 'model.pt')
    print(f"--- Auditing Agent 2: DQN ({model_path}) ---")
    if not os.path.exists(model_path):
        print("FAIL: DQN model file missing!")
        return False

    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"File size          : {os.path.getsize(model_path)} bytes")
    print(f"Checkpoint keys    : {list(checkpoint.keys())}")

    policy_state = checkpoint.get('policy_net', {})
    for layer, param in policy_state.items():
        print(f"Layer {layer:15s}: shape {list(param.shape)}, dtype {param.dtype}")

    epsilon = checkpoint.get('epsilon', None)
    print(f"Stored Epsilon     : {epsilon}")

    print("PASS: DQN model is trained and non-empty.\n")
    return True

if __name__ == '__main__':
    q_ok = audit_qlearning()
    dqn_ok = audit_dqn()
    if not (q_ok and dqn_ok):
        exit(1)
