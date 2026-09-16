import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (MLP) with Layer Normalization for Double DQN v2.
    """
    def __init__(self, input_dim=38, hidden_dim1=128, hidden_dim2=64, num_actions=6):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_actions)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class DoubleDQNAgent:
    """
    Double Deep Q-Network Agent with Polyak soft target updates,
    action masking, Huber loss, and single-threaded CPU execution guarantee.
    """
    def __init__(self, input_dim=38, num_actions=6, lr=5e-4, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.02, tau=0.005, device=None):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau

        if device is None:
            env_dev = os.environ.get("TORCH_DEVICE")
            if env_dev:
                self.device = torch.device(env_dev)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                torch.set_num_threads(1)
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = QNetwork(input_dim=input_dim, num_actions=num_actions).to(self.device)
        self.target_net = QNetwork(input_dim=input_dim, num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state_vec, safe_action_indices=None, is_training=True):
        """
        Selects an action using epsilon-greedy strategy with safety action masking.
        """
        candidate_indices = safe_action_indices if (safe_action_indices and len(safe_action_indices) > 0) else list(range(self.num_actions))

        if is_training and torch.rand(1).item() < self.epsilon:
            random_choice = int(torch.randint(0, len(candidate_indices), (1,)).item())
            return candidate_indices[random_choice]

        with torch.no_grad():
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0)

            if safe_action_indices and len(safe_action_indices) > 0:
                masked_q = torch.full_like(q_values, float('-inf'))
                for idx in safe_action_indices:
                    masked_q[idx] = q_values[idx]
                best_idx = torch.argmax(masked_q).item()
            else:
                best_idx = torch.argmax(q_values).item()

            return int(best_idx)

    def soft_update_target_net(self):
        """
        Polyak target network update: theta_target = tau * theta_policy + (1 - tau) * theta_target
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update_target_net(self):
        """
        Hard copy from policy network to target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'input_dim': self.input_dim,
            'num_actions': self.num_actions
        }, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint.get('target_net', checkpoint['policy_net']))
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            return True
        return False
