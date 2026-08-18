import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim=29, hidden_dim1=128, hidden_dim2=64, num_actions=6):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, input_dim=31, num_actions=6, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # CPU single thread enforcement for tournament compliance
        torch.set_num_threads(1)
        self.device = torch.device("cpu")

        self.policy_net = QNetwork(input_dim=input_dim, num_actions=num_actions).to(self.device)
        self.target_net = QNetwork(input_dim=input_dim, num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state_vec, safe_action_indices=None, is_training=True):
        if is_training and torch.rand(1).item() < self.epsilon:
            if safe_action_indices and len(safe_action_indices) > 0:
                random_choice = int(torch.randint(0, len(safe_action_indices), (1,)).item())
                return safe_action_indices[random_choice]
            return int(torch.randint(0, self.num_actions, (1,)).item())

        with torch.no_grad():
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0)

            if safe_action_indices and len(safe_action_indices) > 0:
                # Mask unsafe actions with -inf
                masked_q = torch.full_like(q_values, float('-inf'))
                for idx in safe_action_indices:
                    masked_q[idx] = q_values[idx]
                best_idx = torch.argmax(masked_q).item()
            else:
                best_idx = torch.argmax(q_values).item()

            return best_idx

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
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
