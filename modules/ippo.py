import torch
import torch.nn as nn
import torch.nn.functional as F


class IPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super(IPPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def get_action_and_log_prob(self, obs):
        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist

    def evaluate_actions(self, obs, actions):
        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class IPPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(IPPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        # obs: [batch_size * n_agents, obs_dim] or [n_agents, obs_dim]
        value = self.net(obs)
        return value
