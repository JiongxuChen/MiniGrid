"""Adapted from Farama's MiniGrid SB3 tutorial (accessed 2026-11-08).

https://minigrid.farama.org/content/training/
Tutorial credits Lucas Willems' rl-starter-files for the CNN architecture.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        if observation_space.shape != (3, 7, 7):
            raise ValueError(f"Expected CHW (3,7,7), got {observation_space.shape}")
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, (2, 2)), nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

