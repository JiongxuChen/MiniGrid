"""SB3 PPO with only the last rollout shortened to meet an exact step budget."""
from time import perf_counter

from stable_baselines3 import PPO


class BudgetPPO(PPO):
    def __init__(self, *args, **kwargs):
        self.last_update_step = 0
        self.rollout_history = []
        self.collect_seconds = 0.0
        self.update_seconds = 0.0
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        remaining = self._total_timesteps - self.num_timesteps
        if remaining % env.num_envs:
            raise ValueError("Budget must be divisible by n_envs")
        count = min(n_rollout_steps, remaining // env.num_envs)
        if count < 2:
            raise ValueError("Final rollout needs at least 2 steps for normalized advantages")
        if rollout_buffer.buffer_size != count:
            self.rollout_buffer = self.rollout_buffer_class(
                count, self.observation_space, self.action_space, device=self.device,
                gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=env.num_envs,
                **self.rollout_buffer_kwargs,
            )
            rollout_buffer = self.rollout_buffer
        start = perf_counter()
        result = super().collect_rollouts(env, callback, rollout_buffer, count)
        self.collect_seconds += perf_counter() - start
        return result

    def train(self):
        start = perf_counter()
        super().train()
        self.update_seconds += perf_counter() - start
        self.last_update_step = self.num_timesteps
        self.rollout_history.append(dict(end_step=self.num_timesteps,
                                         rollout_steps=self.rollout_buffer.buffer_size,
                                         n_updates=self._n_updates))

