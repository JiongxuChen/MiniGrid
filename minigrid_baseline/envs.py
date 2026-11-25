import hashlib

import gymnasium as gym
import numpy as np
import minigrid  # noqa: F401 -- register MiniGrid environments
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from .config import TEST_ENVS


class EpisodeOutcomeInfo(gym.Wrapper):
    """Record outcome flags only; observations, rewards and dynamics are unchanged."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info, success=int(at_goal(self)),
                        terminated=int(terminated), truncated=int(truncated))
        return obs, reward, terminated, truncated, info


def make_env(env_id, render=False, monitor_path=None):
    if env_id not in TEST_ENVS:
        raise ValueError(f"Baseline supports only {TEST_ENVS}, got {env_id}")
    env = gym.make(env_id, render_mode="rgb_array" if render else None)
    env = ImgObsWrapper(env)
    assert env.observation_space.shape == (7, 7, 3)
    assert env.action_space.n == 7
    env = EpisodeOutcomeInfo(env)
    return Monitor(env, filename=str(monitor_path) if monitor_path else None,
                   info_keywords=("success", "terminated", "truncated"))


def make_train_env(env_id, monitor_path=None):
    # Explicit transpose prevents relying on version-dependent auto-wrapping.
    return VecTransposeImage(DummyVecEnv([lambda: make_env(env_id, monitor_path=monitor_path)]))


def at_goal(env):
    base = env.unwrapped
    cell = base.grid.get(*base.agent_pos)
    return cell is not None and cell.type == "goal"


def initial_state_hash(env):
    base = env.unwrapped
    h = hashlib.sha256(base.grid.encode().tobytes())
    h.update(np.asarray([*base.agent_pos, base.agent_dir], dtype=np.int64).tobytes())
    return h.hexdigest()

