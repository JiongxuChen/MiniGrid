"""Independent, fixed-map evaluation; observations alone enter the policy."""
from time import perf_counter

import numpy as np

from .envs import at_goal, initial_state_hash, make_env
from .io import file_hash, isolated_rng, policy_hash, write_csv, write_json
from .metrics import episode_metrics


def evaluate(model, env_id, seeds, train_seed, checkpoint_step, split,
             model_path, deterministic=True):
    start = perf_counter()
    rows = []
    before = policy_hash(model)
    mode = model.policy.training
    with isolated_rng():
        env = make_env(env_id)
        try:
            for episode_index, episode_seed in enumerate(seeds):
                obs, _ = env.reset(seed=episode_seed)
                layout_hash = initial_state_hash(env)
                episode_return, length = 0.0, 0
                terminated = truncated = False
                while not (terminated or truncated):
                    chw = np.transpose(obs, (2, 0, 1))
                    action, _ = model.predict(chw, deterministic=deterministic)
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    episode_return += float(reward)
                    length += 1
                success = at_goal(env)
                if success != (episode_return > 0):
                    raise RuntimeError("DoorKey success and original positive reward disagree")
                rows.append(dict(train_seed=train_seed, env_id=env_id, split=split,
                                 checkpoint_step=checkpoint_step, episode_index=episode_index,
                                 episode_seed=episode_seed, success=int(success),
                                 initial_state_sha256=layout_hash,
                                 **{"return": episode_return}, length=length,
                                 terminated=int(terminated), truncated=int(truncated),
                                 deterministic=int(deterministic), model_path=str(model_path),
                                 policy_sha256=before))
        finally:
            env.close()
            model.policy.set_training_mode(mode)
    if policy_hash(model) != before:
        raise RuntimeError("Evaluation changed policy parameters")
    summary = dict(train_seed=train_seed, env_id=env_id, split=split,
                   checkpoint_step=checkpoint_step, **episode_metrics(rows),
                   eval_environment_steps=sum(r["length"] for r in rows),
                   wall_seconds=perf_counter() - start, policy_sha256=before)
    return rows, summary


def save_evaluation(rows, summary, csv_path, summary_path, model_path):
    checkpoint_hash = file_hash(model_path)
    for row in rows:
        row["model_file_sha256"] = checkpoint_hash
    summary["model_file_sha256"] = checkpoint_hash
    write_csv(csv_path, rows)
    write_json(summary_path, summary)

