"""Train the reproducible MiniGrid PPO baseline."""
import argparse
import copy
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3.common.logger import configure

from minigrid_baseline.budget_ppo import BudgetPPO
from minigrid_baseline.callbacks import PeriodicEvaluation
from minigrid_baseline.config import (TRAIN_ENV, load_config, load_episode_seeds,
                                      project_path, select_seeds)
from minigrid_baseline.envs import make_train_env
from minigrid_baseline.features import MinigridFeaturesExtractor
from minigrid_baseline.io import metadata, write_json


def run_training(config, seed, output, device="cpu", smoke=False):
    config = copy.deepcopy(config)
    output = Path(output).resolve()
    budget = config["total_steps"]
    n_steps = config["ppo"]["n_steps"]
    remainder = budget % n_steps
    if budget < 2 or remainder == 1:
        raise ValueError("Normalized-advantage PPO needs >=2 samples per final rollout")
    if config["env_id"] != TRAIN_ENV or config["n_envs"] != 1:
        raise ValueError("Baseline trains on DoorKey-6x6 with one environment")
    if not config["deterministic_eval"] or config["policy"]["normalize_images"]:
        raise ValueError("Baseline requires deterministic evaluation and raw symbolic observations")
    if not smoke and (budget != 1000000 or config["eval_freq"] != 50000
                      or config["eval_episodes"] != 100 or seed not in (0, 1, 2)):
        raise ValueError("Non-course budgets/seeds require --smoke and separate outputs")
    if config["eval_freq"] <= 0:
        raise ValueError("eval_freq must be positive")
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use cpu or install a compatible PyTorch build")
    seeds_data = load_episode_seeds(config["episode_seeds"])
    seeds = select_seeds(seeds_data, TRAIN_ENV, "validation", config["eval_episodes"])
    output.mkdir(parents=True, exist_ok=False)
    config.update(train_seed=seed, requested_device=device, smoke=smoke)
    write_json(output / "config.json", config)
    write_json(output / "eval_episode_seeds.json", seeds_data)
    write_json(output / "metadata.json", metadata())
    write_json(output / "status.json", dict(status="running", smoke=smoke))
    env = make_train_env(TRAIN_ENV, output / "training")
    start = perf_counter()
    try:
        model = BudgetPPO("CnnPolicy", env, seed=seed, device=device,
                          policy_kwargs=dict(features_extractor_class=MinigridFeaturesExtractor,
                                             features_extractor_kwargs=dict(features_dim=config["policy"]["features_dim"]),
                                             normalize_images=False),
                          verbose=1, **config["ppo"])
        model.set_logger(configure(str(output / "logs"), ["stdout", "csv"]))
        (output / "model_structure.txt").write_text(str(model.policy) + "\n")
        write_json(output / "resolved_policy.json", dict(
            policy_kwargs=str(model.policy_kwargs), optimizer=str(model.policy.optimizer),
            observation_space=str(model.observation_space), action_space=str(model.action_space),
            actual_device=str(model.device), torch_threads=torch.get_num_threads()))
        callback = PeriodicEvaluation(output, seed, seeds, config["eval_freq"], budget)
        model.learn(total_timesteps=budget, callback=callback, log_interval=1)
        if model.num_timesteps != budget or model.last_update_step != budget:
            raise RuntimeError("Exact budget or final PPO update missing")
        callback.evaluate_now(budget, final=True)
        expected = [0, *range(config["eval_freq"], budget, config["eval_freq"]), budget]
        if callback.evaluated_steps != expected:
            raise RuntimeError("Periodic evaluation/checkpoint schedule mismatch")
        # SB3 normally dumps before train(); explicitly flush final update diagnostics.
        model.logger.dump(model.num_timesteps)
        wall = perf_counter() - start
        timings = dict(job_wall_seconds=wall, periodic_eval_seconds=callback.eval_seconds,
                       checkpoint_save_seconds=callback.save_seconds,
                       sampling_seconds=max(0.0, model.collect_seconds - callback.in_rollout_seconds),
                       ppo_update_seconds=model.update_seconds,
                       training_steps=model.num_timesteps,
                       eval_environment_steps=callback.eval_environment_steps,
                       note="sampling_seconds excludes evaluations; setup/logging/IO remain in job_wall_seconds")
        write_json(output / "timings.json", timings)
        write_json(output / "rollout_history.json", model.rollout_history)
        write_json(output / "status.json", dict(status="complete", smoke=smoke,
                                                training_steps=budget, last_update_step=model.last_update_step))
        return model, callback
    except BaseException as exc:
        write_json(output / "status.json", dict(status="incomplete", smoke=smoke,
                                                error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--env-id", default=TRAIN_ENV, choices=[TRAIN_ENV])
    parser.add_argument("--algo", default="ppo", choices=["ppo"])
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Short diagnostic only; never a course result")
    args = parser.parse_args()
    config = load_config(args.config)
    config["env_id"] = args.env_id
    if args.total_steps is not None:
        config["total_steps"] = args.total_steps
    if args.smoke:
        config.update(total_steps=args.total_steps or 192, eval_freq=80, eval_episodes=2)
        config["ppo"].update(n_steps=128, batch_size=64, n_epochs=1)
    output = args.output or project_path(("outputs/smoke" if args.smoke else config["output_root"]) + f"/seed_{args.seed}")
    run_training(config, args.seed, output, args.device, args.smoke)


if __name__ == "__main__":
    main()

