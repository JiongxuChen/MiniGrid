from time import perf_counter

from stable_baselines3.common.callbacks import BaseCallback

from .config import TRAIN_ENV
from .evaluation import evaluate, save_evaluation
from .io import isolated_rng, write_csv, write_json


class PeriodicEvaluation(BaseCallback):
    def __init__(self, output, train_seed, seeds, freq, total_steps):
        super().__init__()
        self.output, self.train_seed, self.seeds = output, train_seed, seeds
        self.freq, self.total_steps = freq, total_steps
        self.eval_seconds = 0.0
        self.save_seconds = 0.0
        self.eval_environment_steps = 0
        self.evaluated_steps = []
        self.in_rollout_seconds = 0.0

    def _on_training_start(self):
        self.evaluate_now(0, final=False)

    def _on_step(self):
        step = self.model.num_timesteps
        if step % self.freq == 0 and step < self.total_steps:
            start = perf_counter()
            self.evaluate_now(step, final=False)
            self.in_rollout_seconds += perf_counter() - start
        return True

    def evaluate_now(self, step, final):
        if step in self.evaluated_steps:
            raise RuntimeError(f"Duplicate evaluation at {step}")
        started = perf_counter()
        path = self.output / ("final_model.zip" if final else f"checkpoints/step_{step:09d}.zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        with isolated_rng():
            self.model.save(path)
        self.save_seconds += perf_counter() - started
        rows, summary = evaluate(self.model, TRAIN_ENV, self.seeds, self.train_seed,
                                 step, "validation", path)
        summary["last_update_step"] = self.model.last_update_step
        summary["after_final_update"] = final
        self.eval_seconds += summary["wall_seconds"]
        self.eval_environment_steps += summary["eval_environment_steps"]
        self.evaluated_steps.append(step)
        save_evaluation(rows, summary, self.output / f"validation/step_{step:09d}.csv",
                        self.output / f"validation/step_{step:09d}.json", path)
        write_csv(self.output / "learning_curve.csv", [summary], append=True)
        write_json(path.with_suffix(".json"), dict(
            checkpoint_step=step, last_update_step=self.model.last_update_step,
            after_final_update=final, resumable_exactly=False,
            model_file_sha256=summary["model_file_sha256"], policy_sha256=summary["policy_sha256"]))
        print(f"[validation] step={step} updated_through={self.model.last_update_step} "
              f"success={summary['success_rate']:.3f} episodes={len(rows)}", flush=True)

