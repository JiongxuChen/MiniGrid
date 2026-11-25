import math
import statistics


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def episode_metrics(rows):
    if not rows:
        raise ValueError("No episode results")
    successes = [r for r in rows if int(r["success"])]
    low, high = wilson(len(successes), len(rows))
    return dict(episodes=len(rows), successes=len(successes),
                success_rate=len(successes) / len(rows),
                mean_return=statistics.mean(float(r["return"]) for r in rows),
                median_success_steps=statistics.median(int(r["length"]) for r in successes) if successes else None,
                success_wilson_low=low, success_wilson_high=high)


def curve_metrics(points, budget):
    points = sorted(points, key=lambda r: int(r["checkpoint_step"]))
    steps = [int(r["checkpoint_step"]) for r in points]
    if not points or steps[0] != 0 or steps[-1] != budget or len(set(steps)) != len(steps):
        raise ValueError("AUC requires unique points including step 0 and the final budget")
    rates = [float(r["success_rate"]) for r in points]
    auc = sum((steps[i + 1] - steps[i]) * (rates[i] + rates[i + 1]) / 2
              for i in range(len(steps) - 1))
    threshold = next((step for step, rate in zip(steps, rates) if rate >= 0.8), None)
    return dict(training_auc=auc, training_auc_normalized=auc / budget,
                first_80_percent_step=threshold,
                first_80_percent_status="reached" if threshold is not None else f"not_reached_>{budget}")
