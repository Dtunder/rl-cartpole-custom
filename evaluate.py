"""Seeded comparison of random, LQR, and CEM policies."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import LinearPolicy, train_cem
from controllers import LQRController
from custom_cartpole import WindCartPoleEnv


MAX_STEPS = 500
DEFAULT_SEED = 3100
# The cart actuator delivers +-10 N, so sinusoidal wind above 10 N is not controllable at all;
# conditions are chosen to span from easy to the edge of controllability.
CONDITIONS = [
    ("none", 0.0),
    ("random", 10.0),
    ("random", 20.0),
    ("random", 25.0),
    ("sinusoidal", 5.0),
    ("sinusoidal", 8.0),
    ("sinusoidal", 10.0),
]


def _run_episode(
    mode: str,
    intensity: float,
    seed: int,
    agent_name: str,
    policy: object | None,
) -> tuple[int, float]:
    """Return (steps survived, mean absolute pole angle in degrees)."""
    env = WindCartPoleEnv(wind_mode=mode, wind_intensity=intensity)
    try:
        observation, _ = env.reset(seed=seed)
        if agent_name == "random":
            env.action_space.seed(seed + 10_000)
        abs_angles: list[float] = []
        for steps in range(1, MAX_STEPS + 1):
            if agent_name == "random":
                action = env.action_space.sample()
            else:
                action = policy.act(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            abs_angles.append(abs(float(observation[2])))
            if terminated or truncated:
                return steps, float(np.degrees(np.mean(abs_angles)))
        return MAX_STEPS, float(np.degrees(np.mean(abs_angles)))
    finally:
        env.close()


def evaluate(episodes: int, seed: int, smoke: bool) -> list[dict[str, object]]:
    if episodes < 1:
        raise ValueError("episodes must be positive")

    controller_env = WindCartPoleEnv(wind_mode="none", wind_intensity=0.0)
    lqr = LQRController(controller_env)
    controller_env.close()
    learned = train_cem(
        seed=seed,
        generations=2 if smoke else 4,
        population_size=6 if smoke else 10,
        max_steps=MAX_STEPS,
    )
    policies: list[tuple[str, object | None]] = [
        ("random", None),
        ("LQR", lqr),
        ("CEM linear policy", learned),
    ]

    records: list[dict[str, object]] = []
    for mode, intensity in CONDITIONS:
        returns_by_agent: dict[str, list[int]] = {name: [] for name, _ in policies}
        angles_by_agent: dict[str, list[float]] = {name: [] for name, _ in policies}
        for episode in range(episodes):
            episode_seed = seed + 20_000 + episode
            for agent_name, policy in policies:
                steps, mean_angle = _run_episode(mode, intensity, episode_seed, agent_name, policy)
                returns_by_agent[agent_name].append(steps)
                angles_by_agent[agent_name].append(mean_angle)
        for agent_name, returns in returns_by_agent.items():
            values = np.asarray(returns, dtype=np.float64)
            records.append(
                {
                    "wind_mode": mode,
                    "wind_intensity": f"{intensity:.1f}",
                    "agent": agent_name,
                    "n_episodes": episodes,
                    "mean_return": f"{values.mean():.3f}",
                    "std_return": f"{values.std(ddof=0):.3f}",
                    "median_return": f"{np.median(values):.3f}",
                    "success_rate": f"{np.mean(values >= MAX_STEPS):.3f}",
                    "mean_abs_angle_deg": f"{np.mean(angles_by_agent[agent_name]):.3f}",
                }
            )
    return records


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "wind_mode", "wind_intensity", "agent", "n_episodes",
        "mean_return", "std_return", "median_return", "success_rate",
        "mean_abs_angle_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_plot(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['wind_mode']}\n{float(r['wind_intensity']):g} N" for r in records if r["agent"] == "random"]
    agents = ["random", "LQR", "CEM linear policy"]
    x = np.arange(len(labels), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for index, agent in enumerate(agents):
        rows = [r for r in records if r["agent"] == agent]
        means = [float(r["mean_return"]) for r in rows]
        deviations = [float(r["std_return"]) for r in rows]
        ax.bar(x + (index - 1) * width, means, width, yerr=deviations, capsize=2, label=agent)
    ax.axhline(MAX_STEPS, color="black", linewidth=0.8, linestyle="--")
    ax.set(title="WindCartPole: steps survived under wind disturbance (12 seeded episodes)", ylabel="Steps survived (cap 500)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, MAX_STEPS + 60)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=160, metadata={"Software": "WindCartPole evaluation"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a shorter CI smoke evaluation")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    episodes = 2 if args.smoke else args.episodes
    records = evaluate(episodes=episodes, seed=args.seed, smoke=args.smoke)
    csv_path = args.output_dir / "results.csv"
    plot_path = args.output_dir / "comparison.png"
    write_csv(records, csv_path)
    write_plot(records, plot_path)
    print(f"Wrote {csv_path} and {plot_path} ({episodes} seeded episodes per condition and agent)")
    for row in records:
        print(
            f"{row['wind_mode']:11s} {float(row['wind_intensity']):3.1f} "
            f"{row['agent']:18s} mean={float(row['mean_return']):7.2f} "
            f"std={float(row['std_return']):7.2f} success={float(row['success_rate']):.1%}"
        )


if __name__ == "__main__":
    main()
