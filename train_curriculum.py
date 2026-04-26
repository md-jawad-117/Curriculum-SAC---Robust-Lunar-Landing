

import os
import time
import json
import gymnasium as gym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sac import SACAgent


# ─────────────────────── Curriculum Definition ─────────────────────── #

CURRICULUM_STAGES = [
    {
        "name": "Stage 1 — Calm",
        "enable_wind": False,
        "wind_power": 0.0,
        "turbulence_power": 0.0,
        "promotion_threshold": 200,   # avg reward to advance
        "promotion_window": 30,       # over last N episodes
    },
    {
        "name": "Stage 2 — Light Wind",
        "enable_wind": True,
        "wind_power": 8.0,
        "turbulence_power": 0.5,
        "promotion_threshold": 180,
        "promotion_window": 30,
    },
    {
        "name": "Stage 3 — Moderate Wind",
        "enable_wind": True,
        "wind_power": 15.0,
        "turbulence_power": 1.0,
        "promotion_threshold": 160,
        "promotion_window": 30,
    },
    {
        "name": "Stage 4 — Heavy Wind",
        "enable_wind": True,
        "wind_power": 20.0,
        "turbulence_power": 2.0,
        "promotion_threshold": None,  # final stage, no promotion
        "promotion_window": None,
    },
]


class CurriculumScheduler:
    """Manages stage progression based on agent performance."""

    def __init__(self, stages=CURRICULUM_STAGES):
        self.stages = stages
        self.current_stage_idx = 0
        self.stage_history = []        # (episode, stage_idx) transitions
        self.stage_episode_start = 0   # episode when current stage began

    @property
    def current_stage(self):
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self):
        return self.current_stage["name"]

    @property
    def is_final_stage(self):
        return self.current_stage_idx == len(self.stages) - 1

    def make_env(self):
        """Create gymnasium env with current stage parameters."""
        s = self.current_stage
        return gym.make(
            "LunarLanderContinuous-v3",
            enable_wind=s["enable_wind"],
            wind_power=s["wind_power"],
            turbulence_power=s["turbulence_power"],
        )

    def check_promotion(self, episode_rewards, current_episode):
        """Check if agent should advance to the next stage."""
        if self.is_final_stage:
            return False

        s = self.current_stage
        window = s["promotion_window"]
        threshold = s["promotion_threshold"]

        if len(episode_rewards) < window:
            return False

        avg = np.mean(episode_rewards[-window:])
        if avg >= threshold:
            self.stage_history.append((current_episode, self.current_stage_idx))
            self.current_stage_idx += 1
            self.stage_episode_start = current_episode
            return True
        return False

    def summary(self):
        return {
            "current_stage": self.stage_name,
            "stage_idx": self.current_stage_idx,
            "transitions": self.stage_history,
        }


# ─────────────────────── Evaluation ─────────────────────── #

def evaluate(agent, wind_power=0.0, turbulence_power=0.0, enable_wind=False,
             n_episodes=10, seed=42):
    """Evaluate under specific conditions."""
    env = gym.make(
        "LunarLanderContinuous-v3",
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return np.mean(rewards), np.std(rewards)


def evaluate_all_stages(agent, stages=CURRICULUM_STAGES):
    """Evaluate the agent on every curriculum stage — for the final report."""
    results = {}
    for i, s in enumerate(stages):
        mean, std = evaluate(
            agent,
            wind_power=s["wind_power"],
            turbulence_power=s["turbulence_power"],
            enable_wind=s["enable_wind"],
        )
        results[s["name"]] = {"mean": mean, "std": std}
        print(f"  {s['name']:30s} → {mean:>7.1f} ± {std:.1f}")
    return results


# ─────────────────────── Plotting ─────────────────────── #

def moving_average(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_curriculum_training(episode_rewards, stage_transitions, critic_losses,
                              policy_losses, alphas, stage_names,
                              save_path="results/curriculum_training.png"):
    """6-panel training dashboard with stage boundaries."""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Curriculum SAC — LunarLanderContinuous-v3",
                 fontsize=16, fontweight="bold")

    stage_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

    # ── 1. Episode returns with stage shading ──
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
    if len(episode_rewards) > 20:
        ma = moving_average(episode_rewards, 20)
        ax.plot(range(19, 19 + len(ma)), ma, color="darkblue", lw=2, label="20-ep avg")
    ax.axhline(200, ls="--", color="green", alpha=0.4, label="Solved (200)")

    # shade stage regions
    boundaries = [0] + [t[0] for t in stage_transitions] + [len(episode_rewards)]
    for i in range(len(boundaries) - 1):
        color = stage_colors[min(i, len(stage_colors) - 1)]
        ax.axvspan(boundaries[i], boundaries[i + 1], alpha=0.08, color=color)
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        if i < len(stage_names):
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -300,
                    stage_names[min(i, len(stage_names) - 1)],
                    ha="center", fontsize=8, style="italic", alpha=0.7)

    # stage transition lines
    for ep, _ in stage_transitions:
        ax.axvline(ep, color="red", ls=":", alpha=0.6)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Episode Returns (colored by stage)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. Rolling average by stage ──
    ax = axes[0, 1]
    boundaries_full = [0] + [t[0] for t in stage_transitions] + [len(episode_rewards)]
    for i in range(len(boundaries_full) - 1):
        start, end = boundaries_full[i], boundaries_full[i + 1]
        segment = episode_rewards[start:end]
        color = stage_colors[min(i, len(stage_colors) - 1)]
        if len(segment) > 10:
            ma = moving_average(segment, 10)
            ax.plot(range(start + 9, start + 9 + len(ma)), ma, color=color,
                    lw=2, label=stage_names[min(i, len(stage_names) - 1)])
    ax.axhline(200, ls="--", color="green", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (10-ep avg)")
    ax.set_title("Per-Stage Progress")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. Critic loss ──
    ax = axes[1, 0]
    if critic_losses:
        ax.plot(critic_losses, alpha=0.3, color="coral")
        if len(critic_losses) > 50:
            ma = moving_average(critic_losses, 50)
            ax.plot(range(49, 49 + len(ma)), ma, color="darkred", lw=2)
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Loss")
    ax.grid(True, alpha=0.3)

    # ── 4. Policy loss ──
    ax = axes[1, 1]
    if policy_losses:
        ax.plot(policy_losses, alpha=0.3, color="mediumpurple")
        if len(policy_losses) > 50:
            ma = moving_average(policy_losses, 50)
            ax.plot(range(49, 49 + len(ma)), ma, color="indigo", lw=2)
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)

    # ── 5. Alpha ──
    ax = axes[2, 0]
    if alphas:
        ax.plot(alphas, color="teal")
    ax.set_xlabel("Update step")
    ax.set_ylabel("α")
    ax.set_title("Entropy Coefficient (α)")
    ax.grid(True, alpha=0.3)

    # ── 6. Stage timeline ──
    ax = axes[2, 1]
    stage_indices = []
    for ep_idx in range(len(episode_rewards)):
        stage = 0
        for t_ep, t_stage in stage_transitions:
            if ep_idx >= t_ep:
                stage = t_stage + 1
        stage_indices.append(stage)
    ax.step(range(len(stage_indices)), stage_indices, where="post", color="navy", lw=2)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=8)
    ax.set_xlabel("Episode")
    ax.set_title("Curriculum Stage Progression")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Curriculum training curves saved → {save_path}")


def plot_comparison(baseline_rewards, curriculum_rewards,
                    save_path="results/baseline_vs_curriculum.png"):
    """Side-by-side comparison: baseline SAC vs curriculum SAC."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if len(baseline_rewards) > 20:
        ma_b = moving_average(baseline_rewards, 20)
        ax.plot(range(19, 19 + len(ma_b)), ma_b, color="gray", lw=2,
                label="Baseline SAC (heavy wind from start)")
    if len(curriculum_rewards) > 20:
        ma_c = moving_average(curriculum_rewards, 20)
        ax.plot(range(19, 19 + len(ma_c)), ma_c, color="royalblue", lw=2,
                label="Curriculum SAC (progressive)")

    ax.axhline(200, ls="--", color="green", alpha=0.4, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (20-ep moving avg)")
    ax.set_title("Baseline vs Curriculum SAC — LunarLanderContinuous-v3")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Comparison plot saved → {save_path}")


# ─────────────────────── Training Loops ─────────────────────── #

def train_curriculum(episodes=1200, warmup_steps=5000, device="cpu",
                     save_dir="models", results_dir="results"):
    """Train SAC with curriculum learning."""

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    scheduler = CurriculumScheduler()
    env = scheduler.make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=1_000_000,
        device=device,
    )

    episode_rewards = []
    critic_losses, policy_losses, alphas = [], [], []
    best_eval_reward = -float("inf")
    total_steps = 0
    start_time = time.time()

    print("=" * 65)
    print("  Curriculum SAC — LunarLanderContinuous-v3")
    print(f"  Episodes : {episodes}")
    print(f"  Device   : {device}")
    print(f"  Stages   : {len(CURRICULUM_STAGES)}")
    for i, s in enumerate(CURRICULUM_STAGES):
        promo = s['promotion_threshold'] if s['promotion_threshold'] else "—"
        print(f"    [{i+1}] {s['name']:25s} wind={s['wind_power']:>5.1f}  "
              f"turb={s['turbulence_power']:.1f}  promote@{promo}")
    print("=" * 65)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if total_steps >= warmup_steps:
                info = agent.update()
                if info:
                    critic_losses.append(info["critic_loss"])
                    policy_losses.append(info["policy_loss"])
                    alphas.append(info["alpha"])

        episode_rewards.append(ep_reward)

        # ── Check for stage promotion ──
        promoted = scheduler.check_promotion(episode_rewards, ep)
        if promoted:
            env.close()
            env = scheduler.make_env()
            print(f"\n{'='*65}")
            print(f"  ★ PROMOTED to {scheduler.stage_name} at episode {ep}")
            print(f"{'='*65}\n")
            agent.save(f"{save_dir}/sac_stage{scheduler.current_stage_idx}.pt")

        # ── Logging ──
        if ep % 10 == 0:
            avg_20 = np.mean(episode_rewards[-20:])
            elapsed = time.time() - start_time
            alpha_val = alphas[-1] if alphas else 0.0
            print(
                f"Ep {ep:>4d} [{scheduler.stage_name:20s}] | "
                f"Return {ep_reward:>8.1f} | "
                f"Avg20 {avg_20:>8.1f} | "
                f"α {alpha_val:.4f} | "
                f"Steps {total_steps:>7d} | "
                f"Time {elapsed:>6.0f}s"
            )

        # ── Periodic evaluation (on current stage conditions) ──
        if ep % 50 == 0:
            s = scheduler.current_stage
            eval_mean, eval_std = evaluate(
                agent,
                wind_power=s["wind_power"],
                turbulence_power=s["turbulence_power"],
                enable_wind=s["enable_wind"],
            )
            print(f"  └─ Eval: {eval_mean:.1f} ± {eval_std:.1f}")
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                agent.save(f"{save_dir}/sac_curriculum_best.pt")
                print(f"  └─ New best! ({eval_mean:.1f})")

    env.close()

    # ── Final saves ──
    agent.save(f"{save_dir}/sac_curriculum_final.pt")

    stage_names = [s["name"] for s in CURRICULUM_STAGES]
    plot_curriculum_training(
        episode_rewards, scheduler.stage_history,
        critic_losses, policy_losses, alphas, stage_names,
        save_path=f"{results_dir}/curriculum_training.png",
    )

    # ── Evaluate on all stages ──
    print("\n" + "=" * 65)
    print("  Final evaluation across all stages:")
    print("=" * 65)
    eval_results = evaluate_all_stages(agent)

    # ── Save results as JSON ──
    results = {
        "episode_rewards": [float(r) for r in episode_rewards],
        "stage_transitions": scheduler.stage_history,
        "eval_all_stages": {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in eval_results.items()},
        "total_steps": total_steps,
        "total_time": time.time() - start_time,
    }
    with open(f"{results_dir}/curriculum_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Total time: {time.time() - start_time:.0f}s")
    print(f"  Total steps: {total_steps}")
    return episode_rewards


def train_baseline(episodes=1200, warmup_steps=5000, device="cpu",
                   save_dir="models", results_dir="results"):
    """Train standard SAC directly on the hardest conditions (for comparison)."""

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    hardest = CURRICULUM_STAGES[-1]
    env = gym.make(
        "LunarLanderContinuous-v3",
        enable_wind=True,
        wind_power=hardest["wind_power"],
        turbulence_power=hardest["turbulence_power"],
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=1_000_000,
        device=device,
    )

    episode_rewards = []
    total_steps = 0
    start_time = time.time()

    print("=" * 65)
    print("  Baseline SAC — Heavy wind from start")
    print(f"  Episodes : {episodes}")
    print(f"  Wind     : {hardest['wind_power']}, Turb: {hardest['turbulence_power']}")
    print("=" * 65)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if total_steps >= warmup_steps:
                agent.update()

        episode_rewards.append(ep_reward)

        if ep % 10 == 0:
            avg_20 = np.mean(episode_rewards[-20:])
            print(f"Ep {ep:>4d} [Baseline] | Return {ep_reward:>8.1f} | "
                  f"Avg20 {avg_20:>8.1f} | Steps {total_steps:>7d} | "
                  f"Time {time.time()-start_time:>6.0f}s")

    env.close()
    agent.save(f"{save_dir}/sac_baseline_final.pt")

    # save rewards
    with open(f"{results_dir}/baseline_rewards.json", "w") as f:
        json.dump([float(r) for r in episode_rewards], f)

    print(f"\n  Baseline done. Time: {time.time()-start_time:.0f}s")
    return episode_rewards


# ─────────────────────── Main ─────────────────────── #

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPISODES = 1200

    print("\n" + "▓" * 65)
    print("  PHASE 1: Training Curriculum SAC")
    print("▓" * 65 + "\n")
    curriculum_rewards = train_curriculum(episodes=EPISODES, device=device)

    print("\n" + "▓" * 65)
    print("  PHASE 2: Training Baseline SAC (comparison)")
    print("▓" * 65 + "\n")
    baseline_rewards = train_baseline(episodes=EPISODES, device=device)

    print("\n" + "▓" * 65)
    print("  PHASE 3: Generating comparison plot")
    print("▓" * 65 + "\n")
    plot_comparison(baseline_rewards, curriculum_rewards)

    print("\n✓ All done! Check results/ for plots and models/ for checkpoints.")
