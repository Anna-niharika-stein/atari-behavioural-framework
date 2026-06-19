"""
eval_pong_dqn.py

Evaluate a pretrained DQN agent on Pong and write a step-by-step CSV trace.
Part of a thesis project comparing human and AI gameplay on Atari games.

The agent observes pixels only (AtariWrapper handles preprocessing internally).
For analysis, RAM bytes are also logged as a side-channel via env.unwrapped.ale.getRAM().
The CSV schema matches eval_pong_ppo.py so results from both agents can be
loaded and compared directly.

-------------------------------------------------------------------------------
Setup
-------------------------------------------------------------------------------
Download the pretrained model from the SB3 Zoo before running:

    pip install rl_zoo3 huggingface_sb3
    python -m rl_zoo3.load_from_hub --algo dqn --env PongNoFrameskip-v4 \
        -orga sb3 -f logs/

The model file will be saved to:
    logs/dqn/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
    python eval_pong_dqn.py \
        --model_path "logs/dqn/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip" \
        --episodes 15 \
        --deterministic \
        --no_clip_reward \
        --log_path "logs/dqn_pong_15ep.csv"

-------------------------------------------------------------------------------
Protocol notes
-------------------------------------------------------------------------------
- terminal_on_life_loss is set to False for evaluation. Pong has no lives
  mechanic so this has no practical effect, but it is kept consistent with
  the Breakout DQN eval and all PPO eval scripts for a uniform protocol.

- Unlike Breakout, Pong does not require a FIRE action to start play, so
  no AutoFireWrapper is needed here.
"""

import ale_py  # noqa: F401 — registers ALE envs with Gymnasium

import argparse
import csv
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ram(env) -> np.ndarray:
    """Return a copy of the current ALE RAM (128 bytes)."""
    return np.array(env.unwrapped.ale.getRAM(), dtype=np.uint8).copy()


def get_lives(env) -> Optional[int]:
    """Return the current number of lives, or None if unavailable."""
    try:
        return int(env.unwrapped.ale.lives())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    render_mode=None,
    clip_reward: bool = True,
    terminal_on_life_loss: bool = False,
    frame_skip: int = 4,
    screen_size: int = 84,
    noop_max: int = 30,
) -> gym.Env:
    """Build and return a wrapped Pong environment for DQN evaluation.

    Uses PongNoFrameskip-v4 (v4, not v5) to match the environment the Zoo
    DQN model was originally trained on.
    """
    env = gym.make("PongNoFrameskip-v4", render_mode=render_mode)
    env = AtariWrapper(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        clip_reward=clip_reward,
        action_repeat_probability=0.0,
    )
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained DQN agent on Pong and log a CSV trace."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the DQN .zip file downloaded from the SB3 Zoo.")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment in a window.")
    parser.add_argument("--log_path", type=str, default=r".\logs\dqn_pong_eval.csv",
                        help="Destination path for the CSV trace file.")
    parser.add_argument("--noop_max", type=int, default=30,
                        help="Max number of no-op actions at the start of each episode.")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--no_clip_reward", action="store_true",
                        help="Log raw scores instead of clipped rewards.")
    parser.add_argument("--n_stack", type=int, default=4,
                        help="Number of frames to stack as input to the network.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy (deterministic) action selection.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    render_mode = "human" if args.render else None
    clip_reward = not args.no_clip_reward

    def env_fn():
        return make_env(
            render_mode=render_mode,
            clip_reward=clip_reward,
            terminal_on_life_loss=False,
            frame_skip=args.frame_skip,
            screen_size=args.screen_size,
            noop_max=args.noop_max,
        )

    venv = DummyVecEnv([env_fn])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=args.n_stack)

    model = DQN.load(
        args.model_path,
        env=venv,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
            "optimize_memory_usage": False,
            "handle_timeout_termination": False,
        },
    )

    # CSV schema is identical to eval_pong_ppo.py for direct comparability.
    ram_pre_cols = [f"ram_pre_{i}" for i in range(128)]
    ram_post_cols = [f"ram_post_{i}" for i in range(128)]
    fieldnames = [
        "run_ts", "episode", "step", "action", "reward", "done",
        "episode_return", "lives_pre", "lives_post",
    ] + ram_pre_cols + ram_post_cols

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_env = venv.envs[0]

    with open(args.log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, args.episodes + 1):
            obs = venv.reset()
            done = False
            ep_ret = 0.0
            step_idx = 0

            while not done:
                ram_pre = get_ram(base_env)
                lives_pre = get_lives(base_env)

                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, dones, infos = venv.step(action)

                ram_post = get_ram(base_env)
                lives_post = get_lives(base_env)

                r = float(reward[0])
                done = bool(dones[0])
                ep_ret += r

                row = {
                    "run_ts": run_ts,
                    "episode": ep,
                    "step": step_idx,
                    "action": int(action[0]),
                    "reward": r,
                    "done": int(done),
                    "episode_return": ep_ret,
                    "lives_pre": lives_pre if lives_pre is not None else "",
                    "lives_post": lives_post if lives_post is not None else "",
                }
                for i in range(128):
                    row[f"ram_pre_{i}"] = int(ram_pre[i])
                    row[f"ram_post_{i}"] = int(ram_post[i])

                writer.writerow(row)
                step_idx += 1

            print(f"Episode {ep}: return={ep_ret:.2f}, steps={step_idx}")

    venv.close()
    print(f"\nWrote eval log to:\n  {os.path.abspath(args.log_path)}")


if __name__ == "__main__":
    main()
