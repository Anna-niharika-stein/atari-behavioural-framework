"""
eval_breakout_ppo.py

Evaluate a pretrained PPO agent on Breakout and write a step-by-step CSV trace.
Part of a thesis project comparing human and AI gameplay on Atari games.

The agent observes pixels only (AtariPreprocessing + VecTransposeImage + VecFrameStack).
For analysis, RAM bytes are also logged as a side-channel via env.unwrapped.ale.getRAM().
The CSV schema matches eval_breakout_dqn.py so results from both agents can be
loaded and compared directly.

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
    python eval_breakout_ppo.py \
        --model_path runs/breakout_ppo/ppo_breakout.zip \
        --episodes 15 \
        --deterministic \
        --no_clip_reward \
        --log_path logs/ppo_breakout_15ep.csv

With rendering:

    python eval_breakout_ppo.py \
        --model_path runs/breakout_ppo/ppo_breakout.zip \
        --episodes 2 \
        --deterministic \
        --render \
        --log_path logs/ppo_breakout_rendered.csv

-------------------------------------------------------------------------------
Protocol notes
-------------------------------------------------------------------------------
- The agent observes greyscale 84x84 frames with frame_skip=4, matching the
  standard Atari preprocessing used during training.

- An AutoFireWrapper automatically sends FIRE after reset and after each life
  loss so the ball is always in play. This avoids the agent stalling in a
  waiting state that was never encountered during training.

- Episodes are capped at max_episode_steps (default 10 000 decision steps) via
  a TimeLimit wrapper, consistent with the DQN Breakout eval script.
"""

import ale_py  # noqa: F401 — registers ALE envs with Gymnasium

import argparse
import csv
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class AutoFireWrapper(gym.Wrapper):
    """Press FIRE after reset and optionally after each life loss.

    Breakout requires a FIRE action to serve the ball. Without this wrapper,
    a pixel-only agent has no incentive to press FIRE and will stall after
    losing a life.

    Args:
        env: The wrapped Gymnasium environment.
        fire_presses: How many FIRE actions to send in sequence.
        on_life_loss: If True, also fire automatically after a life is lost.
    """

    def __init__(self, env: gym.Env, fire_presses: int = 2, on_life_loss: bool = True):
        super().__init__(env)
        self.fire_presses = int(fire_presses)
        self.on_life_loss = bool(on_life_loss)
        self._fire_action: Optional[int] = None
        self._last_lives: Optional[int] = None

    def _find_fire_action(self) -> int:
        meanings = self.env.unwrapped.get_action_meanings()
        if "FIRE" not in meanings:
            raise RuntimeError(f"FIRE not in action meanings: {meanings}")
        return meanings.index("FIRE")

    def _get_lives(self) -> Optional[int]:
        try:
            return int(self.env.unwrapped.ale.lives())
        except Exception:
            return None

    def _do_fire(self) -> Tuple[Any, float, bool, bool, Dict]:
        last = None
        for _ in range(self.fire_presses):
            last = self.env.step(self._fire_action)
            obs, reward, terminated, truncated, info = last
            if terminated or truncated:
                break
        if last is None:
            obs, reward, terminated, truncated, info = self.env.step(self._fire_action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._fire_action = self._find_fire_action()
        self._last_lives = self._get_lives()
        obs2, _, terminated, truncated, info2 = self._do_fire()
        if terminated or truncated:
            return obs2, info2
        self._last_lives = self._get_lives()
        return obs2, info2

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.on_life_loss and not (terminated or truncated):
            lives = self._get_lives()
            if (
                lives is not None
                and self._last_lives is not None
                and lives < self._last_lives
            ):
                obs2, _, term2, trunc2, info2 = self._do_fire()
                obs = obs2
                terminated = term2
                truncated = trunc2
                info.update(info2)
                self._last_lives = self._get_lives()
            else:
                self._last_lives = lives
        return obs, reward, terminated, truncated, info


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
    max_episode_steps: int = 10000,
    fire_presses: int = 2,
    on_life_loss: bool = True,
    frame_skip: int = 4,
    screen_size: int = 84,
    grayscale: bool = True,
    clip_reward: bool = True,
) -> gym.Env:
    """Build and return a wrapped Breakout environment for PPO evaluation.

    Uses AtariPreprocessing (gymnasium) rather than AtariWrapper (SB3) to match
    the preprocessing applied during PPO training.
    """
    env = gym.make(
        "ALE/Breakout-v5",
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        grayscale_newaxis=True,
        scale_obs=False,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    env = AutoFireWrapper(env, fire_presses=fire_presses, on_life_loss=on_life_loss)
    if clip_reward:
        env = gym.wrappers.TransformReward(env, lambda r: float(np.sign(r)))
    if not hasattr(env.action_space, "n") or env.action_space.n != 4:
        raise RuntimeError(f"Expected Discrete(4) action space, got: {env.action_space}")
    return env


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained PPO agent on Breakout and log a CSV trace."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the PPO .zip file.")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment in a window.")
    parser.add_argument("--log_path", type=str, default=r".\logs\ppo_breakout_eval.csv",
                        help="Destination path for the CSV trace file.")
    parser.add_argument("--max_episode_steps", type=int, default=10000,
                        help="Episode step cap (default matches DQN Breakout eval).")
    parser.add_argument("--fire_presses", type=int, default=2,
                        help="Number of FIRE actions to send after reset / life loss.")
    parser.add_argument("--no_life_loss_fire", action="store_true",
                        help="Disable automatic FIRE after a life is lost.")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--no_grayscale", action="store_true",
                        help="Use RGB observations instead of grayscale.")
    parser.add_argument("--no_clip_reward", action="store_true",
                        help="Log raw scores instead of clipped rewards.")
    parser.add_argument("--n_stack", type=int, default=4,
                        help="Number of frames to stack as input to the network.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy (deterministic) action selection.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    render_mode = "human" if args.render else None
    on_life_loss = not args.no_life_loss_fire
    grayscale = not args.no_grayscale
    clip_reward = not args.no_clip_reward

    def env_fn():
        return make_env(
            render_mode=render_mode,
            max_episode_steps=args.max_episode_steps,
            fire_presses=args.fire_presses,
            on_life_loss=on_life_loss,
            frame_skip=args.frame_skip,
            screen_size=args.screen_size,
            grayscale=grayscale,
            clip_reward=clip_reward,
        )

    venv = DummyVecEnv([env_fn])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=args.n_stack)

    model = PPO.load(args.model_path, env=venv)

    # CSV schema is identical to eval_breakout_dqn.py for direct comparability.
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
