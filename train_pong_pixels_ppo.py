"""
train_pong_pixels_ppo.py

Pixel-only PPO training on ALE/Pong-v5. RAM snapshots are written into `info`
via RAMLoggingWrapper for logging compatibility with the Breakout setup, but the
policy only ever sees pixels.

Why PPO:
- No replay buffer, so no large memory allocations with pixel observations
- Scales well with n_envs (8+)

Pong vs. Breakout differences:
- Discrete(6) action space: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
- Point-based scoring, no life counter -> on_life_loss_fire defaults to False

Wrapper stack (should match eval later):
Gym ALE -> AtariPreprocessing (resize/skip/grayscale) -> TimeLimit -> Monitor
        -> AutoFireWrapper -> RAMLoggingWrapper -> (optional reward clip)
VecEnv -> VecTransposeImage -> VecFrameStack
"""

import ale_py  # noqa: F401  # ensures ALE is available

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# -----------------------------
# Wrappers
# -----------------------------
class AutoFireWrapper(gym.Wrapper):
    """
    Automatically presses FIRE:
      - a few times after reset (to serve)
      - after a life loss (to re-serve), if enabled

    For Pong, on_life_loss defaults to False because Pong is point-based
    and does not use a life counter the way Breakout does.
    """

    def __init__(self, env: gym.Env, fire_presses: int = 2, on_life_loss: bool = False):
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
            if lives is not None and self._last_lives is not None and lives < self._last_lives:
                obs2, _, term2, trunc2, info2 = self._do_fire()
                obs = obs2
                terminated = term2
                truncated = trunc2
                info.update(info2)
                self._last_lives = self._get_lives()
            else:
                self._last_lives = lives

        return obs, reward, terminated, truncated, info


class RAMLoggingWrapper(gym.Wrapper):
    """
    Adds RAM snapshots to `info` (side-channel only):
      - info["ram_pre"]   = RAM before step
      - info["ram_post"]  = RAM after step
      - info["ram_reset"] = RAM after reset (optional)

    NOTE: This does NOT change the observation; agent still sees pixels.
    """

    def __init__(self, env: gym.Env, include_reset: bool = True):
        super().__init__(env)
        self.include_reset = bool(include_reset)

    def _read_ram(self) -> np.ndarray:
        ram = self.env.unwrapped.ale.getRAM()
        return np.array(ram, dtype=np.uint8).copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.include_reset:
            info = dict(info) if info is not None else {}
            info["ram_reset"] = self._read_ram()
        return obs, info

    def step(self, action):
        ram_pre = self._read_ram()
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram_post = self._read_ram()
        info = dict(info) if info is not None else {}
        info["ram_pre"] = ram_pre
        info["ram_post"] = ram_post
        return obs, reward, terminated, truncated, info


# -----------------------------
# Env factory
# -----------------------------
def make_pong_pixels_env(
    render_mode=None,
    max_episode_steps: int = 10_000,
    fire_presses: int = 2,
    on_life_loss: bool = False,
    frame_skip: int = 4,
    screen_size: int = 84,
    grayscale: bool = True,
    clip_reward: bool = True,
):
    env = gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",
        frameskip=1,  # frame skipping is handled by AtariPreprocessing
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=render_mode,
    )

    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        grayscale_newaxis=True,  # (H, W, 1) so SB3 recognises it as an image space
        scale_obs=False,
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    env = AutoFireWrapper(env, fire_presses=fire_presses, on_life_loss=on_life_loss)
    env = RAMLoggingWrapper(env, include_reset=True)

    if clip_reward:
        env = gym.wrappers.TransformReward(env, lambda r: float(np.sign(r)))

    if not hasattr(env.action_space, "n"):
        raise RuntimeError(f"Expected a discrete action space, got: {env.action_space}")

    return env


# -----------------------------
# Paths
# -----------------------------
@dataclass
class Paths:
    out_dir: str
    model_path: str
    best_model_dir: str
    checkpoint_dir: str


def build_paths(out_dir: str) -> Paths:
    os.makedirs(out_dir, exist_ok=True)
    best_model_dir = os.path.join(out_dir, "best_model")
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return Paths(
        out_dir=out_dir,
        model_path=os.path.join(out_dir, "ppo_pong_pixels.zip"),
        best_model_dir=best_model_dir,
        checkpoint_dir=checkpoint_dir,
    )


# -----------------------------
# Train
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=r".\runs\pong_pixels_ppo")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=10_000)

    parser.add_argument("--fire_presses", type=int, default=2)
    # on_life_loss_fire is off by default for Pong; pass the flag to enable it
    parser.add_argument("--on_life_loss_fire", action="store_true")

    # Pixel preprocessing (must match eval later)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--no_grayscale", action="store_true")
    parser.add_argument("--no_clip_reward", action="store_true")
    parser.add_argument("--n_stack", type=int, default=4)

    # PPO hyperparams (Atari-ish defaults)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--n_steps", type=int, default=128)        # rollout length per env
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--eval_freq", type=int, default=250_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--checkpoint_freq", type=int, default=500_000)

    args = parser.parse_args()
    paths = build_paths(args.out_dir)

    grayscale = not args.no_grayscale
    clip_reward = not args.no_clip_reward

    def env_fn():
        return make_pong_pixels_env(
            render_mode=None,
            max_episode_steps=args.max_episode_steps,
            fire_presses=args.fire_presses,
            on_life_loss=args.on_life_loss_fire,
            frame_skip=args.frame_skip,
            screen_size=args.screen_size,
            grayscale=grayscale,
            clip_reward=clip_reward,
        )

    # Training env
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)
    vec_env = VecTransposeImage(vec_env)  # HWC -> CHW
    vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)

    # Eval env
    eval_env = make_vec_env(env_fn, n_envs=1, seed=args.seed + 123)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=paths.best_model_dir,
        log_path=os.path.join(paths.out_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=paths.checkpoint_dir,
        name_prefix="ppo_checkpoint",
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=os.path.join(paths.out_dir, "tb"),
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
    )

    model.save(paths.model_path)

    vec_env.close()
    eval_env.close()

    print("\nSaved:")
    print("  Model:", paths.model_path)
    print("  Best model dir:", paths.best_model_dir)
    print("  Checkpoints:", paths.checkpoint_dir)


if __name__ == "__main__":
    main()
