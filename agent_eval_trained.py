import argparse
import numpy as np

from common_logging import EnvConfig, make_env, StepCSVLogger


def load_sb3_model(model_path: str):
    """Loads a Stable-Baselines3 model. Tries PPO then DQN."""
    try:
        from stable_baselines3 import PPO, DQN  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "stable-baselines3 is not installed. Install it with:\n"
            "  pip install stable-baselines3[extra]"
        ) from e

    last_err = None
    for cls in (PPO, DQN):
        try:
            return cls.load(model_path)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load model as PPO or DQN: {model_path}\nLast error: {last_err}")


def get_action_meanings(env):
    try:
        return env.unwrapped.get_action_meanings()
    except Exception:
        return None


def get_fire_action(env) -> int:
    meanings = get_action_meanings(env)
    if meanings and "FIRE" in meanings:
        return meanings.index("FIRE")
    return 1  # Breakout typical


def get_lives(env) -> int:
    """Returns current lives if available, else -1."""
    try:
        return int(env.unwrapped.ale.lives())
    except Exception:
        return -1


def step_and_log(env, logger, episode_id, t, ep_return, ep_len, obs, action):
    """
    Take one env.step(action) and log exactly one row with the PRE-STEP obs.
    Returns updated (next_obs, terminated, truncated, ep_return, ep_len, t).
    """
    next_obs, reward, terminated, truncated, info = env.step(int(action))

    ep_return += float(reward)
    ep_len += 1

    logger.log_step(
        episode_id=episode_id,
        timestep=t,
        action=int(action),
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        ep_return_so_far=ep_return,
        ep_length_so_far=ep_len,
        ram_obs=obs,
    )

    t += 1
    return next_obs, terminated, truncated, ep_return, ep_len, t


def auto_fire(env, logger, episode_id, t, ep_return, ep_len, obs, presses=2):
    """
    Press FIRE a few times AND LOG those steps so the trace stays faithful.
    """
    fire_action = get_fire_action(env)
    terminated = False
    truncated = False

    for _ in range(presses):
        obs, terminated, truncated, ep_return, ep_len, t = step_and_log(
            env, logger, episode_id, t, ep_return, ep_len, obs, fire_action
        )
        if terminated or truncated:
            break

    return obs, terminated, truncated, ep_return, ep_len, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to SB3 .zip model (PPO/DQN).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--log_path", type=str, default="logs/agent_trace.csv")
    parser.add_argument("--render", action="store_true", help="Render to a human window (slower).")
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--sticky", type=float, default=0.0, help="repeat_action_probability")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ram_wide", action="store_true", help="Log RAM as 128 columns (recommended).")

    # Optional helpers (for demos / sanity checks)
    parser.add_argument("--auto_fire_on_reset", action="store_true", help="Press FIRE after reset (serve ball).")
    parser.add_argument(
        "--auto_fire_on_life_loss",
        action="store_true",
        help="Press FIRE after a detected life loss (serve again).",
    )
    parser.add_argument("--auto_fire_presses", type=int, default=2, help="How many FIRE presses when auto-fire triggers.")

    args = parser.parse_args()

    cfg = EnvConfig(frameskip=args.frameskip, repeat_action_probability=args.sticky, seed=args.seed)
    env = make_env(cfg, render_mode="human" if args.render else None)

    obs, info = env.reset(seed=cfg.seed)
    if not (isinstance(obs, np.ndarray) and obs.shape == (128,)):
        raise RuntimeError(f"Expected RAM obs shape (128,), got: {type(obs)} {getattr(obs, 'shape', None)}")

    meanings = get_action_meanings(env)
    if meanings:
        print("Action meanings:", meanings)

    model = load_sb3_model(args.model_path)
    logger = StepCSVLogger(args.log_path, log_ram_as_128_columns=args.ram_wide)

    returns, lengths = [], []

    episode_id = 0
    while episode_id < args.episodes:
        terminated = False
        truncated = False
        ep_return = 0.0
        ep_len = 0
        t = 0

        # Auto-FIRE once per episode (if enabled)
        if args.auto_fire_on_reset:
            obs, terminated, truncated, ep_return, ep_len, t = auto_fire(
                env, logger, episode_id, t, ep_return, ep_len, obs, presses=args.auto_fire_presses
            )

        prev_lives = get_lives(env)

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, terminated, truncated, ep_return, ep_len, t = step_and_log(
                env, logger, episode_id, t, ep_return, ep_len, obs, action
            )

            # Auto-FIRE after life loss (optional; can be noisy depending on lives reporting)
            if args.auto_fire_on_life_loss and not (terminated or truncated):
                lives = get_lives(env)
                # Only treat as life loss when both readings are valid and it actually decreases
                if prev_lives >= 0 and lives >= 0 and lives < prev_lives:
                    obs, terminated, truncated, ep_return, ep_len, t = auto_fire(
                        env, logger, episode_id, t, ep_return, ep_len, obs, presses=args.auto_fire_presses
                    )
                prev_lives = lives

        returns.append(ep_return)
        lengths.append(ep_len)

        episode_id += 1
        obs, info = env.reset(seed=args.seed + episode_id)

    logger.close()
    env.close()

    if returns:
        print(f"Eval episodes: {len(returns)}")
        print(f"Return mean/std: {float(np.mean(returns)):.3f} / {float(np.std(returns)):.3f}")
        print(f"Length mean/std: {float(np.mean(lengths)):.1f} / {float(np.std(lengths)):.1f}")

    print(f"Saved trained-agent trace to: {args.log_path}")


if __name__ == "__main__":
    main()
