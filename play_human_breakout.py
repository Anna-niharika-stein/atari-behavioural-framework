"""
play_human_breakout.py
======================
Human gameplay interface for ALE/Breakout-v5 with full trace logging.

OVERVIEW
--------
This script lets a human player play Breakout through a Pygame window and logs
every decision step to a CSV file in the same format as the agent evaluation logs.
This makes the resulting human trace directly comparable to PPO and DQN agent
traces using the behavioural trace framework described in the thesis.

HOW IT WORKS
------------
1. The ALE/Breakout-v5 environment is created with frameskip=1 and sticky actions
   disabled. This means the environment steps forward one raw ALE frame at a time,
   and every action issued is applied exactly as given (no randomised repetition
   of the previous action).

2. A Pygame window displays the game at a scaled resolution. The script reads
   the currently held keyboard keys on every decision step rather than relying
   on OS key-repeat events, which gives more responsive paddle control.

3. The human player controls the paddle using the arrow keys:
       LEFT arrow  — move paddle left
       RIGHT arrow — move paddle right
       SPACE/ENTER — fire (serve the ball)
       P           — pause / unpause
       ESC         — quit early

4. Serving is handled automatically at the start of each episode and after
   each life loss, matching the AutoFireWrapper behaviour used in the PPO
   agent pipeline. As a result, the human never needs to press FIRE to serve,
   and FIRE actions in the log reflect only deliberate key presses.

5. On every decision step, before and after the environment steps, the full
   128-byte ALE RAM vector is read directly from the emulator. This RAM snapshot
   is stored in the CSV alongside the action, reward, and episode metadata.
   RAM is not used to control the paddle; it is logged purely as a compact
   side-channel state descriptor for the downstream behavioural analysis.

6. One logged CSV row corresponds to one decision step: the chosen action is
   held for `action_repeat` consecutive raw ALE frames, and the reward is the
   sum over those frames. With action_repeat=1 (the default for this script),
   each row is one raw ALE frame. This differs from the agent logs where
   action_repeat=4 (matching AtariPreprocessing frame_skip=4). The mismatch
   is handled analytically during preprocessing: human frames are aggregated
   into non-overlapping 4-frame windows before action-structure and
   state-visitation descriptors are computed.

7. All internal frames within a decision step are rendered to the Pygame window
   for smooth visual feedback, even though only one row is logged per step.

OUTPUT
------
A timestamped CSV is written to .\\logs\\human by default. Each row corresponds
to one decision step and contains:

    run_ts          — session timestamp (links rows to a recording session)
    episode         — episode number
    step            — decision step index within the episode
    action          — integer action selected at this step
    reward          — reward summed over action_repeat frames
    done            — 1 if the episode ended at this step, 0 otherwise
    episode_return  — running cumulative reward for this episode
    lives_pre       — lives before the step
    lives_post      — lives after the step
    ram_pre_0..127  — 128-byte ALE RAM vector before the action
    ram_post_0..127 — 128-byte ALE RAM vector after the action

USAGE
-----
    # Run one episode (default):
    # python play_human_breakout.py

    # Run 5 episodes:
    # python play_human_breakout.py --episodes 5

    # Run with action_repeat=4 to match agent decision cadence exactly:
    # python play_human_breakout.py --episodes 1 --action_repeat 4

    # Full recommended command matching PPO eval defaults (used in thesis analysis):
    # python play_human_breakout.py --episodes 1 --display_scale 4 --frame_fps 35 --action_repeat 1 --auto_serve

REQUIREMENTS
------------
    pip install "gymnasium[atari,accept-rom-license]" ale-py pygame numpy

NOTES
-----
- Tested with gymnasium>=0.29, ale-py>=0.9, pygame>=2.1, numpy>=1.24.
- The CSV schema matches the agent evaluation logs produced by the thesis
  pipeline, enabling direct comparison through the behavioural trace framework.
- action_repeat defaults to 1 for responsive human play. The agent uses
  action_repeat=4 via AtariPreprocessing. This mismatch is resolved during
  preprocessing by aggregating human rows into 4-frame decision windows.
- To change the number of episodes, FPS, display scale, or other settings,
  edit the defaults in the argparse block below or pass command-line arguments
  (see USAGE above).
"""

import ale_py  # noqa: F401 — registers ALE environments with Gymnasium

import argparse
import csv
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


# ---------------------------------------------------------------------------
# ALE side-channel helpers
# ---------------------------------------------------------------------------

def get_ram(env) -> np.ndarray:
    """Read the full 128-byte ALE RAM vector from the emulator."""
    ram = env.unwrapped.ale.getRAM()
    return np.array(ram, dtype=np.uint8).copy()


def get_lives(env) -> Optional[int]:
    """Return the current life count, or None if not available."""
    try:
        return int(env.unwrapped.ale.lives())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    render_mode: str = "rgb_array",
    max_episode_steps: int = 10_000,
    sticky_actions: float = 0.0,
):
    """
    Create the ALE Breakout environment with the settings used in the thesis pipeline.

    frameskip=1              — step one raw ALE frame at a time; decision cadence
                               is handled by action_repeat in the main loop.
    full_action_space=False  — use the reduced 4-action space: NOOP, FIRE, RIGHT, LEFT.
    repeat_action_probability=0.0 — sticky actions disabled.
    """
    env = gym.make(
        "ALE/Breakout-v5",
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=sticky_actions,
        full_action_space=False,
        render_mode=render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if not hasattr(env.action_space, "n") or env.action_space.n != 4:
        raise RuntimeError(f"Expected Discrete(4) action space, got: {env.action_space}")

    return env


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def blit_scaled(screen, rgb: np.ndarray, out_w: int, out_h: int):
    """Render a raw RGB frame to the Pygame window at the configured scale."""
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (out_w, out_h))
    screen.blit(surf, (0, 0))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Keyboard → action mapping
# ---------------------------------------------------------------------------

def sample_key_action(action_left: int, action_right: int, action_fire: int, action_noop: int) -> int:
    """
    Read the currently held keys and return the corresponding action index.

    FIRE takes priority over directional input. Contradictory left+right input
    is resolved to NOOP to avoid jitter.
    """
    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
        return action_fire

    left  = bool(keys[pygame.K_LEFT])
    right = bool(keys[pygame.K_RIGHT])

    if left and not right:
        return action_left
    if right and not left:
        return action_right

    return action_noop


# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

def main():
    if pygame is None:
        raise RuntimeError("pygame not installed. Install with:\n  pip install pygame")

    parser = argparse.ArgumentParser(description="Human Breakout gameplay interface with RAM trace logging.")
    parser.add_argument("--episodes",         type=int,   default=1,             help="Number of episodes to record (default: %(default)s)")
    parser.add_argument("--out_dir",          type=str,   default=r".\logs\human", help="Output directory for CSV log (default: %(default)s)")
    parser.add_argument("--log_path",         type=str,   default=None,          help="Override auto-generated log path with a specific file path")
    parser.add_argument("--frame_fps",        type=int,   default=60,            help="Target display frame rate (default: %(default)s)")
    parser.add_argument("--display_scale",    type=int,   default=4,             help="Pygame window scale factor (default: %(default)s)")
    parser.add_argument("--start_paused",     action="store_true", default=False, help="Start each episode paused")
    parser.add_argument("--action_repeat",    type=int,   default=1,             help="Raw env steps per logged decision step; set to 4 to match agent frame_skip (default: %(default)s)")
    parser.add_argument("--auto_serve",       action="store_true", default=True,  help="Automatically press FIRE after reset and life loss (default: True)")
    parser.add_argument("--seed",             type=int,   default=0,             help="Base random seed (default: %(default)s)")
    parser.add_argument("--max_episode_steps",type=int,   default=10_000,        help="Maximum steps per episode (default: %(default)s)")
    parser.add_argument("--sticky_actions",   type=float, default=0.0,           help="Sticky action probability (default: %(default)s)")
    args = parser.parse_args()

    scale         = max(1, int(args.display_scale))
    action_repeat = max(1, int(args.action_repeat))
    frame_fps     = max(1, int(args.frame_fps))

    # Resolve output path.
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.log_path is None:
        os.makedirs(args.out_dir, exist_ok=True)
        log_path = os.path.join(args.out_dir, f"human_pixels_{run_ts}.csv")
    else:
        os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
        log_path = args.log_path

    # Set up environment.
    env = make_env(
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
        sticky_actions=args.sticky_actions,
    )

    meanings = env.unwrapped.get_action_meanings()
    print("Action meanings:", meanings)
    ACTION_NOOP  = meanings.index("NOOP")
    ACTION_FIRE  = meanings.index("FIRE")
    ACTION_RIGHT = meanings.index("RIGHT")
    ACTION_LEFT  = meanings.index("LEFT")
    print("Controls: LEFT/RIGHT arrows = move, SPACE/ENTER = fire, P = pause, ESC = quit")

    # Set up Pygame display.
    pygame.init()
    # Must reset before render (Gymnasium OrderEnforcer).
    env.reset(seed=args.seed + 1)
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None. Ensure render_mode='rgb_array'.")

    h, w = frame.shape[:2]
    out_w, out_h = w * scale, h * scale
    screen = pygame.display.set_mode((out_w, out_h))
    pygame.display.set_caption("Breakout — Human Play + RAM Logging")
    clock = pygame.time.Clock()
    # Read held key state directly; do not rely on OS key-repeat events.
    pygame.event.set_grab(True)
    pygame.key.set_repeat(0)

    # CSV schema.
    ram_pre_cols  = [f"ram_pre_{i}"  for i in range(128)]
    ram_post_cols = [f"ram_post_{i}" for i in range(128)]
    fieldnames = [
        "run_ts", "episode", "step", "action", "reward",
        "done", "episode_return", "lives_pre", "lives_post",
    ] + ram_pre_cols + ram_post_cols

    def serve_ball():
        """Press FIRE twice to serve the ball after reset or life loss."""
        for _ in range(2):
            env.step(ACTION_FIRE)

    paused  = bool(args.start_paused)
    running = True

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, args.episodes + 1):
            env.reset(seed=args.seed + ep)

            if args.auto_serve:
                serve_ball()

            done     = False
            ep_ret   = 0.0
            step_idx = 0

            if args.start_paused:
                paused = True
                print("Paused — press SPACE/ENTER to start. P toggles pause. ESC quits.")

            while running and not done:
                # Process window and keyboard events.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_p:
                            paused = not paused
                        elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                            if paused:
                                paused = False
                                if args.auto_serve:
                                    serve_ball()

                if not running:
                    break

                # Render on every iteration, including while paused.
                frame = env.render()
                if frame is not None:
                    blit_scaled(screen, frame, out_w, out_h)

                if paused:
                    clock.tick(30)
                    continue

                # Sample key state as close to the env step as possible.
                decision_action = sample_key_action(ACTION_LEFT, ACTION_RIGHT, ACTION_FIRE, ACTION_NOOP)

                ram_pre   = get_ram(env)
                lives_pre = get_lives(env)

                block_reward = 0.0
                terminated   = False
                truncated    = False

                # Hold the decision action for action_repeat raw ALE frames.
                for _ in range(action_repeat):
                    _, r, term, trunc, _info = env.step(decision_action)
                    block_reward += float(r)
                    terminated = terminated or bool(term)
                    truncated  = truncated  or bool(trunc)

                    # Render every internal frame for smooth motion.
                    frame = env.render()
                    if frame is not None:
                        blit_scaled(screen, frame, out_w, out_h)

                    clock.tick(frame_fps)

                    if terminated or truncated:
                        break

                done = bool(terminated or truncated)

                # Auto-serve after life loss to match the agent's AutoFireWrapper behaviour.
                lives_mid = get_lives(env)
                if (
                    args.auto_serve
                    and not done
                    and lives_mid  is not None
                    and lives_pre  is not None
                    and lives_mid < lives_pre
                ):
                    serve_ball()

                ram_post   = get_ram(env)
                lives_post = get_lives(env)
                ep_ret    += block_reward

                row = {
                    "run_ts":         run_ts,
                    "episode":        ep,
                    "step":           step_idx,
                    "action":         int(decision_action),
                    "reward":         float(block_reward),
                    "done":           int(done),
                    "episode_return": float(ep_ret),
                    "lives_pre":      lives_pre  if lives_pre  is not None else "",
                    "lives_post":     lives_post if lives_post is not None else "",
                }
                for i in range(128):
                    row[f"ram_pre_{i}"]  = int(ram_pre[i])
                    row[f"ram_post_{i}"] = int(ram_post[i])

                writer.writerow(row)
                step_idx += 1

            if not running:
                break

            print(f"Episode {ep}: return={ep_ret:.2f}, decision_steps={step_idx}")

    env.close()
    pygame.quit()
    print("\nSaved human reference trace to:")
    print(" ", os.path.abspath(log_path))


if __name__ == "__main__":
    main()
