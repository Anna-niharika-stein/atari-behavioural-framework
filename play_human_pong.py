"""
play_human_pong.py
==================
Human gameplay interface for ALE/Pong-v5 with full trace logging.

OVERVIEW
--------
This script lets a human player play Pong through a Pygame window and logs
every step to a CSV file in the same format as the agent evaluation logs.
This makes the resulting human trace directly comparable to PPO and DQN agent
traces using the behavioural trace framework described in the thesis.

HOW IT WORKS
------------
1. The ALE/Pong-v5 environment is created with frameskip=1 and sticky actions
   disabled. This means the environment steps forward one raw ALE frame at a
   time, and every action issued is applied exactly as given (no randomised
   repetition of the previous action).

2. A Pygame window displays the game at a scaled resolution. The script reads
   the currently held keyboard keys on every frame rather than relying on OS
   key-repeat events, which gives more responsive paddle control.

3. The human player controls the right paddle using the arrow keys or WASD:
       UP / W      — move paddle up (mapped to RIGHT in ALE action space)
       DOWN / S    — move paddle down (mapped to LEFT in ALE action space)
       SPACE/ENTER — fire / start point
       P           — pause / unpause
       ESC         — quit early

4. Serving is handled automatically at the start of each episode and after
   each point, so the human never needs to press FIRE to start play. As a
   result, no FIRE actions appear in the human log — only NOOP, RIGHT, and LEFT
   (movement-level controls).

5. On every frame, before and after the environment step, the full 128-byte
   ALE RAM vector is read directly from the emulator. This RAM snapshot is
   stored in the CSV alongside the action, reward, and episode metadata.
   RAM is not used to control the paddle; it is logged purely as a compact
   side-channel state descriptor for the downstream behavioural analysis.

6. Because the human interface uses action_repeat=1 (one action per raw ALE
   frame) while the agents use action_repeat=4, the logged human traces are
   at a finer temporal resolution than the agent logs. This mismatch is
   handled analytically during preprocessing: human frames are aggregated
   into non-overlapping 4-frame windows before action-structure and
   state-visitation descriptors are computed.

OUTPUT
------
A timestamped CSV is written to ./logs/human_pong/ by default. Each row
corresponds to one raw ALE frame and contains:

    run_ts          — session timestamp (links rows to a recording session)
    episode         — episode number
    step            — frame index within the episode
    action          — integer action selected at this frame
    reward          — immediate reward after the action
    done            — 1 if the episode ended at this step, 0 otherwise
    episode_return  — running cumulative reward for this episode
    lives_pre       — lives before the step (retained for compatibility)
    lives_post      — lives after the step
    ram_pre_0..127  — 128-byte ALE RAM vector before the action
    ram_post_0..127 — 128-byte ALE RAM vector after the action

USAGE
-----
    # Run one episode (used in thesis analysis):
    # python play_human_pong.py

    # Run 1 episode, saving to a custom directory:
    # python play_human_pong.py --episodes 1 --out_dir ./data/human_pong

    # Run with slower paddle movement (try 2 or 3 if control feels too fast):
    # python play_human_pong.py --movement_every_n_frames 2

REQUIREMENTS
------------
    pip install "gymnasium[atari,accept-rom-license]" ale-py pygame numpy

NOTES
-----
- Tested with gymnasium>=0.29, ale-py>=0.9, pygame>=2.1, numpy>=1.24.
- The CSV schema matches the agent evaluation logs produced by the thesis
  pipeline, enabling direct comparison through the behavioural trace framework.
- To change the number of episodes, FPS, display scale, or other settings,
  edit the CONFIG dictionary below or pass command-line arguments (see USAGE).
"""

import ale_py  # noqa: F401 — registers ALE environments with Gymnasium

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

try:
    import pygame
except ImportError as exc:
    raise RuntimeError(
        "pygame is required for human gameplay. Install with: pip install pygame"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "env_id": "ALE/Pong-v5",
    # Number of episodes to record.
    "episodes": 1,
    # Output directory for the CSV log. A timestamped filename is auto-generated.
    "out_dir": "./logs/human_pong",
    # Override the auto-generated log path by setting this to a full file path.
    "log_path": None,

    # Display and timing.
    # 45 FPS is a good balance between responsiveness and display load.
    # Lower this (e.g. 30) if the window feels sluggish on your machine.
    "frame_fps": 45,
    # Scale factor for the Pygame window (raw ALE frames are 210x160 pixels).
    "display_scale": 3,
    # If True, the game starts paused — press SPACE/ENTER to begin.
    "start_paused": False,

    # Keep at 1 for responsive human play and accurate raw-frame logging.
    # Human traces are aggregated to 4-frame windows later during analysis.
    "action_repeat": 1,

    # Paddle speed smoothing.
    # Atari moves the paddle by a fixed amount each time a movement action is sent.
    # If the paddle feels too fast, send a movement action only every N frames
    # and NOOP in between. 1 = full speed, 2 = gentler, 3 = even slower.
    "movement_every_n_frames": 1,

    # Automatically press FIRE at the start of each episode to serve the ball.
    # No life-loss auto-serve is used for Pong (unlike Breakout).
    "auto_serve": True,
    "serve_presses": 2,

    "seed": 0,
    "max_episode_steps": 10_000,
    # Sticky actions disabled so logged actions match exactly what was selected.
    "sticky_actions": 0.0,
}


# ---------------------------------------------------------------------------
# Environment helpers
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


def make_env(
    env_id: str = "ALE/Pong-v5",
    render_mode: str = "rgb_array",
    max_episode_steps: int = 10_000,
    sticky_actions: float = 0.0,
):
    """
    Create the ALE Pong environment with the settings used in the thesis pipeline.

    frameskip=1     — step one raw ALE frame at a time; no skipping.
    full_action_space=False — use the reduced 6-action Pong space.
    repeat_action_probability=0.0 — sticky actions disabled.
    """
    env = gym.make(
        env_id,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=sticky_actions,
        full_action_space=False,
        render_mode=render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if not hasattr(env.action_space, "n"):
        raise RuntimeError(f"Expected discrete action space, got: {env.action_space}")

    return env


def blit_scaled(screen, rgb: np.ndarray, out_w: int, out_h: int):
    """Render a raw RGB frame to the Pygame window at the configured scale."""
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (out_w, out_h))
    screen.blit(surf, (0, 0))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Keyboard → action mapping
# ---------------------------------------------------------------------------

def safe_action_index(meanings, preferred_names, fallback="NOOP"):
    """
    Return the action index for the first name in preferred_names that exists
    in the environment's action meanings list, falling back to fallback.
    """
    if isinstance(preferred_names, str):
        preferred_names = [preferred_names]
    for name in preferred_names:
        if name in meanings:
            return meanings.index(name)
    if fallback in meanings:
        return meanings.index(fallback)
    return 0


def build_action_mapping(meanings):
    """
    Build the keyboard → action index map for Pong.

    ALE Pong uses RIGHT/LEFT rather than UP/DOWN for paddle directions.
    The visual keys UP/W and DOWN/S are mapped onto whichever directional
    actions are available in the environment's reduced action space.
    """
    return {
        "NOOP":      safe_action_index(meanings, "NOOP"),
        "FIRE":      safe_action_index(meanings, "FIRE"),
        # UP/W maps to RIGHT (paddle moves up on screen).
        "UP":        safe_action_index(meanings, ["UP", "RIGHT"]),
        # DOWN/S maps to LEFT (paddle moves down on screen).
        "DOWN":      safe_action_index(meanings, ["DOWN", "LEFT"]),
        "UPFIRE":    safe_action_index(meanings, ["UPFIRE",   "RIGHTFIRE", "UP",   "RIGHT"], fallback="FIRE"),
        "DOWNFIRE":  safe_action_index(meanings, ["DOWNFIRE", "LEFTFIRE",  "DOWN", "LEFT"],  fallback="FIRE"),
    }


def sample_key_action(action_map: dict, frame_idx: int, movement_every_n_frames: int = 1) -> int:
    """
    Read the currently held keys and return the corresponding action index.

    movement_every_n_frames controls paddle speed smoothing: a movement action
    is only sent on frames where (frame_idx % movement_every_n_frames == 0);
    NOOP is sent on all other frames. This prevents the paddle from feeling
    too fast at high frame rates without changing the frame cadence.
    """
    keys = pygame.key.get_pressed()

    fire = bool(keys[pygame.K_SPACE] or keys[pygame.K_RETURN])
    up   = bool(keys[pygame.K_UP]    or keys[pygame.K_w])
    down = bool(keys[pygame.K_DOWN]  or keys[pygame.K_s])

    # Ignore contradictory direction input to avoid jitter.
    if up and down:
        return action_map["FIRE"] if fire else action_map["NOOP"]

    if fire and not up and not down:
        return action_map["FIRE"]

    movement_every_n_frames = max(1, int(movement_every_n_frames))
    allow_movement = (frame_idx % movement_every_n_frames == 0)

    if not allow_movement:
        return action_map["NOOP"]

    if fire and up:   return action_map["UPFIRE"]
    if fire and down: return action_map["DOWNFIRE"]
    if up:            return action_map["UP"]
    if down:          return action_map["DOWN"]

    return action_map["NOOP"]


# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

def run_session(cfg: dict):
    """Run the full human play session and write the CSV log."""

    scale                  = max(1, int(cfg["display_scale"]))
    action_repeat          = max(1, int(cfg["action_repeat"]))
    frame_fps              = max(1, int(cfg["frame_fps"]))
    movement_every_n_frames = max(1, int(cfg["movement_every_n_frames"]))

    # Resolve output path.
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg["log_path"] is None:
        Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(cfg["out_dir"], f"human_pong_pixels_{run_ts}.csv")
    else:
        Path(os.path.dirname(cfg["log_path"]) or ".").mkdir(parents=True, exist_ok=True)
        log_path = cfg["log_path"]

    # Set up environment.
    env = make_env(
        env_id=cfg["env_id"],
        render_mode="rgb_array",
        max_episode_steps=cfg["max_episode_steps"],
        sticky_actions=cfg["sticky_actions"],
    )

    meanings = env.unwrapped.get_action_meanings()
    print("Action space   :", env.action_space)
    print("Action meanings:", meanings)
    action_map = build_action_mapping(meanings)
    print("Key mapping    :", action_map)
    print("Controls: UP/W = up, DOWN/S = down, SPACE/ENTER = fire, P = pause, ESC = quit")
    print(f"Paddle smoothing: movement sent every {movement_every_n_frames} frame(s).")

    # Set up Pygame display.
    pygame.init()
    env.reset(seed=cfg["seed"] + 1)
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None. Ensure render_mode='rgb_array'.")

    h, w = frame.shape[:2]
    out_w, out_h = w * scale, h * scale
    screen = pygame.display.set_mode((out_w, out_h))
    pygame.display.set_caption("Pong — Human Play + RAM Logging")
    clock = pygame.time.Clock()
    # Read held key state directly; do not rely on OS key-repeat events.
    pygame.key.set_repeat(0)

    # CSV schema.
    ram_pre_cols  = [f"ram_pre_{i}"  for i in range(128)]
    ram_post_cols = [f"ram_post_{i}" for i in range(128)]
    fieldnames = [
        "run_ts", "episode", "step", "action", "reward",
        "done", "episode_return", "lives_pre", "lives_post",
    ] + ram_pre_cols + ram_post_cols

    def serve_ball():
        """Press FIRE a small number of times to start the point."""
        for _ in range(int(cfg["serve_presses"])):
            env.step(action_map["FIRE"])

    paused  = bool(cfg["start_paused"])
    running = True

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, cfg["episodes"] + 1):
            env.reset(seed=cfg["seed"] + ep)

            if cfg["auto_serve"]:
                serve_ball()

            done    = False
            ep_ret  = 0.0
            step_idx = 0

            if cfg["start_paused"]:
                paused = True
                print("Paused — press SPACE/ENTER to start. P toggles pause. ESC quits.")

            while running and not done:
                # Process window and keyboard events.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_p:
                            paused = not paused
                        elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                            if paused:
                                paused = False
                                if cfg["auto_serve"]:
                                    serve_ball()

                if not running:
                    break

                if paused:
                    frame = env.render()
                    if frame is not None:
                        blit_scaled(screen, frame, out_w, out_h)
                    clock.tick(30)
                    continue

                # Sample key state as close to the env step as possible.
                decision_action = sample_key_action(action_map, step_idx, movement_every_n_frames)

                ram_pre   = get_ram(env)
                lives_pre = get_lives(env)

                block_reward = 0.0
                terminated   = False
                truncated    = False

                for _ in range(action_repeat):
                    _, r, term, trunc, _info = env.step(decision_action)
                    block_reward += float(r)
                    terminated = terminated or bool(term)
                    truncated  = truncated  or bool(trunc)

                    # Render once after each step to avoid extra display latency.
                    frame = env.render()
                    if frame is not None:
                        blit_scaled(screen, frame, out_w, out_h)

                    clock.tick(frame_fps)

                    if terminated or truncated:
                        break

                done = bool(terminated or truncated)

                ram_post   = get_ram(env)
                lives_post = get_lives(env)
                ep_ret    += block_reward

                row = {
                    "run_ts":          run_ts,
                    "episode":         ep,
                    "step":            step_idx,
                    "action":          int(decision_action),
                    "reward":          float(block_reward),
                    "done":            int(done),
                    "episode_return":  float(ep_ret),
                    "lives_pre":       lives_pre  if lives_pre  is not None else "",
                    "lives_post":      lives_post if lives_post is not None else "",
                }
                for i in range(128):
                    row[f"ram_pre_{i}"]  = int(ram_pre[i])
                    row[f"ram_post_{i}"] = int(ram_post[i])

                writer.writerow(row)
                step_idx += 1

            if not running:
                break

            print(f"Episode {ep}: return={ep_ret:.2f}, steps={step_idx}")

    env.close()
    pygame.quit()
    print("Saved human reference trace to:", os.path.abspath(log_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Pong gameplay interface with RAM trace logging.")
    parser.add_argument("--episodes",               type=int,   default=CONFIG["episodes"],               help="Number of episodes to record (default: %(default)s)")
    parser.add_argument("--out_dir",                type=str,   default=CONFIG["out_dir"],                help="Output directory for CSV log (default: %(default)s)")
    parser.add_argument("--frame_fps",              type=int,   default=CONFIG["frame_fps"],              help="Target display frame rate (default: %(default)s)")
    parser.add_argument("--display_scale",          type=int,   default=CONFIG["display_scale"],          help="Pygame window scale factor (default: %(default)s)")
    parser.add_argument("--movement_every_n_frames",type=int,   default=CONFIG["movement_every_n_frames"],help="Paddle speed smoothing: send movement every N frames (default: %(default)s)")
    parser.add_argument("--seed",                   type=int,   default=CONFIG["seed"],                   help="Base random seed (default: %(default)s)")
    parser.add_argument("--max_episode_steps",      type=int,   default=CONFIG["max_episode_steps"],      help="Maximum steps per episode (default: %(default)s)")
    parser.add_argument("--start_paused",           action="store_true",                                  help="Start each episode paused")
    args = parser.parse_args()

    # Merge CLI arguments into CONFIG.
    cfg = CONFIG.copy()
    cfg.update({k: v for k, v in vars(args).items() if v is not None})

    run_session(cfg)
