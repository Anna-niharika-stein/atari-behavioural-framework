# Atari Behavioural Framework

A modular pipeline for collecting, aligning, and comparing human and reinforcement-learning agent gameplay in Atari game environments. Developed as part of a Master's thesis at Eindhoven University of Technology (Human-Centered AI), contributing to the [GAMECHAR project](https://cordis.europa.eu/project/id/101220528) (ERC, TU/e).

The framework goes beyond score-based evaluation by extracting three classes of behavioural descriptors from gameplay traces: **performance**, **action-structure**, and **RAM-based state visitation**. It is demonstrated on Breakout and Pong using PPO and DQN agents alongside a human reference player.

---

## Repository structure

```
atari-behavioural-framework/
│
├── Behavioural Analysis Pipelines/   # Jupyter notebooks for Breakout and Pong: descriptor extraction and analysis
│                                     # Input: logged CSV traces; Output: figures and tables
│
├── train_breakout_ppo.py             # Train PPO agent on Breakout from scratch
├── train_pong_ppo.py                 # Train PPO agent on Pong from scratch
│
├── eval_breakout_ppo.py              # Evaluate trained PPO on Breakout, log trace CSV
├── eval_breakout_dqn.py              # Evaluate pretrained DQN on Breakout, log trace CSV
├── eval_pong_ppo.py                  # Evaluate trained PPO on Pong, log trace CSV
├── eval_pong_dqn.py                  # Evaluate pretrained DQN on Pong, log trace CSV
│
├── play_human_breakout.py            # Pygame interface: human plays Breakout, logs trace CSV
└── play_human_pong.py                # Pygame interface: human plays Pong, logs trace CSV
```

---

## How it works

Every script (agent or human) writes a **shared step-level CSV** with the same schema:

| Column | Description |
|---|---|
| `run_ts` | Run timestamp / session identifier |
| `episode` | Episode number |
| `step` | Step index within episode |
| `action` | Action selected (encoded integer) |
| `reward` | Immediate reward |
| `done` | Terminal flag |
| `episode_return` | Cumulative reward within episode |
| `lives_pre` / `lives_post` | ALE lives before and after transition (Breakout) |
| `ram_pre_0` … `ram_pre_127` | Full 128-byte ALE RAM snapshot before action |
| `ram_post_0` … `ram_post_127` | Full 128-byte ALE RAM snapshot after action |

The **Behavioural Analysis Pipelines** notebooks consume these CSVs and produce all descriptors and figures reported in the thesis.

---

## Environment and action spaces

| Setting | Breakout | Pong |
|---|---|---|
| Environment | `ALE/Breakout-v5` | `ALE/Pong-v5` |
| Reduced action space | ✓ | ✓ |
| Action mapping | 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT | 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE |
| Sticky actions | Disabled (`repeat_action_probability=0.0`) | Disabled |
| Episode horizon | 10 000 agent steps | 10 000 agent steps |
| Frame skip | 4 (via `AtariPreprocessing`) | 4 (via `AtariPreprocessing`) |

---

## Requirements

```
python >= 3.9
gymnasium[atari]
ale-py
stable-baselines3
huggingface_sb3        # for downloading pretrained DQN checkpoints
pygame
numpy
pandas
scikit-learn
matplotlib
scipy
```

Install dependencies:

```bash
pip install gymnasium[atari] ale-py stable-baselines3 huggingface_sb3 pygame numpy pandas scikit-learn matplotlib scipy
```

---

## Usage

### 1 — Train PPO agents

```bash
python train_breakout_ppo.py
python train_pong_ppo.py
```

Both scripts train a `CnnPolicy` PPO agent for 10 million timesteps using 8 parallel environments and seed 0. Checkpoints are saved during training.

### 2 — Evaluate agents and log traces

**PPO (trained from scratch):**
```bash
python eval_breakout_ppo.py
python eval_pong_ppo.py
```

**DQN (pretrained from RL Baselines3 Zoo via Hugging Face):**
```bash
python eval_breakout_dqn.py
python eval_pong_dqn.py
```

Each script runs 30 deterministic episodes and writes a CSV trace file containing actions, rewards, terminal flags, and full RAM snapshots at every step.

### 3 — Collect human reference gameplay

```bash
python play_human_breakout.py
python play_human_pong.py
```

Launches a pygame window displaying the live game. Keyboard controls:

| Key | Action |
|---|---|
| ← | LEFT |
| → | RIGHT |
| No key | NOOP |

Serving is handled automatically — FIRE is never logged for the human player. Each session writes one CSV file per episode.

### 4 — Run behavioural analysis

Open the notebooks in `Behavioural Analysis Pipelines/` in order. They expect the CSV trace files produced by steps 2 and 3. Each notebook produces the descriptor tables and figures reported in the thesis.

---

## Behavioural descriptors

| Class | Descriptors |
|---|---|
| **Performance** | Episode return, episode length (raw ALE frames), reward density (reward per 1 000 frames), first-reward latency |
| **Action structure** | Action distribution, Shannon entropy, switching rate, mean run length, trigram and four-gram diversity, dominant motifs |
| **RAM-based state visitation** | Exact RAM-state uniqueness and Jaccard overlap, PCA projection, k-means cluster visitation shares (k=15), Jensen–Shannon divergence between cluster distributions |

Temporal alignment: human raw-frame input is aggregated into non-overlapping 4-frame decision windows (modal action) to match the agent decision cadence before descriptor extraction.

RAM clustering uses `MiniBatchKMeans` with `random_state=0` and `n_init=10`, ensuring all reported Jensen–Shannon distances are exactly reproducible from the same input CSVs.

---

## Reproducibility notes

- PPO training seed: `0`
- DQN source: [`sb3/dqn-BreakoutNoFrameskip-v4`](https://huggingface.co/sb3/dqn-BreakoutNoFrameskip-v4) and [`sb3/dqn-PongNoFrameskip-v4`](https://huggingface.co/sb3/dqn-PongNoFrameskip-v4) via Hugging Face
- Sticky actions disabled for interpretability (departs from Machado et al., 2018 recommendations — documented in thesis)
- Evaluation reward: raw/unclipped for all sources

---

## Related work

> Stein, A. (2026). *A Behavioural Framework for Comparing Human and RL Gameplay in Atari Game Environments*. Master's thesis, Eindhoven University of Technology.

---

## Acknowledgements

This work was developed within the [GAMECHAR project](https://cordis.europa.eu/project/id/101220528) — Scalable AI-driven Framework for Comprehensive Gameplay Characterization (ERC, TU/e). Supervisors: Max Birk, Chris Snijders, Peter Ruijten-Dodoiu.
