# Connect 4 AI Arena — Stage 2: Reinforcement Learning

A business-framed investigation into whether reinforcement learning can improve on a strong supervised Connect 4 CNN. Two RL approaches — Policy Gradient (REINFORCE) and Double DQN — were trained via self-play and evaluated against a Monte Carlo Tree Search (MCTS) opponent at four difficulty levels.

**Team:** Bhagya Puppala · Franco Salinas · Frank Rong · Hank Liu
*Team project completed as part of the Optimization II course, MSBA program, McCombs School of Business, UT Austin (Project 3).*

## Relationship to Stage 1

This stage is **not** a continuation of the exact model deployed in [Stage 1](../stage-1-full-stack-deployment). The Policy Gradient agent was fine-tuned starting from a separate, more advanced CNN (`connect4_cnn_final.h5`) that Franco built individually for this course's "Project 1" assignment — a multi-branch architecture trained on a much larger dataset (~545,000 positions from 70,000 self-play games) than the Stage 1 Anvil deployment CNN. That baseline CNN's own training notebook isn't included in this repo.

The Stage 1 CNN (`cnnmodel2.h5`) does make an appearance here, but only as one of five models in the **training opponent pool** — giving the PG agent a more diverse set of sparring partners during self-play, not as the model being improved.

## Executive Summary

Starting from a supervised CNN baseline (63.4% top-1 accuracy on unseen MCTS positions), this project trained two RL agents via self-play and compared them head-to-head and against MCTS at four difficulty levels (50/100/200/500 simulations).

**The headline finding: for Connect 4 specifically, supervised imitation of a strong MCTS teacher was already close to the ceiling REINFORCE fine-tuning could reach.** The Policy Gradient agent performed competitively with the supervised baseline but didn't clearly surpass it, and actually trailed it at moderate difficulty. The DQN agent, undertrained relative to plan, developed a structural bias toward edge columns and lost the large majority of head-to-head games against PG.

## Problem Setup — MDP Formulation

Connect 4 is framed as a finite-horizon Markov Decision Process:

| Component | Definition |
|---|---|
| State | 6×7×2 binary tensor — channel 0 is the current player's pieces, channel 1 is the opponent's (same encoding as Stage 1, eliminates sign ambiguity and lets one model play either color) |
| Action | Column index 0–6, with a legality mask excluding full columns |
| Reward | Sparse and terminal only: +1 win, −1 loss, 0 draw/non-terminal step |
| Transition dynamics | Deterministic given the board and both players' moves; all stochasticity comes from the opponent's policy |
| Discount factor (γ) | 0.99 for Policy Gradient, 0.95 for DQN |

## Starting Point: Baseline CNN

Both RL agents were initialized from or evaluated against a CNN identified as the strongest individually-built model across all four teammates' own "Project 1" submissions, selected via round-robin (17% win rate against the combined field of four models). This baseline achieved **63.4% top-1 accuracy** on unseen MCTS positions via supervised imitation learning — strong enough, as the results below show, that REINFORCE fine-tuning couldn't reliably improve on it.

## Policy Gradient (REINFORCE)

**Approach:** M1 (the CNN above) plays self-play games against a randomly sampled opponent from a fixed 5-model pool (which includes the Stage 1 CNN, another teammate's CNN, another teammate's deep CNN, and a DQN checkpoint). Each game produces (board, move, discounted return) triplets; a single full-batch gradient step per round updates M1 toward moves that led to wins, using normalized discounted returns as REINFORCE sample weights.

**Key training details:**

| Parameter | Value |
|---|---|
| Learning rate | 2×10⁻⁵ (Adam) |
| Discount factor (γ) | 0.99 |
| Games per round / total rounds | 60 games × 300 rounds = 18,000 games |
| Batch | All collected triplets per round (full-batch, not mini-batch) |
| Reward normalization | Zero-mean, unit-variance per batch |
| Random opening moves | 0–5 per game |
| Snapshot addition to opponent pool | Disabled — degraded performance in 3 of 5 pilot runs |
| Opponent's sampling temperature | τ = 2.0 (softened, so M1 sees a mix of wins and losses instead of losing ~90% of games at τ = 1.0) |

**Training result (verified directly from the notebook's saved output):** best win rate against MCTS-500 reached **68.0% at round 160**, with the final round finishing at 54.0%. The independent 256-game tournament evaluation reported in the write-up (not reproducible from this notebook's saved output, so presented as-reported) puts the submitted checkpoint's win rate vs. MCTS-500 at **56.2%**.

## Deep Q-Network (Double DQN)

**Approach:** trained from scratch with an ε-greedy policy over a replay buffer, using Double DQN's decoupled action-selection/action-evaluation to reduce overestimation bias.

| Parameter | Value |
|---|---|
| Architecture | Conv64→BN→Conv128→BN→Conv128→BN→Flatten→Dense256→Dense128→Q(7), tanh output |
| Total parameters | 1,634,247 (verified via `model.summary()`) |
| Discount factor (γ) | 0.95 |
| Replay buffer | 100,000 transitions, batch size 64 |
| Target network sync | Every 500 gradient steps |
| ε schedule | 1.0 → 0.05, exponential decay (factor 0.99995/episode) |
| Gradient clipping | clipnorm = 10.0 |

**⚠️ Episode count — flagging an unresolved discrepancy:** the project write-up's hyperparameter table states DQN was trained for **30,000 episodes**. However, this notebook's own saved training log shows output continuing through at least **episode 60,000** (`Ep 60000 | WinRate(1000): 0.601...`), and there appear to be multiple DQN checkpoints with different date stamps (`dqn_final_042526.h5`, `dqn_final_042726.h5`), suggesting more than one training run exists. It's not clear from what's in this repo which run's episode count corresponds to the final evaluated model. Both figures are noted here rather than picking one.

**Why DQN's training win rate doesn't match its MCTS evaluation:** the notebook logged a self-play win rate against its training opponent pool (3 CNNs + rotating DQN snapshots) between 29% and 66%, using a partially-random ε = 0.05–0.20 policy against opponents that themselves sample stochastically rather than always playing their best move. The final evaluation against MCTS uses a fully greedy policy (ε = 0) against a deterministic, near-optimal search opponent — a much harder and more honest test. The training metric measures *self-play improvement*; the MCTS evaluation measures *absolute strength*. DQN also developed a bias toward edge-column play that a stochastic CNN opponent doesn't punish, but that MCTS exploits immediately.

## Evaluation & Comparison

All results below are **as reported in the project write-up** (`docs/project_report.pdf`) — the underlying 256-game evaluation matrix was generated in a separate section of the notebook whose output cells weren't saved, so these numbers aren't independently re-derivable from what's in this repo.

**Win rate vs. MCTS, by difficulty (256 games each, 2 random opening moves):**

| Model | vs MCTS-50 | vs MCTS-100 | vs MCTS-200 | vs MCTS-500 |
|---|---|---|---|---|
| Baseline (supervised CNN) | 89.5% | 85.9% | 76.6% | 57.0% |
| PG (REINFORCE) | 88.3% | 80.9% | 68.0% | 56.2% |
| DQN (Double DQN) | 20.4% | 17.9% | 14.0% | 11.8% |

The supervised baseline matched or beat PG at every level, most clearly at moderate difficulty (MCTS-100: 85.9% vs 80.9%; MCTS-200: 76.6% vs 68.0%).

**Head-to-head, PG vs. DQN (256 games):** PG won 215 games (84%) to DQN's 41 (16%).

**Move frequency — the diagnostic behind DQN's collapse:** the baseline CNN plays the center column (col 3) 62.0% of the time; PG closely mirrors this at 55.1%, having apparently learned the same center-control heuristic purely from sparse win/loss self-play rewards, with no labeled data telling it to do so. DQN, by contrast, plays column 3 only 21.4% of the time and shows a pronounced bias toward column 6 (61.7%) — a training pathology that stochastic CNN opponents don't punish but that MCTS exploits immediately, and the most direct explanation for its weak MCTS performance.

**First vs. second player:** Connect 4 guarantees a first-player win under optimal play, and that edge shows up in the data at low-to-moderate MCTS difficulty (a 4–6 percentage point gap at MCTS-50/100), narrowing to roughly zero by MCTS-500.

## Why PG Outperformed DQN

- **Signal quality:** PG trains on full Monte Carlo returns over complete episodes — an unbiased estimate of long-run outcome. DQN bootstraps from its own Q-estimates, which takes substantially more episodes to converge reliably.
- **Compatible starting point:** PG fine-tuned the existing CNN's policy head directly. DQN replaced the output with Q-value regression from scratch, discarding the pretrained feature representations that gave the baseline its strength.
- **Compute budget:** DQN's training was not able to fully resolve its column-bias problem within the episodes available (see the episode-count discrepancy above).

## Business Recommendation

The core finding: for a game as constrained as Connect 4, where a strong supervised teacher (MCTS) is cheaply available, supervised imitation is close to the practical ceiling, and REINFORCE fine-tuning adds compute cost without a reliable return. RL becomes clearly worthwhile when no strong supervised teacher exists, when the game is too complex for imitation to generalize (Go, Chess, StarCraft), or when real-time adaptation to a specific opponent matters — none of which apply here. The report's recommendation: for games in Connect 4's complexity class, invest in supervised imitation; reserve dedicated RL investment for materially harder games.

## Repository Structure

```
stage-2-reinforcement-learning/
├── README.md
├── notebooks/
│   └── 01_pg_dqn_training_and_evaluation.ipynb
└── docs/
    └── project_report.pdf
```

No trained model weights (`.h5`/`.keras`) are included in this stage — the notebook and report are the available artifacts.

## Tech Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · Monte Carlo Tree Search (MCTS) as evaluation opponent

## Full Project Write-Up

The complete report — including the theoretical derivations for REINFORCE and Double DQN, full hyperparameter appendix, and all figures — is available at [`docs/project_report.pdf`](./docs/project_report.pdf).
