# Connect 4 AI Arena

Two separate academic projects exploring AI approaches to Connect 4: a full-stack system that generates its own training data and deploys trained models as live opponents, and a reinforcement learning investigation into whether self-play can improve on a strong supervised CNN. The two stages used different teams and different baseline models — see each stage's README for details on how (and whether) they connect.

## Stages

### [Stage 1 — Full-Stack Deployment](./stage-1-full-stack-deployment)
Generates a training dataset from scratch via Monte Carlo Tree Search (MCTS) self-play, trains a CNN and a Transformer to imitate strong play, and deploys both as live opponents through a Dockerized backend connected to an Anvil web front end, hosted on AWS Lightsail.

**Team:** Nikhil Kumar, Franco Salinas, Muzaffar Yezdan, Justin Yang — MSBA, McCombs School of Business

### [Stage 2 — Reinforcement Learning](./stage-2-reinforcement-learning)
A business-framed investigation comparing Policy Gradient (REINFORCE) and Double DQN self-play training against a supervised CNN baseline, evaluated head-to-head and against MCTS at four difficulty levels. Note: this stage fine-tunes a separate, more advanced CNN built for this project — not the exact model deployed in Stage 1 (see that stage's README for the full explanation).

**Team:** Bhagya Puppala, Franco Salinas, Frank Rong, Hank Liu — MSBA, McCombs School of Business (Optimization II, Project 3)

## Repository Structure

```
connect4-ai-arena/
├── stage-1-full-stack-deployment/
│   ├── notebooks/       # MCTS data generation, CNN training, Transformer training
│   ├── deployment/       # Docker + Anvil Uplink backend, trained models
│   └── docs/             # Project write-up and gameplay screenshots
└── stage-2-reinforcement-learning/
    ├── notebooks/        # Policy Gradient + DQN training and evaluation
    └── docs/             # Project write-up
```
