# Connect 4 AI Arena

An AI-powered Connect 4 platform built in two stages: first a full-stack system that generates its own training data and deploys trained models as live opponents, then a reinforcement learning phase that improves on the strongest model from Stage 1 through self-play.

## Stages

### [Stage 1 — Full-Stack Deployment](./stage-1-full-stack-deployment)
Generates a training dataset from scratch via Monte Carlo Tree Search (MCTS) self-play, trains a CNN and a Transformer to imitate strong play, and deploys both as live opponents through a Dockerized backend connected to an Anvil web front end, hosted on AWS Lightsail.

**Team:** Nikhil Kumar, Franco Salinas, Muzaffar Yezdan, Justin Yang — MSBA, McCombs School of Business

### Stage 2 — Reinforcement Learning *(coming soon)*
Takes the best Stage 1 CNN as a starting policy and improves it further using self-play and policy gradient (REINFORCE) training.

## Repository Structure

```
connect4-ai-arena/
├── stage-1-full-stack-deployment/
│   ├── notebooks/       # MCTS data generation, CNN training, Transformer training
│   ├── deployment/       # Docker + Anvil Uplink backend, trained models
│   └── docs/             # Project write-up and gameplay screenshots
└── stage-2-reinforcement-learning/   # (in progress)
```
