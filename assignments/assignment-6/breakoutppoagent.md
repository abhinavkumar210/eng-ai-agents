# Breakout PPO Agent

A Proximal Policy Optimization (PPO) agent that learns to play Atari Breakout using deep reinforcement learning.

## Setup

This project uses Python 3.10 with uv for dependency management.

```bash
# Sync dependencies
uv sync

# Run optimized training (recommended - faster GPU utilization)
uv run python agent_vectorized.py

# Or run original version
uv run python agent.py
```

## Training

The agent trains for 40,000 iterations using 8 parallel environments. Training progress is logged to TensorBoard.

### Monitoring Training

To view training metrics in real-time:

```bash
tensorboard --logdir=runs/breakout_ppo
```

Then open http://localhost:6006 in your browser to see:
- Average reward per episode for each environment
- Training progress over time

### Checkpoints

The agent automatically saves checkpoints when it achieves new maximum rewards:
- Saved in `runs/breakout_ppo/checkpoints/` directory
- Format: `actorcritic_{max_reward}.pt` (state dict) and `actorcritic_{max_reward}` (full model)

## Requirements

- Python 3.10
- CUDA-capable GPU (recommended for training speed)
- Dependencies managed via `pyproject.toml`