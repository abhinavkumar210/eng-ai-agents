**Assignment 6 — DS681**

Date of submission: 12/14/2025
Tasks Completed: PPO and GRPO Agents in Atari Breakout

Task 1 — PPO Baseline Reproduction
Task 2 — GRPO Derivation and Implementation
Task 3 — PPO vs GRPO Experimental Comparison

The goal of this assignment is to extend an existing Proximal Policy Optimization (PPO) agent for the Atari Breakout environment and implement a variant known as Group Relative Policy Optimization (GRPO). GRPO is a simplified policy-gradient method that removes the value function and instead computes advantages using group-normalized trajectory returns. The assignment focuses on adapting GRPO—originally proposed for large-scale language-model training—to a classical reinforcement learning control task and empirically comparing its behavior to PPO under a controlled training budget.

------------------------------------------------------------------------------------------

**Environment:**
This assignment is based on the repository:
    - Breakout PPO Agent: https://github.com/pantelis/breakout-ppo-agent

The repository provides:
    - A working PPO implementation
    - Vectorized Atari environments
    - TensorBoard logging
    - Training and evaluation scripts

The repository contents were cloned and copied directly into the assignment-6 folder, and all modifications were made within this codebase.

------------------------------------------------------------------------------------------------

**Tasks Completed:**
Task 1 — PPO Baseline Reproduction

The first task was to reproduce a baseline PPO run using the provided implementation.

Method:
    - The original PPO training loop in agent_vectorized.py was examined to understand:
        - rollout collection
        - Generalized Advantage Estimation (GAE)
        - clipped policy loss
        - value loss and entropy regularization
    - The training budget was reduced to allow execution on CPU:
        - fewer iterations
        - shorter rollout length
        - smaller batch size
    - TensorBoard logging frequency was increased so learning curves could be observed during short runs.

Result:
    - PPO successfully ran end-to-end.
    - TensorBoard logs were generated in runs/breakout_ppo.
    - A figure showing mean episodic return vs environment steps was extracted and saved.

----------------------------------------------------

Task 2 — GRPO Derivation and Implementation
Task 2a — GRPO Theory

GRPO eliminates the learned value function used in PPO. Instead of computing per-timestep advantages using a critic, it computes group-relative advantages from complete trajectories.

For a group of G trajectories:
    - 𝑅𝑖 is the total return of trajectory 𝑖
    - 𝜇 is the mean return across the group
    - 𝜎 is the standard deviation of returns
    - The advantage is defined as: 𝐴𝑖 = (𝑅𝑖 − 𝜇)/𝜎

This single scalar advantage is assigned to every timestep of trajectory 𝑖 and used directly in the PPO clipped surrogate objective. The value loss term is removed entirely, while entropy regularization is retained.

Task 2b — GRPO Implementation

GRPO was implemented with minimal but conceptually meaningful changes.

Key Implementation Changes:
    - The critic head remains in the network but is not used during GRPO training.
    - No value prediction or value loss is computed in GRPO mode.
    - Training operates on groups of completed episodes.
    - Group-relative advantages are computed per episode and assigned to all steps.
    - PPO-style clipped policy loss is reused with the new advantages.
    - Entropy regularization is retained.
    - A lightweight GRPO-only training loop was implemented for correctness verification.

Verification:
    - GRPO runs correctly on CPU.
    - Group returns 𝑅𝑖, mean 𝜇, standard deviation 𝜎, and advantages 𝐴𝑖 are printed during training.
    - TensorBoard logs are generated in runs/breakout_grpo.

--------------------------------------------------

Task 3 — Experimental Evaluation

The final task compares PPO and GRPO under the same small training budget.

Method:
    - PPO and GRPO were run separately using short CPU-friendly budgets.
    - Both methods logged mean episodic return to TensorBoard.
    - Learning curves were extracted and plotted together.

Result:
    - PPO and GRPO exhibit different early-learning behaviors.
    - GRPO runs without a critic and demonstrates higher variance due to group-based normalization.
    - PPO is more stable early on, while GRPO shows sensitivity to episode grouping.
    - The comparison satisfies the assignment requirement for controlled empirical evaluation.

----------------------------------------------------------------------------------------------

**Files:**
Notebooks:
    - 01_ppo_baseline.ipynb
        Reproduces a short PPO baseline and extracts the learning curve.

    - 02_grpo_check.ipynb
        Implements and verifies GRPO, including group-relative advantages and critic-free optimization.

    - 03_compare_ppo_vs_grpo.ipynb
        Loads PPO and GRPO logs and produces a comparison plot.

Figures:
    - figures/ppo_curve.png — PPO baseline learning curve
    - figures/grpo_curve.png — GRPO learning curve
    - figures/ppo_vs_grpo.png — PPO vs GRPO comparison

Core Code:
    - agent_vectorized.py — Modified to support GRPO experiments
    - pyproject.toml — Environment and dependency specification

------------------------------------------------------------------------------------------------

**Libraries and Tools Used:**
    - Python 3.10
    - PyTorch
    - Gymnasium (Atari)
    - Stable-Baselines3 (wrappers only)
    - TensorBoard
    - NumPy
    - Jupyter Notebook (VS Code)

----------------------------------------------------------------------

**Summary**
This assignment demonstrates how GRPO can be adapted from language-model training to a classical reinforcement learning environment. By removing the critic and using group-normalized trajectory returns, GRPO offers a simpler optimization objective with different stability and variance properties compared to PPO. The experiments confirm correct implementation and provide a controlled comparison between the two methods.