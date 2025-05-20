## Task

Advanced RL Code Optimizer Assignment

```
## Advanced RL Take-Home Assignment: Reward-Guided Code Optimizer

## Objective

Design and evaluate a reinforcement learning agent that **rewrites or transforms Python code snippets** to optimize a specified objective, guided by a reward signal.

---

## Core Task

1. Define or use a small corpus of Python functions (3–10 simple functions is enough).
2. Build a transformation agent that proposes code modifications.
3. Define a **reward function** to evaluate outputs (e.g., runtime speed, brevity, or correctness).
4. Use a **reinforcement learning algorithm** (REINFORCE, PPO, or similar) to train the agent.

---

## Required Components

### ✅ Code Transformation Agent

- Can be a model (e.g., LLM), token-level mutator, or AST-based transformer.
- Must output valid Python code.

### 🎯 Reward Design (Required)

Implement **two types** of rewards:
1. **Heuristic-based** (e.g., token length, `timeit`, static analysis).
2. **Learned reward model** (optional but encouraged):
- Train a small classifier or regressor on pairs (A better than B).

### 🔁 RL Algorithm

Use REINFORCE, PPO, or a simplified variant. Focus on:
- Sample efficiency
- Stability of updates
- Use of baselines or advantage estimators

---

## Stretch Goals

- Add pass/fail test suite signals to the reward
- Apply self-improvement (generate better data over time)
- Evaluate generalization on unseen functions

---

## Dataset Options

- Create synthetic Python snippets (~5–10 lines each)
- Or use [CodeSearchNet](https://huggingface.co/datasets/code_search_net) (Python subset)
- Or write 3–5 common utility functions as your base corpus

---

## Tools

- Libraries: `gym`, `transformers`, `trl`, `peft`, `ast`, `timeit`, `skrl`, or others
- Optional: Use CodeLlama, GPT-2, or a local model for generation

---

## Deliverables

1. **Code** (notebook or repo) with:
    - Agent implementation
    - Reward functions
    - Training loop and logs
2. **Report (1–2 pages or final notebook section)** covering:
    - Reward design decisions
    - Learning curve or before/after examples
    - Observations on model behavior, reward hacking, or generalization

---

## Time Budget: ~6–8 hours

This assignment is open-ended. Focus on demonstrating depth in RL formulation, reward design, and structured experimentation.

You are not expected to productionize your agent — aim for thoughtful design, working prototypes, and insightful evaluation.
```