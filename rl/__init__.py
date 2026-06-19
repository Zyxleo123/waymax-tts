"""Reinforcement-learning utilities for Waymax (PPO via Stable-Baselines3).

Modules:
    scenario_source : loads Waymax scenarios from failure-case JSONs or TFRecords.
    waymax_env      : a gymnasium.Env wrapping the Waymax PlanningAgentEnvironment.
    train_ppo       : SB3 PPO training entry point.
    eval_ppo        : runs a trained policy over a scenario set and reports metrics.
"""
