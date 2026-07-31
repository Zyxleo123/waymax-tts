"""Reinforcement-learning utilities for Waymax (SB3 PPO / SAC).

Modules:
    scenario_source : loads Waymax scenarios from failure-case JSONs or TFRecords.
    waymax_env      : a gymnasium.Env wrapping the Waymax PlanningAgentEnvironment.
    train_ppo       : SB3 PPO training entry point.
    eval_ppo        : runs a trained PPO policy over a scenario set and reports metrics.
    train_sac       : SB3 SAC training entry point.
    eval_sac        : runs a trained SAC policy over a scenario set and reports metrics.
"""
