# Copyright 2025 Valeo.


"""Module to collect and aggregate simulation metrics."""

from functools import partial

import numpy as np

from vmax.simulator.metrics import aggregators


_metrics_operands = {
    "ep_len_mean": aggregators.final,  # cumulative number of steps
    "ep_rew_mean": aggregators.final,  # cumulative reward at each timestep
    # Waymax metrics
    "log_divergence": np.mean,  # average L2 distance from expert log trajectory
    "reached_goal": np.max,  # 1.0 if SDC reached its goal at least once in the episode
    # Goal-reaching rate at a range of radii, rather than a single knife-edge threshold.
    "reached_goal_2m": np.max,
    "reached_goal_3m": np.max,
    "reached_goal_5m": np.max,
    # Distance-to-goal diagnostics: `min` is the closest the SDC ever got (did it arrive
    # at all?), `final` is where it ended up (did it arrive and stay?). A failure cluster
    # at 2-8 m means under-speed against a tight radius; one at 20 m+ means the policy is
    # stuck or over-cautious. The binary rate above cannot tell these apart.
    "distance_to_goal": {
        "min_distance_to_goal": np.min,
        "final_distance_to_goal": aggregators.final,
    },
    # Fraction of the initial gap to the goal still left at the end of the episode:
    # 0 means arrived, 1 means no progress towards the goal at all.
    "distance_to_goal_ratio": {
        "min_distance_to_goal_ratio": np.min,
        "final_distance_to_goal_ratio": aggregators.final,
    },
    "offroad": np.max,  # 1.0 if offroad then stop as termination
    "overlap": np.max,  # 1.0 if collision then stop as termination
    "sdc_off_route": np.mean,  # average distance to the closest on-route path
    "sdc_progression": aggregators.final,
    "progress_ratio_nuplan": {
        "progress_ratio": aggregators.final,
        "making_progress": partial(aggregators.final_within_bound, min_value=0.2),
    },
    "sdc_wrongway": np.mean,  # average distance for wrong way driving
    # V-Max metrics
    "run_red_light": np.max,  # 1.0 if red light run then terminate
    "ttc": {
        "min_ttc": np.min,
        "average_ttc": np.mean,
        "ttc_within_bound": partial(aggregators.all_within_bound, min_value=0.95),
    },
    "at_fault_collision": np.max,  # 1.0 if collision is at fault
    "comfort": np.min,  # comfort metric value (0 if thresholds violated)
    "speed_limit": {
        "max_overspeed_m_per_s": np.max,
        "max_overspeed_km_per_h": lambda x: 3.6 * np.max(x),
        "nuplan_speed_compliance": aggregators.nuplan_speed_compliance,
    },
    "on_multiple_lanes": {
        "distance_on_multiple_lanes": np.sum,
        "time_on_multiple_lanes": aggregators.time_spent,
        "multiple_lanes_score": aggregators.multiple_lanes_aggregator,
    },
    "driving_direction_compliance": {
        "distance_into_oncoming_traffic": np.sum,
        "nuplan_driving_direction_compliance": aggregators.nuplan_driving_direction_compliance,
    },
}


def get_termination_keys(env) -> tuple[str, ...]:
    """Return termination metric keys from the training env wrapper chain."""
    from vmax.simulator.wrappers.interfaces.brax import BraxWrapper

    current = env
    while current is not None:
        if isinstance(current, BraxWrapper):
            return tuple(current._termination_keys)
        current = getattr(current, "env", None)
    return ("offroad", "overlap", "run_red_light")


def check_episode_success(batch_metrics: dict, termination_keys: list[str] | tuple[str, ...]) -> float:
    """Return 1.0 if the episode had no termination failure, else 0.0."""
    is_episode_not_finished = 0.0
    for key in termination_keys:
        if key in batch_metrics:
            is_episode_not_finished += float(np.sum(batch_metrics[key]))

    return float(1.0 - (is_episode_not_finished > 0))


def _batch_metrics_for_episode(list_metrics: dict[str, list[np.ndarray]], episode_idx: int) -> dict:
    """Aggregate one episode's metrics using the same operands as full evaluation."""
    batch_metrics = {}
    for key, splits in list_metrics.items():
        if episode_idx >= len(splits) or len(splits[episode_idx]) == 0:
            continue

        l_metric = splits[episode_idx]
        metric_key = key.split("/")[-1]
        operand = _metrics_operands.get(metric_key, np.mean)

        if not isinstance(operand, dict):
            batch_metrics[metric_key] = operand(l_metric)
        else:
            for sub_key, sub_operand in operand.items():
                batch_metrics[sub_key] = sub_operand(l_metric)

    return batch_metrics


def collect(
    metrics: dict,
    key_metric: str,
    termination_keys: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Aggregate the episode metrics from the simulation.

    Args:
        metrics: The simulation metrics.
        key_metric: The key used to split episodes.
        termination_keys: If set, also compute mean per-episode ``accuracy``.

    Returns:
        Aggregated metrics per episode.

    """
    split_indices = _get_split_indices(metrics[key_metric])
    list_metrics = {key: np.split(metric, split_indices) for key, metric in metrics.items()}
    episode_metrics = {}

    for key, metric in metrics.items():
        _key = key.split("/")[-1]
        operand = _metrics_operands.get(_key, np.mean)

        list_metric = list_metrics[key]

        if not isinstance(operand, dict):
            episode_values = np.array([operand(l_metric) for l_metric in list_metric if len(l_metric) > 0])
            episode_metrics[key] = np.mean(episode_values)
        else:
            for sub_key, sub_operand in operand.items():
                episode_values = np.array([sub_operand(l_metric) for l_metric in list_metric if len(l_metric) > 0])
                episode_metrics[sub_key] = np.mean(episode_values)

    if termination_keys is not None:
        accuracies = []
        for i in range(len(list_metrics[key_metric])):
            batch_metrics = _batch_metrics_for_episode(list_metrics, i)
            if batch_metrics:
                accuracies.append(check_episode_success(batch_metrics, termination_keys))
        if accuracies:
            episode_metrics["accuracy"] = float(np.mean(accuracies))

    episode_metrics["nuplan_aggregate_score"] = aggregators.nuplan_aggregate_score(episode_metrics)
    episode_metrics["vmax_aggregate_score"] = aggregators.vmax_aggregate_score(episode_metrics)

    return episode_metrics


def _get_split_indices(steps: np.ndarray) -> np.ndarray:
    """Determine the indices at which to split the metric for each episode.

    Args:
        steps: An array representing the step metric over time.

    Returns:
        An array of indices for splitting episodes.

    """
    indices = np.argwhere(steps == 1) - 1
    indices = np.squeeze(indices) + 1

    return indices
