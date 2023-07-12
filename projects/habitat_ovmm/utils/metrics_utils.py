# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import pandas as pd


def aggregate_metrics(episode_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics for each episode.

    This function takes an episode metrics DataFrame as input and aggregates the metrics for each episode, computing mean, minimum, maximum, and count for each metric column.

    Args:
        episode_metrics_df (pd.DataFrame): The episode metrics DataFrame containing the metrics data.

    Returns:
        pd.DataFrame: The aggregated metrics DataFrame. The columns represent the metrics, and the rows represent the aggregated statistics (mean, min, max, count) for each metric.
    """
    # Drop the columns with string values
    episode_metrics_df = episode_metrics_df.drop(
        columns=["episode_id", "goal_name", "END.ovmm_place_object_phase_success"],
        errors="ignore",
    )

    # Compute aggregated metrics for each column, excluding NaN values, to get mean, min, max, and count
    aggregated_metrics = episode_metrics_df.agg(["mean", "min", "max", "count"], axis=0)
    return aggregated_metrics.T


def compute_stats(aggregated_metrics: pd.DataFrame) -> dict:
    """Compute stage-wise successes and other useful metrics.

    This function takes an aggregated metrics DataFrame as input and computes various statistics related to stage-wise successes and other metrics.

    Args:
        aggregated_metrics (pd.DataFrame): The aggregated metrics DataFrame containing the metrics data.

    Returns:
        dict: A dictionary containing the computed statistics. The dictionary has the following keys:
            - 'episode_count': The number of episodes over which metrics are calculated.
            - 'does_want_terminate': The fraction of times the agent calls terminate.
            - 'num_steps': The mean number of steps taken by the agent.
            - Other stage-wise success metrics: The mean value of stage-wise success metrics with 'END' in their names, after removing 'END.ovmm_' from the metric names.
            - 'partial_success': The mean value of the 'partial_success' metric.
    """
    stats = {}
    stats["episode_count"] = aggregated_metrics.loc["END.ovmm_place_success"]["count"]
    stats["does_want_terminate"] = aggregated_metrics.loc["END.does_want_terminate"][
        "mean"
    ]
    stats["num_steps"] = aggregated_metrics.loc["END.num_steps"]["mean"]

    # find indices in the DataFrame with stage success in their name and compute success rate
    for k in aggregated_metrics.index:
        if ("phase_success" in k and "END" in k) or "overall_success" in k:
            stats[k.replace("END.ovmm_", "")] = aggregated_metrics.loc[k]["mean"]

    stats["partial_success"] = aggregated_metrics.loc["partial_success"]["mean"]
    return stats


def get_stats_from_episode_metrics(
    episode_metrics: pd.DataFrame,
) -> Optional[dict]:  # noqa: C901
    """Compute summary statistics from episode metrics.

    This function computes summary statistics from episode metrics stored in a DataFrame. It aggregates the metrics, computes task success and partial success measures,
    and generates a summary DataFrame containing the computed statistics.

    Args:
        episode_metrics (pd.DataFrame): The episode metrics DataFrame containing the metrics data.

    Returns:
        Optional[dict]: A dictionary containing the computed statistics. The dictionary has the following keys:
            - 'episode_count': The count of episodes for the 'END.ovmm_place_success' metric.
            - 'does_want_terminate': The mean value of the 'END.does_want_terminate' metric.
            - Other stage-wise success metrics: The mean value of stage-wise success metrics with 'END' in their names, after removing 'END.ovmm_' from the metric names.
            - 'partial_success': The mean value of the 'partial_success' metric.
    """

    # Get absolute start_idx's
    episode_ids = (
        episode_metrics.index.str.split("_").str[-1].astype(int)
        + episode_metrics["start_idx"]
    )
    # Convert episode_id to string
    episode_ids = episode_ids.astype(str)

    # The task is considered successful if the agent places the object without robot collisions
    overall_success = (
        episode_metrics["END.robot_collisions.robot_scene_colls"] == 0
    ) * (episode_metrics["END.ovmm_place_success"] == 1)

    # Compute partial success measure
    partial_success = (
        episode_metrics["END.ovmm_find_object_phase_success"]
        + episode_metrics["END.ovmm_pick_object_phase_success"]
        + episode_metrics["END.ovmm_find_recep_phase_success"]
        + overall_success
    ) / 4.0

    episode_metrics = episode_metrics.assign(
        episode_id=episode_ids,
        overall_success=overall_success,
        partial_success=partial_success,
    )
    aggregated_metrics = aggregate_metrics(episode_metrics)
    stats = compute_stats(aggregated_metrics)
    return stats
