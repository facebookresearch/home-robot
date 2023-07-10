import argparse
import json
import os
import sys
from enum import IntEnum, auto
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

verbose = False


def compute_stats(aggregated_metrics: pd.DataFrame) -> dict:
    """Compute stage-wise successes and other useful metrics.

    This function takes an aggregated metrics DataFrame as input and computes various statistics related to stage-wise successes and other metrics.

    Args:
        aggregated_metrics (pd.DataFrame): The aggregated metrics DataFrame containing the metrics data.

    Returns:
        dict: A dictionary containing the computed statistics. The dictionary has the following keys:
            - 'episode_count': The count of episodes for the 'END.ovmm_place_success' metric.
            - 'does_want_terminate': The mean value of the 'END.does_want_terminate' metric.
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
        if ("phase_success" in k and "END" in k) or "task_success" in k:
            stats[k.replace("END.ovmm_", "")] = aggregated_metrics.loc[k]["mean"]

    stats["partial_success"] = aggregated_metrics.loc["partial_success"]["mean"]
    return stats


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


def read_single_json(json_filename: str) -> pd.DataFrame:
    """Read a single JSON file.

    This function reads a single JSON file and returns the data as a pandas DataFrame.

    Args:
        json_filename (str): The path to the JSON file.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the JSON file. Each row represents an episode, and the columns represent the metrics.
            If the JSON file does not exist, a warning is printed, and None is returned.
    """
    if not os.path.exists(json_filename):
        if verbose:
            print(f"Warning: File {json_filename} does not exist")
        return None

    episode_metrics = json.load(open(json_filename))
    episode_metrics = {e: episode_metrics[e] for e in list(episode_metrics.keys())}
    episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient="index")
    return episode_metrics_df


def get_metrics_from_jsons(folder_name: str, exp_name: str) -> pd.DataFrame:
    """Read the metrics DataFrame from JSON files.

    This function reads the metrics data from JSON files located in a specified folder and experiment name.
    The JSON files are expected to contain metrics for a contiguous set of episodes, and the range of episodes is determined by the folder name.
    The start index of the episode range is recorded in the 'start_idx' column of the returned DataFrame.

    Args:
        folder_name (str): The folder name containing the JSON files.
        exp_name (str): The experiment name.

    Returns:
        pd.DataFrame: The DataFrame containing the metrics data. Each row represents an episode, and the columns represent the metrics.
            The 'start_idx' column records the start index of the episode range.
            If no valid JSON files are found or an error occurs during reading, None is returned.
    """
    df = read_single_json(os.path.join(folder_name, exp_name, "episode_results.json"))
    if df is not None:
        df["start_idx"] = 0
        return df

    # collect stats for all episodes
    dfs = []
    for subfolder in os.listdir(os.path.join(folder_name, exp_name)):
        if not os.path.isdir(os.path.join(folder_name, exp_name, subfolder)):
            continue
        json_filename = os.path.join(
            folder_name, exp_name, subfolder, "episode_results.json"
        )
        episode_metrics_df = read_single_json(json_filename)
        if episode_metrics_df is not None:
            episode_metrics_df["start_idx"] = int(subfolder.split("_")[0])
            dfs.append(episode_metrics_df)

    if len(dfs) == 0:
        return None

    return pd.concat(dfs)


def get_summary(args: argparse.Namespace, exclude_substr: str = "viz"):
    """Compute summary statistics from episode metrics.

    This function computes summary statistics from episode metrics stored in JSON files. It aggregates the metrics, computes task success and partial success measures,
    and generates a summary DataFrame containing the computed statistics.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        exclude_substr (str, optional): Substring to exclude from experiment names. Defaults to 'viz'.

    Returns:
        None
    """
    results_dfs = {}

    if args.exp_name is not None:
        exp_names = [args.exp_name]
    else:
        exp_names = os.listdir(os.path.join(args.folder_name))

    for exp_name in exp_names:
        # Exclude the `exp_names` having `exclude_substr` in their names
        if exclude_substr in exp_name:
            continue

        if not os.path.isdir(os.path.join(args.folder_name, exp_name)):
            continue

        episode_metrics = get_metrics_from_jsons(args.folder_name, exp_name)
        if episode_metrics is None:
            continue

        # Get absolute start_idx's
        episode_ids = (
            episode_metrics.index.str.split("_").str[-1].astype(int)
            + episode_metrics["start_idx"]
        )
        # Convert episode_id to string
        episode_ids = episode_ids.astype(str)

        # The task is considered successful if the agent places the object without robot collisions
        task_success = (
            episode_metrics["END.robot_collisions.robot_scene_colls"] == 0
        ) * (episode_metrics["END.ovmm_place_success"] == 1)

        # Compute partial success measure
        partial_success = (
            episode_metrics["END.ovmm_find_object_phase_success"]
            + episode_metrics["END.ovmm_pick_object_phase_success"]
            + episode_metrics["END.ovmm_find_recep_phase_success"]
            + task_success
        ) / 4.0

        episode_metrics = episode_metrics.assign(
            episode_id=episode_ids,
            task_success=task_success,
            partial_success=partial_success,
        )
        aggregated_metrics = aggregate_metrics(episode_metrics)
        stats = compute_stats(aggregated_metrics)
        results_dfs[exp_name] = stats

    # Create DataFrame with exp_name as index
    results_df = pd.DataFrame.from_dict(results_dfs, orient="index")

    # Sort by column names and row names
    results_df = results_df.sort_index(axis=0).sort_index(axis=1)

    # Save results to CSV in the same folder
    results_df.to_csv(os.path.join(args.folder_name, "summary.csv"))
    results_df.to_csv(sys.stdout)


def main():
    """Main entry point of the program.

    Parses the command-line arguments, retrieves the folder_name and exp_name,
    and calls the get_summary function to compute the summary statistics from episode metrics.

    Returns:
        None
    """
    # Parse arguments to read folder_name and exp_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default="datadump/results/eval_hssd")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    get_summary(args)


if __name__ == "__main__":
    main()
