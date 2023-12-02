# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from enum import IntEnum, auto
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append("projects/habitat_ovmm/")
from utils.metrics_utils import get_stats_from_episode_metrics

verbose = False


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

    dfs = []

    if df is not None:
        df["start_idx"] = 0
        dfs.append(df)
    # collect stats for all episodes
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
        episode_metrics = episode_metrics.reset_index()
        episode_metrics.drop_duplicates(subset="index", inplace=True, keep="last")
        episode_metrics.set_index("index", inplace=True)
        # Now save the scene and episode ids of complete episodes to args.folder_name/completed_ids.txt
        episode_ids = episode_metrics.index.values.tolist()
        with open(
            os.path.join(args.folder_name, exp_name, "completed_episodes.txt"), "w"
        ) as f:
            f.write("\n".join(episode_ids))
        stats = get_stats_from_episode_metrics(episode_metrics)

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
    parser.add_argument(
        "--folder_name", type=str, default="datadump/results/eval_hssd_0710/"
    )
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    get_summary(args)


if __name__ == "__main__":
    main()
