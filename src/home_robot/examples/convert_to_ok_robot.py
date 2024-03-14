# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle

import click


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(),
    default="input.pkl",
    help="Input path with default value 'input.pkl'",
)
@click.option(
    "--output-path",
    "-i",
    type=click.Path(),
    default="output.pth",
    help="Ouput path with default value 'output.pth'",
)
def main(
    input_path,
    output_path,
):
    with open(input_path, "rb") as ob:
        data = pickle.load(ob)
        rgb_width = data["rgb"][0].shape[0]
        rgb_height = data["rgb"][0].shape[1]
        camera_matrix = data["camera_K"][0]
        poses = data["base_poses"]

        test_dict = {
            "w": rgb_width,
            "h": rgb_height,
            "K": camera_matrix,
            "poses": poses,
        }

    # TODO: replace this with the correct JSON format
    with open(output_path, "wb") as of:
        pickle.dump(test_dict, of)


if __name__ == "__main__":
    main()
