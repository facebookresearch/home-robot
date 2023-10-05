# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from home_robot_spot.spot_client import SpotClient


def main():
    # Create the environment object
    spot = SpotClient()
    spot.start()


if __name__ == "__main__":
    main()
