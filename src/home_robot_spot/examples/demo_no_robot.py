# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information


class MockDemoAgent:
    def __init__(self):
        self.voxel_map = SparseVoxelMap()

    def say(self, msg: str):
        """Provide input either on the command line or via chat client"""
        if self.chat is not None:
            self.chat.output(msg)
        else:
            print(msg)

    def ask(self, msg: str) -> str:
        """Receive input from the user either via the command line or something else"""
        if self.chat is not None:
            return self.chat.input(msg)
        else:
            return input(msg)

    def confirm_plan(self, plan: str):
        print(f"Received plan: {plan}")
        if "confirm_plan" not in self.parameters or self.parameters["confirm_plan"]:
            execute = self.ask("Do you want to execute (replan otherwise)? (y/n): ")
            return execute[0].lower() == "y"
        else:
            if plan[:7] == "explore":
                print("Currently we do not explore! Explore more to start with!")
                return False
            return True

    def run(self):
        # Should load parameters from the yaml file
        self.voxel_map.read_from_pickle(input_path)
        world_representation = get_obj_centric_world_representation(
            voxel_map.get_instances(), args.context_length
        )
        # task is the prompt, save it
        data["prompt"] = self.get_language_task()
        output = stub.stream_act_on_observations(
            ProtoConverter.wrap_obs_iterator(
                episode_id=random.randint(1, 1000000),
                obs=world_representation,
                goal=data["prompt"],
            )
        )
        if confirm_plan(plan):
            # now it is hacky to get two instance ids TODO: make it more general for all actions
            # get pick instance id
            current_high_level_action = plan.split("; ")[0]
            pick_instance_id = int(
                world_representation.object_images[
                    int(
                        current_high_level_action.split("(")[1]
                        .split(")")[0]
                        .split(", ")[0]
                        .split("_")[1]
                    )
                ].crop_id
            )
            if len(plan.split(": ")) > 2:
                # get place instance id
                current_high_level_action = plan.split("; ")[2]
                place_instance_id = int(
                    world_representation.object_images[
                        int(
                            current_high_level_action.split("(")[1]
                            .split(")")[0]
                            .split(", ")[0]
                            .split("_")[1]
                        )
                    ].crop_id
                )
                print("place_instance_id", place_instance_id)
            break
