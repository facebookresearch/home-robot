import click

from home_robot_hw.remote import StretchClient


@click.command()
def main():
    # Set up robot
    robot = StretchClient
    # TODO: step up place planner

    # Prompt for robot to grasp object
    # TODO

    # Execute place
    # TODO


if __name__ == "__main__":
    main()
