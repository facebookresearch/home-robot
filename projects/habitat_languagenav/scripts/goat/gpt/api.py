# You will be asked to generate Python code to follow navigation instructions
# following the API below. Don't nest if-else statements.


class LandmarkName:
    """
    This class represents the name of a landmark in the house, i.e., a static
    object that can be mapped at a specific location in the house.

    Examples:
        "the kitchen counter"
        "the coffee machine"
        "the sofa"
        "the dishwasher"
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class RoomName:
    """
    This class represents the name of a room in the house.

    Examples:
        "kitchen"
        "living room"
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class MovingObjectReferringExpression:
    """
    This class represents an expression that uniquely identifies a moving object in
    the house.

    Referring expressions can contain the name of the moving object, any of its
    properties, and relationships to landmark objects or rooms in the house.

    Examples:
        "the white mug next to the coffee machine"
        "the fruit basket on the kitchen counter"
        "my keys below a below on the sofa"
    """

    def __init__(self, referring_expression: str):
        self.referring_expression = referring_expression

    def __str__(self):
        return self.referring_expression


class HumanName(MovingObjectReferringExpression):
    """
    This class represents the name of a human in the house, which we treat as a
    specific type of moving object.

    Examples:
        "John"
        "Mary"
    """

    pass


class HouseMap:
    """
    This class is responsible for storing the map of the house.
    """

    def is_landmark_in_map(self, landmark_name: LandmarkName) -> bool:
        """Check if a landmark object is in the map.

        Arguments:
            landmark_name: The name of the landmark object.

        Returns:
            True if the landmark object is in the house map, False otherwise.
        """
        pass

    def is_room_in_map(self, room_name: RoomName) -> bool:
        """Check if a room is in the map.

        Arguments:
            room_name: The name of the room.

        Returns:
            True if the room is in the house map, False otherwise.
        """
        pass


class HouseNavigator:
    """
    This class is responsible for navigating a robot to different locations in the house.

    Attributes:
        house_map: The map of the house.
    """

    def __init__(self, house_map: HouseMap):
        self.house_map = house_map

    def go_to_landmark(self, landmark_name: LandmarkName) -> None:
        """Go to a landmark object in the house.

        This function assumes the landmark object is in the house map.

        Arguments:
            landmark_name: The name of the landmark object.

        Example:
        >>> # Go to the kitchen counter.
        >>> navigator = HouseNavigator(HouseMap())
        >>> landmark_name = LandmarkName("kitchen counter")
        >>> if navigator.house_map.is_landmark_in_map(landmark_name):
        >>>     navigator.go_to_landmark(landmark_name)
        >>> else:
        >>>     print(f"There is no {landmark_name} in the house.")
        """
        pass

    def go_to_room(self, room_name: RoomName) -> None:
        """Go to a room in the house.

        This function assumes the room is in the house map.

        Arguments:
            room_name: The name of the room.

        Example:
        >>> # Go to the kitchen.
        >>> navigator = HouseNavigator(HouseMap())
        >>> room_name = RoomName("kitchen")
        >>> if navigator.house_map.is_room_in_map(room_name):
        >>>     navigator.go_to_room(room_name)
        >>> else:
        >>>     print(f"There is no {room_name} in the house.")
        """
        pass

    def search_for_object_near_landmark(
        self, target: MovingObjectReferringExpression, landmark_name: LandmarkName
    ) -> bool:
        """Search for an object near another object.

        Arguments:
            target: The expression referring to the moving object to search for.
            landmark_name: The name of the object near which to search for the target object.

        Returns:
            True if the moving object was found, False otherwise.

        Example 1:
        >>> # Can you find the white mug next to the coffee machine?
        >>> navigator = HouseNavigator(HouseMap())
        >>> target = MovingObjectReferringExpression("the white mug next to the coffee machine")
        >>> landmark_name = LandmarkName("coffee machine")
        >>> if navigator.house_map.is_landmark_in_map(landmark_name):
        >>>     navigator.search_for_object_near_landmark(target, landmark_name)
        >>> else:
        >>>     print(f"There is no {landmark_name} in the house.")
        """
        pass

    def search_for_object_in_room(
        self, target: MovingObjectReferringExpression, room_name: RoomName
    ) -> bool:
        """Search for an object in a room.

        Arguments:
            target: The expression referring to the moving object to search for.
            room_name: The name of the room in which to search for the target object.

        Returns:
            True if the object was found, False otherwise.

        Example:
        >>> # Can you find the fruit basket in the kitchen?
        >>> navigator = HouseNavigator(HouseMap())
        >>> target = MovingObjectReferringExpression("the fruit basket in the kitchen")
        >>> room_name = RoomName("kitchen")
        >>> if navigator.house_map.is_room_in_map(room_name):
        >>>     navigator.search_for_object_in_room(target, room_name)
        >>> else:
        >>>     print(f"There is no {room_name} in the house.")

        Example 2:
        >>> # Can you find Mary? She's probably in the living room.
        >>> navigator = HouseNavigator(HouseMap())
        >>> target = HumanName("Mary")
        >>> room_name = RoomName("living room")
        >>> if navigator.house_map.is_room_in_map(room_name):
        >>>     navigator.search_for_object_in_room(target, room_name)
        >>> else:
        >>>     print(f"There is no {room_name} in the house.")
        """
        pass

    def follow_human(self, human_name: HumanName):
        """Follow a human in the house.

        This method should be called when the robot is already in the same room as the human.

        Arguments:
            human_name: The name of the human to follow.

        Example 1:
        >>> # Can you follow Mary?
        >>> navigator = HouseNavigator(HouseMap())
        >>> human_name = HumanName("Mary")
        >>> navigator.follow_human(human_name)

        Example 2:
        >>> # Find and follow John, he is either on the sofa or in the bedroom.
        >>> navigator = HouseNavigator(HouseMap())
        >>> human_name = HumanName("John")
        >>> landmark_name = LandmarkName("sofa")
        >>> room_name = RoomName("bedroom")
        >>> if (navigator.house_map.is_landmark_in_map(landmark_name) and
        >>>         navigator.search_for_object_near_landmark(human_name, landmark_name)):
        >>>     navigator.follow_human(human_name)
        >>> elif (navigator.house_map.is_room_in_map(room_name) and
        >>>         navigator.search_for_object_in_room(human_name, room_name)):
        >>>     navigator.follow_human(human_name)
        >>> else:
        >>>     print(f"Could not find {human_name} on the sofa or in the bedroom.")
        """
        pass


# Generate Python code for the following command:
