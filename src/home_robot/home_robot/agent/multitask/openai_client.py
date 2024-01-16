from typing import List, Optional, Tuple
import copy

from openai import OpenAI


DEFAULT_OBJECTS = "fanta can, tennis ball, black head band, purple shampoo bottle, toothpaste, orange packaging, green hair cream jar, green detergent pack,  blue moisturizer, green plastic cover, storage container, blue hair oil bottle, blue pretzels pack, blue hair gel tube, red bottle, blue bottle,  wallet"

DEFAULT_LOCATIONS = "white table, chair, dustbin, gray bed"

PROMPT_INTRO = """
Convert a command into formatted text using some combination of the following two commands:

pick=obj
place=loc

where obj can be a name describing some common household object such that it can be detected by an open-vocabulary object detector, and loc can be some household location which can be detected in the same way.

"""

PROMPT_SPECIFICS = """
obj may be any of these, or something specified in the command: $OBJECTS

loc may be any of these, or something specified in the command: $LOCATIONS
"""

PROMPT_EXAMPLES = """
Example 1:
Command: "get rid of that dirty towel"
Returns:
pick=towel
place=basket

Example 2:
Command: "put the cup in the sink"
Returns:
pick=cup
place=sink

Example 3:
Command: "i need the yellow shampoo bottle, can you put it by the shower?"
Returns:
pick=yellow bottle
place=bathroom counter

Example 4:
Command: "i could really use a sugary drink, i'm going to go lie down"
Returns:
pick=fanta can
place=gray bed

Example 5:
Command: "put the apple and orange on the kitchen table."
Returns:
pick=apple
place=kitchen table
pick=orange
place=kitchen table

You will respond ONLY with the executable commands, i.e. the part following "Returns." Do not include the word Returns. Objects must be specific. The term on the left side of the equals sign must be either pick or place.
"""


class OpenaiClient:
    """Simple client for creating agents using an OpenAI API call."""

    def __init__(self, objects: Optional[str] = None, locations: Optional[str] = None, use_specific_objects: bool = True):
        self.use_specific_objects = use_specific_objects
        if objects is None:
            objects = DEFAULT_OBJECTS
        if locations is None:
            locations = DEFAULT_LOCATIONS
        self.objects = objects
        self.locations = locations
        if self.use_specific_objects:
            specifics = copy.copy(PROMPT_SPECIFICS)
            specifics = specifics.replace("$OBJECTS", self.objects)
            specifics = specifics.replace("$LOCATIONS", self.locations)
            self.prompt = PROMPT_INTRO + specifics + PROMPT_EXAMPLES
        else:
            self.prompt = PROMPT_INTRO + PROMPT_EXAMPLES
        self._openai = OpenAI()

    def parse(self, content: str) -> List[Tuple[str, str]]:
        """parse into list"""
        plan = []
        for command in content.split("\n"):
            action, target = command.split("=")
            plan.append((action, target))
        return plan
        
    def __call__(self, command: str, verbose: bool = False):
        # prompt = copy.copy(self.prompt)
        # prompt = prompt.replace("$COMMAND", command)
        if verbose:
            print(f"{self.prompt=}")
        completion = self._openai.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": command},
          ]
        )
        plan = self.parse(completion.choices[0].message.content)
        if verbose:
            print(completion.choices[0].message)
            print(f"{plan=}")
        return plan


if __name__ == '__main__':
    client = OpenaiClient()
    plan = client("this room is a mess, could you put away the dirty towel?", verbose=True)
    print("\n\n")
    print("OpenAI client returned this plan:", plan)
