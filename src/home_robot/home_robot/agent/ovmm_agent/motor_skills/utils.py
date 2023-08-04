from habitat.core.spaces import ActionSpace
from typing import Tuple
from habitat_baselines.utils.common import get_num_actions

def find_action_range(
    action_space: ActionSpace, search_key: str
) -> Tuple[int, int]:
    """
    Returns the start and end indices of an action key in the action tensor. If
    the key is not found, a Value error will be thrown.
    """

    start_idx = 0
    found = False
    end_idx = get_num_actions(action_space[search_key])
    for k in action_space:
        if k == search_key:
            found = True
            break
        start_idx += get_num_actions(action_space[k])
    if not found:
        raise ValueError(f"Could not find {search_key} action in {action_space}")
    return start_idx, start_idx + end_idx