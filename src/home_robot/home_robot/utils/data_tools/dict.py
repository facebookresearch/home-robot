# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tools for using dicts
"""
from collections import abc


# Recursive d.update(u). Good for configs!
def update(d, u):
    """
    Recursively update a target dictionary (d) with the contents of an update dictionary (u),
    unlike Python's built-in dict.update() method, which only works at the top level, this updates nested dictionaries.
    This function is especially useful for working with configuration files or nested settings.

    Args:
        d (dict): The target dictionary to be updated.
        u (dict): The source dictionary from which to update d.
        Both d and u can contain nested dictionaries.

    Returns:
        The function returns the updated dictionary d, after incorporating all updates from u.

    Example:
        target_dict = {
            "a": 1,
            "b": {
                "c": 3,
                "d": 4
            }
        }

        update_dict = {
            "a": 2,
            "b": {
                "c": 30,
                "e": 50
            },
            "f": 6
        }
        result = update(target_dict, update_dict)
        print(result)
        # Output will be: {'a': 2, 'b': {'c': 30, 'd': 4, 'e': 50}, 'f': 6}
    """
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
