# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class Unit:
    """
    Analogue of Scala Unit type that has a single instance UNIT. Can be used as type
    placeholder. Similar to None but can be used where None doesn't work.
    """

    def __repr__(self) -> str:
        return "Unit"


UNIT = Unit()
