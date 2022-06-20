# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from semantic_parsing_with_constrained_lm.earley.agenda import Agenda, MetaOps


@pytest.mark.parametrize("use_backpointers", [True, False])
def test_push_pop(use_backpointers: bool):
    a = Agenda(use_backpointers=use_backpointers)
    z = MetaOps.zero()
    assert a.push(3, z)
    assert a.push(5, z)
    # duplicate should be ignored
    assert not a.push(3, z)
    assert a.popped == []
    assert a.remaining == [3, 5]
    assert a.pop() == 3
    assert a.popped == [3]
    assert a.remaining == [5]
    # duplicate should be ignored
    assert not a.push(3, z)
    assert a.push(7, z)

    assert a.popped == [3]
    assert a.remaining == [5, 7]

    def it():
        while a:
            yield a.pop()

    assert list(it()) == [5, 7]
