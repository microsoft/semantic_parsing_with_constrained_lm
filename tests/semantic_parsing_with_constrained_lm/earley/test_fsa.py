# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union

from semantic_parsing_with_constrained_lm.earley.fsa import EPS, Alternation, Ranges, Sink, eps_closure


def test_only_epsilon():
    s1 = Sink()
    s2 = Sink()

    s = Alternation[str](is_final=False, next=(s1, s2))
    assert set(s.transition(EPS)) == {s1, s2}
    assert set(s.transition("hello")) == set()


def test_ranges():
    s1 = Sink()
    s2 = Sink()

    r = Ranges[Union[int, str]](
        is_final=False,
        bounds=(10, 20),
        values=(None, s1, s2),
        applicable=lambda x: isinstance(x, int),
    )
    assert set(r.transition(EPS)) == set()
    assert set(r.transition("hello")) == set()
    assert set(r.transition(0)) == set()
    assert set(r.transition(9)) == set()
    assert set(r.transition(10)) == {s1}
    assert set(r.transition(19)) == {s1}
    assert set(r.transition(20)) == {s2}
    assert set(r.transition(1000)) == {s2}


def test_eps_closure():
    s1 = Alternation(is_final=False, next=())
    s2 = Alternation(is_final=False, next=())
    s3 = Alternation(is_final=False, next=())
    s4 = Alternation(is_final=False, next=())
    sink = Sink(is_final=False)

    s1.next = (s1, s2)
    s2.next = (sink, s1)
    s3.next = (s3,)
    s4.next = (s1,)

    assert set(eps_closure({s1})) == {s1, s2, sink}
    assert set(eps_closure({s2})) == {s1, s2, sink}
    assert set(eps_closure({s3})) == {s3}
    assert set(eps_closure({s4})) == {s1, s2, s4, sink}
    assert set(eps_closure({s1, s2})) == {s1, s2, sink}
    assert set(eps_closure({s1, s3})) == {s1, s2, s3, sink}
    assert set(eps_closure({s1, s4})) == {s1, s2, s4, sink}
    assert set(eps_closure({s2, s4})) == {s1, s2, s4, sink}
