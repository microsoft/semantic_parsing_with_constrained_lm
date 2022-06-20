# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict

import numpy as np

from semantic_parsing_with_constrained_lm.earley.earley import Ambig, Leaf, Node
from semantic_parsing_with_constrained_lm.earley.fsa_builders import (
    re_alternative,
    re_atom,
    re_concat,
    re_sequence,
    re_utf8,
)
from semantic_parsing_with_constrained_lm.earley.grammar import DFADottedRule, FixedGrammar, Nonterm
from semantic_parsing_with_constrained_lm.earley.recognize import enumerate_sentences, is_grammatical, parse
from semantic_parsing_with_constrained_lm.util.keydefaultdict import KeyDefaultDict

NONTERMINAL: Dict[str, Nonterm] = KeyDefaultDict(Nonterm)
ROOT = NONTERMINAL["ROOT"]

A = NONTERMINAL["A"]
AA = NONTERMINAL["AA"]
AAA = NONTERMINAL["AAA"]
A_TO_A = DFADottedRule.from_rule(A, re_utf8("a"))
AA_TO_AA = DFADottedRule.from_rule(AA, re_utf8("aa"))
AAA_TO_A_AA = DFADottedRule.from_rule(AAA, re_sequence((A, AA)))
AAA_TO_AA_A = DFADottedRule.from_rule(AAA, re_sequence((AA, A)))
ROOT_TO_AAA_AA = DFADottedRule.from_rule(ROOT, re_sequence((AAA, AA)))
ROOT_TO_AA_AAA = DFADottedRule.from_rule(ROOT, re_sequence((AA, AAA)))
A_GRAMMAR = FixedGrammar[np.uint8, Any](
    root=ROOT,
    expansions={
        ROOT: {ROOT_TO_AAA_AA, ROOT_TO_AA_AAA},
        A: {A_TO_A},
        AA: {AA_TO_AA},
        AAA: {AAA_TO_A_AA, AAA_TO_AA_A},
    },
)

NAME = NONTERMINAL["name"]
ACTION = NONTERMINAL["action"]
HUGGED_RULE = DFADottedRule.from_rule(ACTION, re_utf8(" hugged "), alias="hug")
HIGH_FIVED_RULE = DFADottedRule.from_rule(ACTION, re_utf8(" high-fived "), alias="hi5")
ROOT_TO_NAME_ACTION_NAME = DFADottedRule.from_rule(
    ROOT, re_sequence((NAME, ACTION, NAME))
)
E: Nonterm = NONTERMINAL["empty"]
GRAMMAR_WITH_NAMES = FixedGrammar[np.uint8, Any](
    root=ROOT,
    expansions={
        ROOT: {ROOT_TO_NAME_ACTION_NAME},
        NAME: {
            DFADottedRule.from_rule(
                NAME, re_concat(re_utf8("J"), re_atom(NONTERMINAL["name_j"]))
            ),
            DFADottedRule.from_rule(NAME, re_utf8("Sam")),
        },
        NONTERMINAL["name_j"]: {
            DFADottedRule.from_rule(
                NONTERMINAL["name_j"],
                re_alternative(
                    re_concat(
                        re_utf8("am"), re_alternative(re_utf8("es"), re_utf8("ie"))
                    ),
                    re_concat(
                        re_utf8("o"),
                        re_alternative(re_utf8("hn"), re_utf8("se"), re_utf8("seph")),
                    ),
                ),
            ),
        },
        ACTION: {HUGGED_RULE, HIGH_FIVED_RULE},
    },
)
AB = NONTERMINAL["AB"]
BC = NONTERMINAL["BC"]
C = NONTERMINAL["C"]
R_TO_A_BC = DFADottedRule.from_rule(ROOT, re_sequence((A, BC)))
R_TO_AB_C = DFADottedRule.from_rule(ROOT, re_sequence((AB, C)))
AB_TO_AB = DFADottedRule.from_rule(AB, re_utf8("ab"))
BC_TO_BC = DFADottedRule.from_rule(BC, re_utf8("bc"))
C_TO_C = DFADottedRule.from_rule(C, re_utf8("c"))
AMBIG_GRAMMAR = FixedGrammar[np.uint8, Any](
    root=ROOT,
    expansions={
        ROOT: {R_TO_A_BC, R_TO_AB_C},
        A: {A_TO_A},
        AB: {AB_TO_AB},
        BC: {BC_TO_BC},
        C: {C_TO_C},
    },
)

UINT8: Dict[str, np.uint8] = KeyDefaultDict(lambda x: np.uint8(ord(x)))


def test_enumerate_sentences():
    sentences = list(enumerate_sentences(A_GRAMMAR))
    assert sentences == [[UINT8["a"]] * 5]


def test_is_grammatical():
    for sentence in enumerate_sentences(A_GRAMMAR):
        assert is_grammatical(sentence, A_GRAMMAR)
    assert not is_grammatical([UINT8["a"]], A_GRAMMAR)
    assert not is_grammatical([UINT8["a"]] * 4, A_GRAMMAR)
    assert is_grammatical([UINT8["a"]] * 5, A_GRAMMAR)


def test_ambiguous_parse_1():
    sentence = [UINT8["a"]] * 5
    aa_parse = Node(
        rule=AA_TO_AA, children=[Leaf(terminal=UINT8["a"]), Leaf(terminal=UINT8["a"])]
    )
    a_parse = Node(rule=A_TO_A, children=[Leaf(terminal=UINT8["a"])])
    aaa_parse = Ambig[np.uint8](
        children=[
            Node(rule=AAA_TO_A_AA, children=[a_parse, aa_parse]),
            Node(rule=AAA_TO_AA_A, children=[aa_parse, a_parse]),
        ]
    )
    expected_parse = Ambig[np.uint8](
        children=[
            Node(rule=ROOT_TO_AAA_AA, children=[aaa_parse, aa_parse]),
            Node(rule=ROOT_TO_AA_AAA, children=[aa_parse, aaa_parse]),
        ]
    )
    assert is_grammatical(sentence, A_GRAMMAR)
    p = parse(np.array(sentence), A_GRAMMAR)
    # converting to Tree b/c i'm too lazy to put positions in each of the leaves
    assert p.to_tree() == expected_parse.to_tree()


def test_ambiguous_parse_2():
    sentence = [UINT8[c] for c in ["a", "b", "c"]]
    expected_parse = Ambig[np.uint8](
        children=[
            Node(
                rule=R_TO_A_BC,
                children=[
                    Node(rule=A_TO_A, children=[Leaf(UINT8["a"])]),
                    Node(rule=BC_TO_BC, children=[Leaf(UINT8["b"]), Leaf(UINT8["c"])]),
                ],
            ),
            Node(
                rule=R_TO_AB_C,
                children=[
                    Node(rule=AB_TO_AB, children=[Leaf(UINT8["a"]), Leaf(UINT8["b"])]),
                    Node(rule=C_TO_C, children=[Leaf(UINT8["c"])]),
                ],
            ),
        ],
    )
    assert is_grammatical(sentence, AMBIG_GRAMMAR)
    p = parse(np.array(sentence), AMBIG_GRAMMAR)
    # converting to Tree b/c i'm too lazy to put positions in each of the leaves
    assert p.to_tree() == expected_parse.to_tree()


def test_grammar_with_names():
    names = ["James", "Jamie", "John", "Jose", "Joseph", "Sam"]
    for name1 in names:
        for action in (" hugged ", " high-fived "):
            for name2 in names:
                assert is_grammatical(
                    np.frombuffer(
                        (name1 + action + name2).encode("utf-8"), dtype=np.uint8
                    ),
                    GRAMMAR_WITH_NAMES,
                )

    wrong_names = ["Jo", "Samantha"]
    for name1 in wrong_names:
        for action in (" hugged ", " high-fived "):
            for name2 in names:
                assert not is_grammatical(
                    np.frombuffer(
                        (name1 + action + name2).encode("utf-8"), dtype=np.uint8
                    ),
                    GRAMMAR_WITH_NAMES,
                )
