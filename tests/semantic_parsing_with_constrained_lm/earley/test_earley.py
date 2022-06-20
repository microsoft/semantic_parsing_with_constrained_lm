# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict

from lark import Tree

from semantic_parsing_with_constrained_lm.earley.earley import Ambig, Leaf, Node
from semantic_parsing_with_constrained_lm.earley.grammar import FixedGrammar, LinearDottedRule, Nonterm
from semantic_parsing_with_constrained_lm.earley.recognize import enumerate_sentences, is_grammatical, parse
from semantic_parsing_with_constrained_lm.util.keydefaultdict import KeyDefaultDict

NONTERMINAL: Dict[str, Nonterm] = KeyDefaultDict(Nonterm)
ROOT = NONTERMINAL["ROOT"]

A = NONTERMINAL["A"]
AA = NONTERMINAL["AA"]
AAA = NONTERMINAL["AAA"]
A_TO_A = LinearDottedRule.from_rule(A, ("a",))
AA_TO_AA = LinearDottedRule.from_rule(AA, ("a", "a"))
AAA_TO_A_AA = LinearDottedRule[str].from_rule(AAA, (A, AA))
AAA_TO_AA_A = LinearDottedRule[str].from_rule(AAA, (AA, A))
ROOT_TO_AAA_AA = LinearDottedRule[str].from_rule(ROOT, (AAA, AA))
ROOT_TO_AA_AAA = LinearDottedRule[str].from_rule(ROOT, (AA, AAA))
A_GRAMMAR = FixedGrammar[str, Any](
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
HUGGED_RULE = LinearDottedRule.from_rule(ACTION, tuple(" hugged "), alias="hug")
HIGH_FIVED_RULE = LinearDottedRule.from_rule(ACTION, tuple(" high-fived "), alias="hi5")
ROOT_TO_NAME_ACTION_NAME = LinearDottedRule[str].from_rule(ROOT, (NAME, ACTION, NAME))
E = NONTERMINAL["empty"]
GRAMMAR_WITH_NAMES = FixedGrammar[str, Any](
    root=ROOT,
    expansions={
        ROOT: {ROOT_TO_NAME_ACTION_NAME},
        NAME: {
            LinearDottedRule.from_rule(NAME, ("J", NONTERMINAL["name_j"])),
            LinearDottedRule.from_rule(NAME, tuple("Sam")),
        },
        NONTERMINAL["name_j"]: {
            LinearDottedRule.from_rule(
                NONTERMINAL["name_j"], ("a", "m", NONTERMINAL["name_jam"])
            ),
            LinearDottedRule.from_rule(
                NONTERMINAL["name_j"], ("o", NONTERMINAL["name_jo"])
            ),
        },
        NONTERMINAL["name_jam"]: {
            LinearDottedRule.from_rule(NONTERMINAL["name_jam"], ("e", "s")),
            LinearDottedRule.from_rule(NONTERMINAL["name_jam"], ("i", "e")),
        },
        NONTERMINAL["name_jo"]: {
            LinearDottedRule.from_rule(NONTERMINAL["name_jo"], ("h", "n")),
            LinearDottedRule.from_rule(
                NONTERMINAL["name_jo"], ("s", "e", NONTERMINAL["name_jose"])
            ),
        },
        NONTERMINAL["name_jose"]: {
            LinearDottedRule[str].from_rule(
                NONTERMINAL["name_jose"], (E, E), alias="heyo_empty_jose"
            ),
            LinearDottedRule.from_rule(NONTERMINAL["name_jose"], ("p", "h")),
        },
        E: {LinearDottedRule.from_rule(E, (), alias="empty_0")},
        ACTION: {HUGGED_RULE, HIGH_FIVED_RULE},
    },
)
AB = NONTERMINAL["AB"]
BC = NONTERMINAL["BC"]
C = NONTERMINAL["C"]
R_TO_A_BC = LinearDottedRule[str].from_rule(ROOT, (A, BC))
R_TO_AB_C = LinearDottedRule[str].from_rule(ROOT, (AB, C))
AB_TO_AB = LinearDottedRule.from_rule(AB, ("a", "b"))
BC_TO_BC = LinearDottedRule.from_rule(BC, ("b", "c"))
C_TO_C = LinearDottedRule.from_rule(C, ("c",))
AMBIG_GRAMMAR = FixedGrammar[str, Any](
    root=ROOT,
    expansions={
        ROOT: {R_TO_A_BC, R_TO_AB_C},
        A: {A_TO_A},
        AB: {AB_TO_AB},
        BC: {BC_TO_BC},
        C: {C_TO_C},
    },
)


def test_enumerate_sentences():
    sentences = list(enumerate_sentences(A_GRAMMAR))
    assert sentences == [["a"] * 5]


def test_is_grammatical():
    for sentence in enumerate_sentences(A_GRAMMAR):
        assert is_grammatical(sentence, A_GRAMMAR)
    assert not is_grammatical(["a"], A_GRAMMAR)
    assert not is_grammatical(["a"] * 4, A_GRAMMAR)
    assert is_grammatical(["a"] * 5, A_GRAMMAR)


def test_ambiguous_parse_1():
    sentence = ["a"] * 5
    aa_parse = Node(rule=AA_TO_AA, children=[Leaf(terminal="a"), Leaf(terminal="a")])
    a_parse = Node(rule=A_TO_A, children=[Leaf(terminal="a")])
    aaa_parse = Ambig[str](
        children=[
            Node(rule=AAA_TO_A_AA, children=[a_parse, aa_parse]),
            Node(rule=AAA_TO_AA_A, children=[aa_parse, a_parse]),
        ],
    )
    expected_parse = Ambig[str](
        children=[
            Node(rule=ROOT_TO_AAA_AA, children=[aaa_parse, aa_parse]),
            Node(rule=ROOT_TO_AA_AAA, children=[aa_parse, aaa_parse]),
        ]
    )
    assert is_grammatical(sentence, A_GRAMMAR)
    p = parse("".join(sentence), A_GRAMMAR)
    # converting to Tree b/c i'm too lazy to put positions in each of the leaves
    assert p.to_tree() == expected_parse.to_tree()


def test_ambiguous_parse_2():
    sentence = ["a", "b", "c"]
    expected_parse = Ambig[str](
        children=[
            Node(
                rule=R_TO_A_BC,
                children=[
                    Node(rule=A_TO_A, children=[Leaf("a")]),
                    Node(rule=BC_TO_BC, children=[Leaf("b"), Leaf("c")]),
                ],
            ),
            Node(
                rule=R_TO_AB_C,
                children=[
                    Node(rule=AB_TO_AB, children=[Leaf("a"), Leaf("b")]),
                    Node(rule=C_TO_C, children=[Leaf("c")]),
                ],
            ),
        ],
    )
    assert is_grammatical(sentence, AMBIG_GRAMMAR)
    p = parse("".join(sentence), AMBIG_GRAMMAR)
    # converting to Tree b/c i'm too lazy to put positions in each of the leaves
    assert p.to_tree() == expected_parse.to_tree()


def test_grammar_with_names():
    # import black
    hugged_parse = Tree("hug", [])
    e_parse = Tree("empty_0", [])
    jose_parse = Tree(
        "name",
        [
            Tree(
                "name_j",
                [Tree("name_jo", [Tree("heyo_empty_jose", [e_parse, e_parse])])],
            )
        ],
    )
    jose_hugged_jose = Tree("ROOT", [jose_parse, hugged_parse, jose_parse])
    names = ["Jose"]
    # names = ["James", "Jamie", "John", "Jose", "Joseph", "Sam"]
    for name1 in names:
        for action in [" hugged "]:  # (" hugged ", " high-fived "):
            for name2 in names:
                s = name1 + action + name2
                p = parse(s, GRAMMAR_WITH_NAMES)
                # print(black.format_str(str(p), mode=black.FileMode()))
                # print(black.format_str(str(p.to_tree()), mode=black.FileMode()))
                assert p.to_tree() == jose_hugged_jose
