# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains PartialParse implementations for use with Earley grammars.

Currently there are 2 implementations:
- EarleyPartialParse
  - Lazily builds a trie of all utterances allowed by the grammar,
    using `advance_only_nonterminals` and `advance_with_terminal`.
  - Takes advantage of `ordered_ids` and `top_k` to stop work once enough valid
    tokens have been found.
  - Regexes must cover contiguous spans from the input utterance.
- UTF8EarleyPartialParse
  - Like EarleyPartialParse, but uses UTF-8 encoded byte strings as the
    underlying.

TODO: Replace all uses of the above with UInt8EarleyPartialParse.
"""
import collections
import dataclasses
import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AnyStr,
    ClassVar,
    DefaultDict,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import torch
from cached_property import cached_property

from semantic_parsing_with_constrained_lm.earley.agenda import Item
from semantic_parsing_with_constrained_lm.earley.earley import EarleyChart
from semantic_parsing_with_constrained_lm.earley.grammar import Grammar
from semantic_parsing_with_constrained_lm.earley.input import Position
from semantic_parsing_with_constrained_lm.util.trie import Trie, TrieSetNode
from semantic_parsing_with_constrained_lm.scfg.char_grammar import Char
from semantic_parsing_with_constrained_lm.scfg.earley_grammar import CFTerminal, EarleyCFGrammar
from semantic_parsing_with_constrained_lm.scfg.parser.token import RegexToken
from semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer


@dataclass(frozen=True)
class CFPosition(Position[CFTerminal]):
    content: Tuple[CFTerminal, ...] = ()

    _prev: Optional["CFPosition"] = dataclasses.field(
        default=None, compare=False, repr=False
    )
    _last: Optional["CFTerminal"] = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def scan(self, terminal: CFTerminal) -> Iterable["CFPosition"]:
        if (
            isinstance(terminal, str)
            and self.content
            and isinstance(self.content[-1], str)
        ):
            return (
                CFPosition(
                    # We already checked self.content[-1] is str, but pyright is confused
                    self.content[:-1] + (cast(str, self.content[-1]) + terminal,),
                    _prev=self,
                    _last=terminal,
                ),
            )
        else:
            return (CFPosition(self.content + (terminal,), _prev=self, _last=terminal),)

    def is_final(self) -> bool:
        """Is this a final position in the input, i.e., are we allowed to end here?"""
        return True

    def __len__(self) -> int:
        """
        A measurement of how far along the position is in the input (e.g., the prefix length).
        May be used by some control strategies to process positions in left-to-right order.
        """
        return sum(len(x) if isinstance(x, str) else 1 for x in self.content)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CFPosition) and self.content == other.content

    def __hash__(self) -> int:
        return hash(self.content)


@dataclass(frozen=True)
class CFPositionWithCopyInfo:
    pos: Position[CFTerminal]
    copy_offsets: Optional[FrozenSet[int]] = None
    # TODO: Put this field in a separate class.
    partial_utf8_bytes: Optional[bytes] = None


@dataclass
class GrammarTokenizerInfo:
    grammar: Grammar[CFTerminal, Any]
    # TODO (richard): make 2d ndarray of chars?
    tokens: List[str]
    tokens_to_id: Dict[str, int]
    utf8_tokens: List[bytes]
    utf8_tokens_to_id: Dict[bytes, int]
    utf8_trie: Trie[int]  # should be Trie[byte] but Python does not have a `byte` type
    start_position: CFPosition = CFPosition()

    @staticmethod
    def create(
        tokenizer: ClampTokenizer,
        preprocessed_grammar: PreprocessedGrammar,
        for_plans: bool,
    ) -> "GrammarTokenizerInfo":
        utf8_token_map = tokenizer.id_to_utf8_token_map
        n = max(utf8_token_map.keys()) + 1
        assert set(utf8_token_map.keys()) == set(range(n))
        utf8_tokens: List[bytes] = [utf8_token_map[i] for i in range(n)]
        return GrammarTokenizerInfo.from_utf8_tokens_list(
            utf8_tokens, preprocessed_grammar, for_plans
        )

    @staticmethod
    def from_tokens_list(
        tokens: List[str], preprocessed_grammar: PreprocessedGrammar, for_plans: bool
    ) -> "GrammarTokenizerInfo":
        grammar = GrammarTokenizerInfo._preproc_to_grammar(
            preprocessed_grammar, for_plans
        )
        utf8_tokens = [token.encode("utf-8") for token in tokens]
        tokens_to_id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        utf8_trie: Trie[int] = Trie(utf8_tokens)
        utf8_tokens_to_id = {t: i for i, t in enumerate(utf8_tokens)}
        return GrammarTokenizerInfo(
            grammar, tokens, tokens_to_id, utf8_tokens, utf8_tokens_to_id, utf8_trie
        )

    @staticmethod
    def from_utf8_tokens_list(
        utf8_tokens: List[bytes],
        preprocessed_grammar: PreprocessedGrammar,
        for_plans: bool,
    ) -> "GrammarTokenizerInfo":
        grammar = GrammarTokenizerInfo._preproc_to_grammar(
            preprocessed_grammar, for_plans
        )
        tokens: List[str] = [
            s.decode("utf-8", errors="backslashreplace") for s in utf8_tokens
        ]
        tokens_to_id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        utf8_trie: Trie[int] = Trie(utf8_tokens)
        utf8_tokens_to_id = {t: i for i, t in enumerate(utf8_tokens)}
        return GrammarTokenizerInfo(
            grammar, tokens, tokens_to_id, utf8_tokens, utf8_tokens_to_id, utf8_trie
        )

    @staticmethod
    def _preproc_to_grammar(
        preprocessed_grammar: PreprocessedGrammar, for_plans: bool
    ) -> EarleyCFGrammar:
        rules = (
            preprocessed_grammar.all_plan_rules
            if for_plans
            else preprocessed_grammar.all_utterance_rules
        )
        return EarleyCFGrammar.from_preprocessed_rules(rules)


@dataclass
class GrammarNodeInfo(Generic[AnyStr]):
    chart: EarleyChart[CFTerminal, Any] = dataclasses.field(repr=False)
    start_position: Position[CFTerminal] = dataclasses.field(repr=False)
    input_utterance: AnyStr
    # Only used when input_utterance is bytes.
    # TODO: Put this field in a separate class?
    initial_copy_offsets: Optional[FrozenSet[int]] = None

    def __post_init__(self):
        if not isinstance(self.input_utterance, bytes):
            return
        initial_copy_offsets = set()
        for i, byte in enumerate(self.input_utterance):
            if byte & 0b10000000 == 0 or byte & 0b11000000 == 0b11000000:
                initial_copy_offsets.add(i)
        self.initial_copy_offsets = frozenset(initial_copy_offsets)


LGN = TypeVar("LGN", bound="LazyGrammarNodeBase")
AnyChar = TypeVar("AnyChar", Char, int)


@dataclass
class LazyGrammarNodeBase(Generic[AnyStr, AnyChar]):
    """A lazily constructed trie of all possible valid strings in the grammar."""

    info: GrammarNodeInfo[AnyStr]
    depth: int

    # Stores information about all leaves of the trie rooted at this node, in a flattened way.
    descendants: DefaultDict[
        Tuple[int, AnyStr],
        DefaultDict[
            Tuple[CFPositionWithCopyInfo, CFTerminal], List[Item[CFTerminal, Any]]
        ],
    ] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
    )
    # Stores all regex terminals that could occur next after this node,
    # and the Items that we need to advance with if we scan that regex.
    regexes: Optional[
        DefaultDict[
            Tuple[CFPositionWithCopyInfo, RegexToken], List[Item[CFTerminal, Any]]
        ]
    ] = None

    # Stores all terminals that are completed at this node, and the Items that
    # we need to advance with to determine what can come next according to the
    # grammar.
    finished_terminals: Optional[
        DefaultDict[
            Tuple[CFPositionWithCopyInfo, CFTerminal], List[Item[CFTerminal, Any]]
        ]
    ] = None

    @cached_property
    def _next_pos_set(self) -> Set[CFPositionWithCopyInfo]:
        if not self.finished_terminals:
            return set()

        result = set()
        for (pos_with_copy_info, terminal), items in self.finished_terminals.items():
            [next_pos] = self.info.chart.advance_with_terminal(
                pos_with_copy_info.pos, terminal, items
            )
            if isinstance(terminal, RegexToken):
                result.add(
                    CFPositionWithCopyInfo(
                        next_pos,
                        pos_with_copy_info.copy_offsets,
                        pos_with_copy_info.partial_utf8_bytes,
                    )
                )
            else:
                result.add(CFPositionWithCopyInfo(next_pos))
        return result

    def process_terminal(self, t: str) -> AnyStr:
        raise NotImplementedError

    @property
    def regex_children_type(self) -> Type["LazyGrammarNodeRegexChildrenBase"]:
        raise NotImplementedError

    @cached_property
    def children(self: LGN) -> Mapping[AnyChar, LGN]:
        """Used for advancing one character in the trie and reaching the next node."""
        next_depth = self.depth + 1

        # If we're at the end of a terminal, we need to advance further to determine new children.
        for next_pos in self._next_pos_set:
            for next_terminal, next_items in self.info.chart.advance_only_nonterminals(
                next_pos.pos
            ).items():
                if isinstance(next_terminal, str):
                    # TODO: Drop copy_offsets from next_pos?
                    self.descendants[self.depth, self.process_terminal(next_terminal)][
                        next_pos, next_terminal
                    ].extend(next_items)
                elif isinstance(next_terminal, RegexToken):
                    if self.regexes is None:
                        self.regexes = collections.defaultdict(list)
                    self.regexes[next_pos, next_terminal].extend(next_items)
                else:
                    raise ValueError(next_terminal)

        result: Dict[AnyChar, LGN] = {}
        # TODO: Do less work when len(self.descendants) is 1
        for (num_prev_chars, terminal), scan_infos in self.descendants.items():
            # print(self.depth, num_prev_chars, len(terminal), terminal)
            if len(terminal) == 0:
                continue
            next_node = result.setdefault(
                terminal[self.depth - num_prev_chars],
                type(self)(self.info, next_depth),
            )
            if num_prev_chars + len(terminal) == next_depth:
                if next_node.finished_terminals:
                    for to_scan, items in scan_infos.items():
                        assert to_scan not in next_node.finished_terminals
                        next_node.finished_terminals[to_scan] = items
                else:
                    next_node.finished_terminals = scan_infos
            else:
                assert (num_prev_chars, terminal) not in next_node.descendants
                next_node.descendants[num_prev_chars, terminal] = scan_infos

        if self.regexes:
            return self.regex_children_type(result, self.regexes, self.info, next_depth)
        else:
            return result

    @cached_property
    def can_end(self) -> bool:
        return any(
            self.info.chart.was_found(
                self.info.chart.grammar.root, self.info.start_position, pos.pos
            )
            for pos in self._next_pos_set
        )


@dataclass
class LazyGrammarNodeRegexChildrenBase(Mapping[AnyChar, LGN]):
    """Used for LazyGrammarNode.children when regexes are involved."""

    underlying: Dict[AnyChar, LGN]
    regexes: Dict[
        Tuple[CFPositionWithCopyInfo, RegexToken], List[Item[CFTerminal, Any]]
    ]

    info: GrammarNodeInfo
    next_depth: int

    _regex_processed: Set[AnyChar] = dataclasses.field(default_factory=set)

    def __getitem__(self, c: AnyChar) -> LGN:
        if c not in self._regex_processed:
            for (pos_with_copy_info, regex_token), chart_items in self.regexes.items():
                # Check that we're copying a contiguous span of the input utterance
                copy_check_indices: Iterable[int]
                if pos_with_copy_info.copy_offsets is None:
                    if self.info.initial_copy_offsets is None:
                        copy_check_indices = range(len(self.info.input_utterance))
                    else:
                        copy_check_indices = self.info.initial_copy_offsets
                else:
                    copy_check_indices = pos_with_copy_info.copy_offsets
                filtered_locs = frozenset(
                    i + 1
                    for i in copy_check_indices
                    if i < len(self.info.input_utterance)
                    and self.info.input_utterance[i] == c
                )
                if not filtered_locs:
                    continue

                new_position = self._check_against_regex(
                    regex_token, c, pos_with_copy_info, filtered_locs
                )
                if new_position is None:
                    continue

                node = self.underlying.setdefault(
                    c, self.lazy_grammar_node_type(self.info, self.next_depth)
                )
                if node.finished_terminals is None:
                    node.finished_terminals = collections.defaultdict(list)
                node.finished_terminals[new_position, regex_token].extend(chart_items)
            self._regex_processed.add(c)

        return self.underlying[c]

    @property
    def lazy_grammar_node_type(self) -> Type[LGN]:
        raise NotImplementedError

    def _check_against_regex(
        self,
        regex_token: RegexToken,
        c: AnyChar,
        pos_with_copy_info: CFPositionWithCopyInfo,
        filtered_locs: FrozenSet[int],
    ) -> Optional[CFPositionWithCopyInfo]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[AnyChar]:
        return iter(self.underlying)

    def __len__(self) -> int:
        return len(self.underlying)


class LazyGrammarNode(LazyGrammarNodeBase[str, Char]):
    """Trie where single Unicode characters are edges."""

    @property
    def regex_children_type(self) -> Type[LazyGrammarNodeRegexChildrenBase]:
        return LazyGrammarNodeRegexChildren

    def process_terminal(self, t: str) -> str:
        return t


class LazyGrammarNodeRegexChildren(
    LazyGrammarNodeRegexChildrenBase[Char, LazyGrammarNode]
):
    @property
    def lazy_grammar_node_type(self) -> Type[LazyGrammarNode]:
        return LazyGrammarNode

    def _check_against_regex(
        self,
        regex_token: RegexToken,
        c: Char,
        pos_with_copy_info: CFPositionWithCopyInfo,
        filtered_locs: FrozenSet[int],
    ) -> Optional[CFPositionWithCopyInfo]:
        if regex_token.compiled.match(c):
            return CFPositionWithCopyInfo(pos_with_copy_info.pos, filtered_locs)
        return None


class UTF8LazyGrammarNode(LazyGrammarNodeBase[bytes, int]):
    """Trie where bytes (from UTF-8-encoded terminals) are edges."""

    @property
    def regex_children_type(self) -> Type[LazyGrammarNodeRegexChildrenBase]:
        return UTF8LazyGrammarNodeRegexChildren

    def process_terminal(self, t: str) -> bytes:
        return t.encode("utf-8")


class UTF8LazyGrammarNodeRegexChildren(
    LazyGrammarNodeRegexChildrenBase[int, UTF8LazyGrammarNode]
):
    @property
    def lazy_grammar_node_type(self) -> Type[UTF8LazyGrammarNode]:
        return UTF8LazyGrammarNode

    def _check_against_regex(
        self,
        regex_token: RegexToken,
        c: int,
        pos_with_copy_info: CFPositionWithCopyInfo,
        filtered_locs: FrozenSet[int],
    ) -> Optional[CFPositionWithCopyInfo]:
        # Figure out what the character actually is
        # TODO: Forbid discontinuing a regex sequence in the middle of an unfinished character
        if pos_with_copy_info.partial_utf8_bytes is None:
            utf8_bytes = bytes([c])
        else:
            utf8_bytes = pos_with_copy_info.partial_utf8_bytes + bytes([c])
        try:
            unicode_char = utf8_bytes.decode("utf-8")
        except UnicodeDecodeError:
            unicode_char = None

        if unicode_char is not None:
            # If the Unicode character is complete, then check it against the regex.
            # TODO: Think about if there are ways to avoid dead ends by checking
            # bytes before they have completed a character?
            utf8_bytes = None
            if not regex_token.compiled.match(unicode_char):
                return None
        return CFPositionWithCopyInfo(pos_with_copy_info.pos, filtered_locs, utf8_bytes)


EPP = TypeVar("EPP", bound="EarleyPartialParseBase")


@dataclass
class EarleyPartialParseBase(Generic[AnyStr, LGN], PartialParse):
    info: GrammarTokenizerInfo
    grammar_node: LGN
    tokens: List[AnyStr]
    _next_node_cache: Dict[int, Optional[LGN]] = dataclasses.field(default_factory=dict)

    _lazy_grammar_node_type: ClassVar[Type[LazyGrammarNodeBase]]

    @classmethod
    def _process_input_utterance(cls, input_utterance: str) -> AnyStr:
        raise NotImplementedError

    @classmethod
    def _initial_node(cls, info: GrammarTokenizerInfo, input_utterance: str) -> LGN:
        chart = EarleyChart(info.grammar, use_backpointers=False)
        chart.seek(info.grammar.root, info.start_position)

        grammar_node: LGN = cls._lazy_grammar_node_type(
            info=GrammarNodeInfo(
                chart,
                info.start_position,
                cls._process_input_utterance(input_utterance),
            ),
            depth=0,
        )  # type: ignore

        start = CFPositionWithCopyInfo(info.start_position)
        for next_terminal, next_items in chart.advance_only_nonterminals(
            info.start_position
        ).items():
            if isinstance(next_terminal, str):
                grammar_node.descendants[
                    0, grammar_node.process_terminal(next_terminal)
                ][start, next_terminal].extend(next_items)
            elif isinstance(next_terminal, RegexToken):
                grammar_node.regexes = collections.defaultdict(list)
                grammar_node.regexes[
                    CFPositionWithCopyInfo(info.start_position), next_terminal
                ].extend(next_items)
            else:
                raise ValueError(next_terminal)
        return grammar_node

    @staticmethod
    @abstractmethod
    def initial(info: GrammarTokenizerInfo, input_utterance: str):
        raise NotImplementedError

    def allowed_next(
        self,
        ordered_ids: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        first_n_tokens_from_lm_to_check: int = 100,
        enable_fallback_from_grammar: bool = True,
    ) -> Tuple[Optional[torch.Tensor], bool]:

        assert ordered_ids is not None
        ordered_ids_list = ordered_ids.tolist()
        all_tokens = self.tokens
        vocab_size = len(all_tokens)
        grammar_node = self.grammar_node

        def token_id_is_valid(i: int) -> bool:
            if not 0 <= i < vocab_size:
                return False
            token_str = all_tokens[i]
            result = grammar_node
            valid_token = True
            for token_char in token_str:
                # Advance the grammar terminal trie
                # TODO: Skip forward multiple characters if possible
                result = result.children.get(token_char)
                if result is None:
                    valid_token = False
                    break

            self._next_node_cache[i] = result
            return valid_token

        def produce_valid_tokens() -> Iterator[int]:
            for i in itertools.islice(
                ordered_ids_list, first_n_tokens_from_lm_to_check
            ):
                if token_id_is_valid(i):
                    yield i

        def valid_tokens_from_grammar() -> Iterator[int]:
            """Intersects the grammar trie with the vocabulary trie to get valid next tokens."""

            def _traverse(
                current_grammar_node: LGN, current_trie_node: TrieSetNode, prefix: bytes
            ) -> Iterator[bytes]:
                if current_trie_node.is_terminal:
                    yield prefix
                grammar_children = current_grammar_node.children
                trie_children = current_trie_node.children
                for n in grammar_children.keys() & trie_children.keys():  # intersection
                    prefix_next = bytearray(prefix)
                    prefix_next.append(n)
                    yield from _traverse(
                        grammar_children[n], trie_children[n], bytes(prefix_next)
                    )

            return (
                self.info.utf8_tokens_to_id[x]
                for x in _traverse(grammar_node, self.info.utf8_trie.root, bytes())
            )

        # TODO: Special case where grammar_node.children has no elements
        # (i.e. tokens_list will be empty)
        tokens_list = list(itertools.islice(produce_valid_tokens(), top_k))
        k = top_k if top_k is not None else len(tokens_list)
        if len(tokens_list) < k and enable_fallback_from_grammar:
            # fallback to trie-based search for valid next tokens
            valid_tokens: Set[int] = set(valid_tokens_from_grammar())
            sorted_valid_tokens: List[int] = []
            for token in ordered_ids_list:
                if token in valid_tokens:
                    sorted_valid_tokens.append(token)
                if len(sorted_valid_tokens) >= k:
                    break
            tokens_list = sorted_valid_tokens

        return torch.tensor(tokens_list, dtype=torch.long), self.grammar_node.can_end

    def append(self: EPP, token: int) -> EPP:
        grammar_node = self._next_node_cache.get(token)
        if grammar_node is None:
            grammar_node = self.grammar_node
            if not 0 <= token < len(self.tokens):
                raise ValueError("token was not in the vocabulary")
            token_str = self.tokens[token]
            for char in token_str:
                grammar_node = grammar_node.children[char]
        return type(self)(self.info, grammar_node, self.tokens)


class EarleyPartialParse(EarleyPartialParseBase[str, LazyGrammarNode]):
    """This class is deprecated. Try to use UInt8EarleyPartialParse instead."""

    _lazy_grammar_node_type: ClassVar[Type[LazyGrammarNode]] = LazyGrammarNode

    @classmethod
    def _process_input_utterance(cls, input_utterance: str) -> str:
        return input_utterance

    @staticmethod
    def initial(
        info: GrammarTokenizerInfo, input_utterance: str
    ) -> "EarleyPartialParse":
        grammar_node = EarleyPartialParse._initial_node(info, input_utterance)
        return EarleyPartialParse(info, grammar_node, info.tokens)


class UTF8EarleyPartialParse(EarleyPartialParseBase[bytes, UTF8LazyGrammarNode]):
    """This class is deprecated. Try to use UInt8EarleyPartialParse instead."""

    _lazy_grammar_node_type: ClassVar[Type[UTF8LazyGrammarNode]] = UTF8LazyGrammarNode

    @classmethod
    def _process_input_utterance(cls, input_utterance: str) -> bytes:
        return input_utterance.encode("utf-8")

    @staticmethod
    def initial(
        info: GrammarTokenizerInfo, input_utterance: str
    ) -> "UTF8EarleyPartialParse":
        grammar_node = UTF8EarleyPartialParse._initial_node(info, input_utterance)
        return UTF8EarleyPartialParse(info, grammar_node, info.utf8_tokens)
