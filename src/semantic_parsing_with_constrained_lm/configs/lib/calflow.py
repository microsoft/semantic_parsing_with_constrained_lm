# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
from typing import Callable, Dict, List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer

from semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.configs.lib.common import (
    PromptOrder,
    SeparateLM,
    make_semantic_parser,
)
from semantic_parsing_with_constrained_lm.datum import Datum
from semantic_parsing_with_constrained_lm.decoding.earley_partial_parse import (
    GrammarTokenizerInfo,
    UTF8EarleyPartialParse,
)
from semantic_parsing_with_constrained_lm.decoding.partial_parse import StartsWithSpacePartialParse
from semantic_parsing_with_constrained_lm.domains.calflow import (
    CalflowDatum,
    CalflowOutputLanguage,
    read_calflow_jsonl,
)
from semantic_parsing_with_constrained_lm.fewshot import PromptBuilder
from semantic_parsing_with_constrained_lm.lm import (
    AutoregressiveModel,
    ClientType,
    IncrementalLanguageModel,
)
from semantic_parsing_with_constrained_lm.model import (
    BeamSearchSemanticParser,
    DecodingSetup,
    ProblemFactory,
)

# This is a magic number computed by Chris, the origins of which are no
# longer exactly known. It should correspond to the maximum number of GPT-2
# tokens we expect any plan would take up, when written as Lispress.
#
# We truncate the number of training examples put inside the prompt so that
# we can add this many more tokens and still stay under the 2048 limit.
#
# When using canonical utterances, this number doesn't need to be as big
# since canonical utterances are not as long as Lispress. However, when
# using 20 training examples per prompt, we are never at risk of reaching
# the 2048 limit, so this point is moot.
MAX_STEPS_FOR_COMPLETION = 313


calflow_max_steps_fn_params: Dict[
    Tuple[CalflowOutputLanguage, ClientType], Tuple[float, float]
] = {
    # python semantic_parsing_with_constrained_lm/scripts/calflow_fit_max_steps.py \
    # --data-path \
    # semantic_parsing_with_constrained_lm/domains/calflow/data/train_300_stratified.jsonl \
    # --tokenizer facebook/bart-large --output-type canonicalUtterance \
    # --max-unreachable 3
    (CalflowOutputLanguage.Canonical, ClientType.BART): (8, 1.7233333333),
    # python semantic_parsing_with_constrained_lm/scripts/calflow_fit_max_steps.py \
    # --data-path \
    # semantic_parsing_with_constrained_lm/domains/calflow/data/train_300_stratified.jsonl \
    # --tokenizer gpt2-xl --output-type canonicalUtterance \
    # --max-unreachable 3
    (CalflowOutputLanguage.Canonical, ClientType.GPT3): (8, 1.7233333333),
    (CalflowOutputLanguage.Canonical, ClientType.SMGPT3): (8, 1.7233333333),
    # python semantic_parsing_with_constrained_lm/scripts/calflow_fit_max_steps.py \
    # --data-path \
    # semantic_parsing_with_constrained_lm/domains/calflow/data/train_300_stratified.jsonl \
    # --tokenizer facebook/bart-large --output-type lispress \
    # --max-unreachable 3
    (CalflowOutputLanguage.Lispress, ClientType.BART): (65, 7.084487179487172),
    # python semantic_parsing_with_constrained_lm/scripts/calflow_fit_max_steps.py \
    # --data-path \
    # semantic_parsing_with_constrained_lm/domains/calflow/data/train_300_stratified.jsonl \
    # --tokenizer gpt2-xl --output-type lispress --max-unreachable 3
    (CalflowOutputLanguage.Lispress, ClientType.GPT3): (65, 7.084487179487172),
    (CalflowOutputLanguage.Lispress, ClientType.SMGPT3): (65, 7.084487179487172),
}


def get_calflow_max_steps_fn(
    output_type: CalflowOutputLanguage,
    client_type: ClientType,
    tokenizer: PreTrainedTokenizer,
) -> Callable[[Datum], Optional[int]]:
    max_steps_intercept, max_steps_slope = calflow_max_steps_fn_params[
        output_type, client_type
    ]

    def fn(datum: Datum) -> Optional[int]:
        return min(
            int(
                len(tokenizer.tokenize(datum.natural)) * max_steps_slope
                + max_steps_intercept
            ),
            MAX_STEPS_FOR_COMPLETION,
        )

    return fn


def make_semantic_parser_for_calflow(
    train_data: List[CalflowDatum],
    lm: AutoregressiveModel,
    use_gpt3: bool,
    beam_size: int,
    output_type: CalflowOutputLanguage,
    client_type: ClientType,
    preprocessed_grammar: PreprocessedGrammar,
    constrained: bool,
    prompt_order: PromptOrder = PromptOrder.BestLast,
    # Settings for using autoregressive models in a few-shot in-context setting
    similarity_lm: Optional[IncrementalLanguageModel] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    num_examples_per_prompt: int = 20,
    problem_factory_builder: Optional[Callable[[DecodingSetup], ProblemFactory]] = None,
) -> BeamSearchSemanticParser:
    if constrained:
        grammar_tokenizer_info = GrammarTokenizerInfo.create(
            lm.tokenizer,
            preprocessed_grammar,
            output_type == CalflowOutputLanguage.Lispress,
        )
        # TODO: Refer to `lm` to decide whether to use UTF8EarleyPartialParse or a different variant
        partial_parse_builder = lambda datum: UTF8EarleyPartialParse.initial(
            grammar_tokenizer_info, datum.natural
        )
    else:
        # TODO: Only impose this if we are using a GPT-2-style tokenizer
        partial_parse = StartsWithSpacePartialParse(lm.tokenizer)
        partial_parse_builder = lambda _: partial_parse

    max_steps_fn = get_calflow_max_steps_fn(output_type, client_type, lm.tokenizer)

    return make_semantic_parser(
        train_data,
        lm,
        use_gpt3,
        MAX_STEPS_FOR_COMPLETION,
        beam_size,
        partial_parse_builder,
        max_steps_fn=max_steps_fn,
        prompt_order=prompt_order,
        similarity_method=SeparateLM(similarity_lm=similarity_lm),  # type: ignore
        prompt_builder=prompt_builder,
        num_examples_per_prompt=num_examples_per_prompt,
        problem_factory_builder=problem_factory_builder,
    )


cached_read_calflow_jsonl = functools.lru_cache(maxsize=None)(read_calflow_jsonl)
