# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from typing import Callable, List, Optional

from semantic_parsing_with_constrained_lm.datum import DatumSub, FullDatumSub
from semantic_parsing_with_constrained_lm.fewshot import (
    GPT2TokenizerQuirks,
    PromptBuilder,
    ShuffleAndSampleChunks,
    TopKSimilar,
    TruncateTokenLength,
)
from semantic_parsing_with_constrained_lm.lm import (
    AutoregressiveModel,
    IncrementalLanguageModel,
    Seq2SeqModel,
)
from semantic_parsing_with_constrained_lm.model import (
    BeamSearchSemanticParser,
    DatumProblemFactory,
    FewShotLMProblemFactory,
    IncrementalLMSimilarityFunction,
    Seq2SeqProblemFactory,
)
from semantic_parsing_with_constrained_lm.search import PartialParse


class PromptOrder(Enum):
    # Shuffle the training examples inside the prompt.
    Shuffle = 0
    # Put the best (most similar to test) training example earliest in the prompt.
    BestFirst = 1
    # Put the best training example at the end of the prompt.
    BestLast = 2


def make_semantic_parser(
    train_data: List[FullDatumSub],
    lm: AutoregressiveModel,
    use_gpt3: bool,
    global_max_steps: int,
    beam_size: int,
    partial_parse_builder: Callable[[DatumSub], PartialParse],
    max_steps_fn: Callable[[DatumSub], Optional[int]],
    prompt_order: PromptOrder = PromptOrder.Shuffle,
    # Settings for using autoregressive models in a few-shot in-context setting
    similarity_lm: Optional[IncrementalLanguageModel] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    num_examples_per_prompt: int = 20,
) -> BeamSearchSemanticParser:

    problem_factory: DatumProblemFactory
    if isinstance(lm, IncrementalLanguageModel):
        if prompt_builder is None:
            prompt_builder = PromptBuilder.for_demo(
                do_include_context=False, use_preamble=True
            )
        if similarity_lm is None:
            similarity_lm = lm

        if use_gpt3:
            if prompt_order == PromptOrder.Shuffle:
                train_selectors = [
                    TopKSimilar(
                        IncrementalLMSimilarityFunction(similarity_lm),
                        k=num_examples_per_prompt,
                    ),
                    ShuffleAndSampleChunks(
                        num_samples=1,
                        num_per_sample=num_examples_per_prompt,
                        random_seed=0,
                    ),
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                    ),
                ]
            elif prompt_order == PromptOrder.BestFirst:
                train_selectors = [
                    TopKSimilar(
                        IncrementalLMSimilarityFunction(similarity_lm),
                        k=num_examples_per_prompt,
                    ),
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                    ),
                ]
            elif prompt_order == PromptOrder.BestLast:
                train_selectors = [
                    TopKSimilar(
                        IncrementalLMSimilarityFunction(similarity_lm),
                        k=num_examples_per_prompt,
                        best_first=False,
                    ),
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                        reverse=True,
                    ),
                ]
        else:
            train_selectors = []

        tokenizer_quirks = GPT2TokenizerQuirks(lm.tokenizer)
        problem_factory = FewShotLMProblemFactory(
            train_data=train_data if use_gpt3 else [],
            train_selectors=train_selectors,
            prompt_builder=prompt_builder,
            incremental_lm=lm,
            partial_parse_builder=partial_parse_builder,
            tokenizer_quirks=tokenizer_quirks,
            length_normalization=0.7,
            top_k=beam_size,
        )
        finalizer = lambda tokens: tokenizer_quirks.postprocess_result(
            lm.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        )
    elif isinstance(lm, Seq2SeqModel):
        problem_factory = Seq2SeqProblemFactory(
            seq2seq_model=lm,
            partial_parse_builder=partial_parse_builder,
            length_normalization=0.7,
            top_k=beam_size,
        )
        finalizer = lm.decode_output

    return BeamSearchSemanticParser(
        problem_factory=problem_factory,
        tokenizer=lm.tokenizer,
        finalizer=finalizer,
        beam_size=beam_size,
        max_steps_fn=max_steps_fn,
    )
