# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

from semantic_parsing_with_constrained_lm.datum import DatumSub, FullDatumSub
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.fewshot import (
    DataRetriever,
    GPT2TokenizerQuirks,
    PromptBuilder,
    ShuffleAndSample,
    TopKSimilar,
    TruncateTokenLength,
)
from semantic_parsing_with_constrained_lm.index.bm25_index import BM25Retriever
from semantic_parsing_with_constrained_lm.lm import (
    HS,
    AutoregressiveModel,
    IncrementalLanguageModel,
    Seq2SeqModel,
)
from semantic_parsing_with_constrained_lm.model import (
    BeamSearchSemanticParser,
    ConstrainedDecodingProblemFactory,
    DecodingSetup,
    FewShotLMDecodingSetup,
    IncrementalLMSimilarityFunction,
    ProblemFactory,
    Seq2SeqDecodingSetup,
)


class PromptOrder(Enum):
    # Shuffle the training examples inside the prompt.
    Shuffle = 0
    # Put the best (most similar to test) training example earliest in the prompt.
    BestFirst = 1
    # Put the best training example at the end of the prompt.
    BestLast = 2


class SimilarityMethod:
    pass


class DefaultLM(SimilarityMethod):
    pass


@dataclass
class SeparateLM(SimilarityMethod):
    similarity_lm: IncrementalLanguageModel


@dataclass
class BM25Indexer(SimilarityMethod):
    pass


def make_semantic_parser(
    train_data: Sequence[FullDatumSub],
    lm: AutoregressiveModel[HS],
    use_gpt3: bool,
    global_max_steps: int,
    beam_size: int,
    partial_parse_builder: Callable[[DatumSub], PartialParse],
    max_steps_fn: Optional[Callable[[DatumSub], Optional[int]]],
    prompt_order: PromptOrder = PromptOrder.Shuffle,
    # Settings for using autoregressive models in a few-shot in-context setting
    prompt_builder: Optional[PromptBuilder] = None,
    num_examples_per_prompt: int = 20,
    problem_factory_builder: Optional[
        Callable[[DecodingSetup[DatumSub, HS]], ProblemFactory[DatumSub, HS]]
    ] = None,
    similarity_method: SimilarityMethod = DefaultLM(),
) -> BeamSearchSemanticParser:
    decoding_setup: DecodingSetup[DatumSub, HS]
    if isinstance(lm, IncrementalLanguageModel):
        if prompt_builder is None:
            prompt_builder = PromptBuilder.for_demo(
                do_include_context=False, use_preamble=True
            )
        similarity_lm = (
            similarity_method.similarity_lm
            if isinstance(similarity_method, SeparateLM)
            else lm
        )

        if use_gpt3:
            if prompt_order == PromptOrder.Shuffle:
                train_retriever: DataRetriever[FullDatumSub, DatumSub] = (
                    BM25Retriever(train_data=train_data, top_k=num_examples_per_prompt)
                    if isinstance(similarity_method, BM25Indexer)
                    else TopKSimilar[FullDatumSub, DatumSub](
                        train_data=train_data,
                        scorer=IncrementalLMSimilarityFunction(similarity_lm),
                        k=num_examples_per_prompt,
                    )
                )
                train_selectors = [
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                    ),
                    ShuffleAndSample(
                        num_per_sample=num_examples_per_prompt,
                        random_seed=0,
                    ),
                ]
            elif prompt_order == PromptOrder.BestFirst:
                train_retriever: DataRetriever[FullDatumSub, DatumSub] = (
                    BM25Retriever(train_data=train_data, top_k=num_examples_per_prompt)
                    if isinstance(similarity_method, BM25Indexer)
                    else TopKSimilar[FullDatumSub, DatumSub](
                        train_data=train_data,
                        scorer=IncrementalLMSimilarityFunction(similarity_lm),
                        k=num_examples_per_prompt,
                    )
                )
                train_selectors = [
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                    ),
                ]
            elif prompt_order == PromptOrder.BestLast:
                train_retriever: DataRetriever[FullDatumSub, DatumSub] = (
                    BM25Retriever(
                        train_data=train_data,
                        top_k=num_examples_per_prompt,
                        best_first=False,
                    )
                    if isinstance(similarity_method, BM25Indexer)
                    else TopKSimilar[FullDatumSub, DatumSub](
                        train_data=train_data,
                        scorer=IncrementalLMSimilarityFunction(similarity_lm),
                        k=num_examples_per_prompt,
                        best_first=False,
                    )
                )
                train_selectors = [
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                        reverse=True,
                    ),
                ]
        else:
            train_retriever = BM25Retriever(
                train_data=train_data, top_k=num_examples_per_prompt
            )
            train_selectors = []

        decoding_setup = FewShotLMDecodingSetup[FullDatumSub, DatumSub, HS](
            # mypy complains that Callable[[FullDatumSub], PartialParse] is
            # expected here, not sure why
            partial_parse_builder=partial_parse_builder,
            train_data=train_data if use_gpt3 else [],
            train_retriever=train_retriever,
            train_selectors=train_selectors,
            prompt_builder=prompt_builder,
            incremental_lm=lm,
            tokenizer_quirks=GPT2TokenizerQuirks(lm.tokenizer),
        )
    elif isinstance(lm, Seq2SeqModel):
        decoding_setup = Seq2SeqDecodingSetup(
            partial_parse_builder=partial_parse_builder, seq2seq_model=lm
        )
    else:
        raise ValueError("Unsupported type for lm")

    problem_factory: ProblemFactory[DatumSub, HS]
    if problem_factory_builder is None:
        problem_factory = ConstrainedDecodingProblemFactory(
            autoregressive_model=lm,
            decoding_setup=decoding_setup,
            length_normalization=0.7,
            top_k=beam_size,
        )
    else:
        problem_factory = problem_factory_builder(decoding_setup)

    return BeamSearchSemanticParser(
        problem_factory=problem_factory,
        tokenizer=lm.tokenizer,
        beam_size=beam_size,
        max_steps_fn=max_steps_fn,
    )
