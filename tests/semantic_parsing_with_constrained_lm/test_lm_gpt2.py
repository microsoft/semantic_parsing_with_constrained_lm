# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=redefined-outer-name
import asyncio
import json
import os

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from semantic_parsing_with_constrained_lm.lm_gpt2 import IncrementalGPT2, Seq2SeqGPT2
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer


@pytest.fixture(scope="module")
def tiny_gpt2_path(tmpdir_factory) -> str:
    path = str(tmpdir_factory.mktemp("tiny_gpt2"))
    model = GPT2LMHeadModel(
        GPT2Config(n_positions=32, n_ctx=32, n_embd=4, n_layer=2, n_head=2)
    )
    model.save_pretrained(path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(path)

    return path


def test_incremental_batching(tiny_gpt2_path: str) -> None:
    tokenizer = GPT2ClampTokenizer.from_pretrained(tiny_gpt2_path)
    tiny_gpt2_model = GPT2LMHeadModel.from_pretrained(tiny_gpt2_path)
    lm = IncrementalGPT2(tiny_gpt2_path, tiny_gpt2_model, tokenizer)

    async def inner():
        first, _ = await lm.execute([0, 1, 2])
        second, _ = await lm.execute([3, 4, 5])

        batched = await asyncio.gather(lm.execute([0, 1, 2]), lm.execute([3, 4, 5]))
        assert torch.allclose(first, batched[0][0])
        assert torch.allclose(second, batched[1][0])

    with torch.no_grad():
        asyncio.run(inner())


def test_incremental_hidden_state(tiny_gpt2_path: str) -> None:
    tokenizer = GPT2ClampTokenizer.from_pretrained(tiny_gpt2_path)
    tiny_gpt2_model = GPT2LMHeadModel.from_pretrained(tiny_gpt2_path)
    lm = IncrementalGPT2(tiny_gpt2_path, tiny_gpt2_model, tokenizer)

    async def inner():
        first, hs0 = await lm.execute([0])
        second, hs1 = await lm.execute([1], hs0)
        third, _ = await lm.execute([2], hs1)

        together, _ = await lm.execute([0, 1, 2])
        assert torch.allclose(torch.cat([first, second, third]), together)

    with torch.no_grad():
        asyncio.run(inner())


def test_seq2seq(tiny_gpt2_path: str) -> None:
    with open(os.path.join(tiny_gpt2_path, "seq2seq_settings.json"), "w") as f:
        json.dump(
            {
                "input_surround": {
                    "bos": [20490, 25],
                    "eos": [198],
                    "starts_with_space": True,
                },
                "output_surround": {
                    "bos": [34556, 25],
                    "eos": [198],
                    "starts_with_space": True,
                },
                "decoder_start_token_id": None,
            },
            f,
        )
    tokenizer = GPT2ClampTokenizer.from_pretrained(tiny_gpt2_path)
    tiny_gpt2_model = GPT2LMHeadModel.from_pretrained(tiny_gpt2_path)
    model = Seq2SeqGPT2(tiny_gpt2_path, tiny_gpt2_model, tokenizer)

    async def inner():
        seq2seq_logprobs, seq2seq_hs = await model.initial(
            model.encode_for_encoder("human"), model.decoder_bos_ids
        )
        assert seq2seq_hs is not None

        lm_logprobs, lm_hs = await model.incremental_model.execute(
            model.tokenizer.encode("Human: human\nComputer:")
        )
        assert lm_hs is not None
        assert torch.allclose(seq2seq_logprobs, lm_logprobs[-2:])

        seq2seq_logprobs, _ = await model.extend([10], seq2seq_hs)
        lm_logprobs, _ = await model.extend([10], lm_hs)
        assert torch.allclose(seq2seq_logprobs, lm_logprobs)

    asyncio.run(inner())
