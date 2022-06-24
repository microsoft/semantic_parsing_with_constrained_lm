# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from semantic_parsing_with_constrained_lm.tokenization import (
    ClampTokenizer,
    GPT2ClampTokenizer,
    T5ClampTokenizer,
)


@pytest.mark.skip("relies on huggingface.co")
def test_tokenization():
    tokenizer: ClampTokenizer
    sents = ["   I love <>    dogs\t\n   ", "I love dogs", "{<>"]
    for tokenizer in [
        GPT2ClampTokenizer.from_pretrained("gpt2"),
        T5ClampTokenizer.from_pretrained(
            "google/t5-base-lm-adapt", output_sequences=sents
        ),
    ]:
        for sent in sents:
            assert tokenizer.detokenize(tokenizer.tokenize(sent)) == sent
            assert tokenizer.decode(tokenizer.encode(sent)) == sent
