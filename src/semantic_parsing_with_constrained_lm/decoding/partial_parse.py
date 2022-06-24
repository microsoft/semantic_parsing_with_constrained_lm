# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer


class PartialParse(ABC):
    @abstractmethod
    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Returns possible ways to extend the current prefix.

        The Tensor is of type long and 1-dimensional, with no duplicate values,
        containing the IDs of the tokens that we could append.
        If it is None, then any token is allowed.
        The bool indicates whether we are allowed to terminate here.

        If ordered_ids and top_k are not None, this may optionally return only
        the first `top_k` token IDs from ordered_ids which comports with the
        grammar, instead of all such token IDs.
        """
        pass

    @abstractmethod
    def append(self, token: int) -> "PartialParse":
        """Return a new PartialParse created by appending this token."""
        pass


class NullPartialParse(PartialParse):
    """PartialParse which admits any sequence."""

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        return None, True

    def append(self, token: int) -> "PartialParse":
        return self


class StartsWithSpacePartialParse(PartialParse):
    def __init__(self, tokenizer: ClampTokenizer):
        valid_tokens = []
        for utf8_token, token_id in tokenizer.utf8_token_to_id_map.items():
            if utf8_token[0] == 32:
                valid_tokens.append(token_id)
        self.valid_tokens = torch.tensor(valid_tokens)

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        return self.valid_tokens, False

    def append(self, token: int) -> "PartialParse":
        return NullPartialParse()
