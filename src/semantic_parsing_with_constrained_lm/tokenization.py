# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import sentencepiece.sentencepiece_model_pb2 as sentencepiece_model
from cached_property import cached_property
from transformers import GPT2Tokenizer, T5Tokenizer

if TYPE_CHECKING:
    # pylint: disable=reimported
    from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
    from transformers.models.t5.tokenization_t5 import T5Tokenizer


@dataclass  # type: ignore
class ClampTokenizer(ABC):
    """
    Tokenizer interface to use for Clamp experiments. Any class implementing this interface should respect all
    whitespaces while tokenizing text. This interface only works for tokenizers using tokens aligned to
    UTF-8-encoded byte boundaries.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[bytes]:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def utf8_token_to_id_map(self) -> Dict[bytes, int]:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        pass

    @abstractmethod
    def save_pretrained(self, tokenizer_loc: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, tokenizer_loc: str) -> "ClampTokenizer":
        pass

    def detokenize(self, tokens: List[bytes]) -> str:
        full_bytes = b"".join(tokens)
        try:
            return full_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return "<Undecodable_UTF-8_string>"

    @cached_property
    def id_to_utf8_token_map(self) -> Dict[int, bytes]:
        return {v: k for k, v in self.utf8_token_to_id_map.items()}

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.utf8_token_to_id_map[token] for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [
            self.id_to_utf8_token_map[token_id]
            for token_id in token_ids
            if token_id in self.id_to_utf8_token_map
        ]
        return self.detokenize(tokens)


@dataclass
class GPT2ClampTokenizer(ClampTokenizer):

    tokenizer: GPT2Tokenizer

    @property
    def vocab_size(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.tokenizer.unk_token_id

    def tokenize(self, text: str) -> List[bytes]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            # Handles text with only whitespaces
            for token in list(text):
                token = "".join(
                    self.tokenizer.byte_encoder[b] for b in token.encode("utf-8")
                )
                tokens.append(token)
        token_bytes = [
            bytes([self.tokenizer.byte_decoder[c] for c in token]) for token in tokens
        ]
        return token_bytes

    @cached_property
    def utf8_token_to_id_map(  # pylint: disable=invalid-overridden-method
        self,
    ) -> Dict[bytes, int]:
        utf8_token_to_id: Dict[bytes, int] = {}
        # encoded_token: UTF-8 encoded strings where bytes corresponding to
        # control characters in ASCII have been mapped to other characters
        for encoded_token, token_id in self.tokenizer.encoder.items():
            # token_bytes: UTF-8 encoded string
            token_bytes = bytes(self.tokenizer.byte_decoder[c] for c in encoded_token)
            utf8_token_to_id[token_bytes] = token_id

        return utf8_token_to_id

    def save_pretrained(self, tokenizer_loc: str) -> None:
        self.tokenizer.save_pretrained(tokenizer_loc)

    @classmethod
    def from_pretrained(cls, tokenizer_loc: str) -> "GPT2ClampTokenizer":
        return GPT2ClampTokenizer(
            tokenizer=GPT2Tokenizer.from_pretrained(tokenizer_loc)
        )


class T5ClampTokenizer(ClampTokenizer):
    def __init__(
        self, tokenizer: T5Tokenizer, output_sequences: Optional[Iterable[str]] = None
    ):
        """
        `output_sequences` if provided, will be used to detect unk symbols and adding them to vocab.
        T5 has 28 extra token ids, if we need to add more than 28 new tokens to the vocab, we need to resize the
        token embeddings.
        """
        # Saving input tokenizer to a temp location so that it can be modified.
        tmp_tokenizer_loc = tempfile.mkdtemp()
        tokenizer.save_pretrained(tmp_tokenizer_loc)

        # Modify the sentencepiece model normalizer settings to not ignore whitespaces.
        spiece_model_file = f"{tmp_tokenizer_loc}/spiece.model"
        m = sentencepiece_model.ModelProto()
        m.ParseFromString(open(spiece_model_file, "rb").read())
        m.normalizer_spec.add_dummy_prefix = False  # pylint: disable=no-member
        m.normalizer_spec.remove_extra_whitespaces = False  # pylint: disable=no-member
        m.normalizer_spec.precompiled_charsmap = b""  # pylint: disable=no-member
        m.denormalizer_spec.add_dummy_prefix = False  # pylint: disable=no-member
        m.denormalizer_spec.remove_extra_whitespaces = (  # pylint: disable=no-member
            False
        )
        m.denormalizer_spec.precompiled_charsmap = b""  # pylint: disable=no-member

        with open(spiece_model_file, "wb") as f:
            f.write(m.SerializeToString())
        self.tokenizer = T5Tokenizer.from_pretrained(tmp_tokenizer_loc)
        self.token_to_id_map = {
            k.replace("▁", " ").encode("utf-8"): v
            for k, v in self.tokenizer.get_vocab().items()
        }
        if output_sequences is not None:
            self.update_tokenizer_with_output_sequences(output_sequences)

    def update_tokenizer_with_output_sequences(
        self, output_sequences: Iterable[str]
    ) -> None:
        tmp_tokenizer_loc = tempfile.mkdtemp()
        self.tokenizer.save_pretrained(tmp_tokenizer_loc)
        spiece_model_file = f"{tmp_tokenizer_loc}/spiece.model"
        m = sentencepiece_model.ModelProto()
        m.ParseFromString(open(spiece_model_file, "rb").read())
        t5_vocab = self.tokenizer.get_vocab()
        # Look for unk tokens and add them to the vocab
        unk_tokens = {
            token
            for output_sequence in output_sequences
            for token in self.tokenizer.tokenize(output_sequence)
            if token not in t5_vocab
        }
        if len(unk_tokens) > 0:
            print(f"Adding unk tokens to the vocab: {unk_tokens}")
        for token in unk_tokens:
            new_token = sentencepiece_model.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            m.pieces.append(new_token)  # pylint: disable=no-member

        with open(spiece_model_file, "wb") as f:
            f.write(m.SerializeToString())

        self.tokenizer = T5Tokenizer.from_pretrained(tmp_tokenizer_loc)
        self.token_to_id_map = {
            k.replace("▁", " ").encode("utf-8"): v
            for k, v in self.tokenizer.get_vocab().items()
        }

    @property
    def vocab_size(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:  # pylint: disable=invalid-overridden-method
        return 0

    @property
    def unk_token_id(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.tokenizer.unk_token_id

    def tokenize(self, text: str) -> List[bytes]:
        tokens = self.tokenizer.tokenize(text)
        return [token.replace("▁", " ").encode("utf-8") for token in tokens]

    @property
    def utf8_token_to_id_map(  # pylint: disable=invalid-overridden-method
        self,
    ) -> Dict[bytes, int]:
        return self.token_to_id_map

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        unk_token_id = 2
        return [self.utf8_token_to_id_map.get(token, unk_token_id) for token in tokens]

    def save_pretrained(self, tokenizer_loc: str) -> None:
        self.tokenizer.save_pretrained(tokenizer_loc)

    @classmethod
    def from_pretrained(
        cls, tokenizer_loc: str, output_sequences: Optional[Iterable[str]] = None
    ) -> "T5ClampTokenizer":
        return T5ClampTokenizer(
            tokenizer=T5Tokenizer.from_pretrained(tokenizer_loc),
            output_sequences=output_sequences,
        )
