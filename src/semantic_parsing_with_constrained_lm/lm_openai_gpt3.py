# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""IncrementalLanguageModel which uses OpenAI's GPT-3 API."""

import ast
import asyncio
import collections
import dataclasses
import datetime
import functools
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import httpx
import more_itertools
import torch
from cached_property import cached_property
from httpx import Response
from httpx._types import HeaderTypes
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.async_tools import limits
from semantic_parsing_with_constrained_lm.async_tools.batch_helper import BatchingHelper, BatchMaker
from semantic_parsing_with_constrained_lm.cache import CacheClient
from semantic_parsing_with_constrained_lm.lm import IncrementalLanguageModel, TokensWithLogprobs
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer, GPT2ClampTokenizer

try:
    from semantic_parsing_with_constrained_lm.internal.cosmos_db_client import make_default_client
    from semantic_parsing_with_constrained_lm.internal.gpt3 import adjust_tokenizer
except ImportError:
    make_default_client = lambda: None
    adjust_tokenizer = lambda _1, _2: None

default_engine = os.environ.get("OPENAI_GPT3_ENGINE", "text-davinci-001")


@dataclass
class OpenAIGPT3State:
    tokens: Tuple[int, ...]


class OpenAIAPIError(Exception):
    def __init__(self, response: Response):
        self.response = response
        self.message = (
            f"Got status {response.status_code} from OpenAI with body:\n{response.text}"
        )
        super().__init__(self.message)


@dataclass
class GPT3Client:
    engine: str = default_engine
    api_key: Optional[str] = None

    cache_client: Optional[CacheClient] = dataclasses.field(
        default_factory=make_default_client
    )
    http_client: httpx.AsyncClient = dataclasses.field(init=False)
    request_limiter: limits.AdaptiveLimiter = dataclasses.field(
        default_factory=functools.partial(
            limits.AdaptiveLimiter, initial_qps=10, max_qps=100
        )
    )
    completions_rate_limited: Callable[
        [Dict[str, Any]], Awaitable[httpx.Response]
    ] = dataclasses.field(init=False)
    completions_url: str = dataclasses.field(init=False)

    def _init_api_key(self, env: str) -> str:
        if self.api_key is None:
            self.api_key = os.getenv(env)
        if self.api_key is None:
            raise ValueError(f"{env} was not set")
        return self.api_key

    def __post_init__(self):
        # We have an internal instance which has a different URL, auth token and header.
        # To access that instance, you can use the engine "codex-cushman-sm" -- note that
        # the "-sm" suffix is not part of the actual underlying engine name, just something
        # we use to switch on here.
        # See https://semanticmachines.slack.com/archives/C017P5M1RSL/p1647450366073519?thread_ts=1646782663.584339&cid=C017P5M1RSL
        # Get keys here: https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/b68b2f37-1d37-4c2f-80f6-c23de402792e/resourceGroups/fod/providers/Microsoft.CognitiveServices/accounts/smopenai/cskeys
        auth_header: HeaderTypes
        if self.engine == "codex-cushman-sm":
            api_key = self._init_api_key("SM_OPENAI_API_KEY")
            self.completions_url = "https://smopenai.openai.azure.com/openai/deployments/codex-cushman/completions?api-version=2021-11-01-preview"
            auth_header = {"api-key": api_key}
        else:
            api_key = self._init_api_key("OPENAI_API_KEY")
            self.completions_url = (
                f"https://api.openai.com/v1/engines/{self.engine}/completions"
            )
            auth_header = {"Authorization": f"Bearer {api_key}"}

        self.http_client = httpx.AsyncClient(
            headers=auth_header,
            # HTTP/2 should be more efficient, but it appears to be buggy in practice
            http2=False,
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=500),
        )
        # Pyright bug forces us to first store the result in `limited`
        # https://github.com/microsoft/pyright/issues/2965
        limited = self.request_limiter(self._completions_with_raise_if_limited)
        self.completions_rate_limited = limited

    async def __aenter__(self):
        await self.http_client.__aenter__()
        if self.cache_client is not None:
            await self.cache_client.__aenter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.http_client.__aexit__(exc_type, exc_value, traceback)
        if self.cache_client is not None:
            await self.cache_client.__aexit__(exc_type, exc_value, traceback)

    async def _completions_with_raise_if_limited(
        self, args_without_engine: Dict[str, Any]
    ) -> httpx.Response:
        request_info = RequestInfo.create(args_without_engine)
        Instrumentation.currently_pending_requests += 1
        Instrumentation.record_request(request_info)
        try:
            response = await self.http_client.post(
                self.completions_url,
                json=args_without_engine,
            )
        except httpx.RequestError as e:
            request_info.finish(False)
            raise limits.RateLimitExceededError() from e
        finally:
            Instrumentation.currently_pending_requests -= 1

        if response.status_code != 200:
            request_info.finish(False)
            if response.status_code in (429, 500, 502, 503):
                raise limits.RateLimitExceededError()
            raise OpenAIAPIError(response)

        request_info.finish(True)
        return response


@dataclass(frozen=True)
class EchoBatchMaker(BatchMaker):
    client: GPT3Client = dataclasses.field(compare=False)

    @property
    def max_batch_size(self) -> int:
        return 1000

    @property
    def timeout(self) -> float:
        return 0.1

    async def execute(self, batched_tokens: List[Sequence[int]]) -> List[List[float]]:
        args = {
            "prompt": batched_tokens,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
        }
        # https://github.com/python/mypy/issues/708
        results = (
            await self.client.completions_rate_limited(args)  # type: ignore
        ).json()
        return [d["logprobs"]["token_logprobs"] for d in results["choices"]]


@dataclass(frozen=True)
class NextLogprobsBatchMaker(BatchMaker):
    client: GPT3Client = dataclasses.field(compare=False)

    @property
    def max_batch_size(self) -> int:
        return 100

    @property
    def timeout(self) -> float:
        return 0.001

    async def execute(
        self, batched_tokens: List[Sequence[int]]
    ) -> List[Dict[str, float]]:
        args = {
            "prompt": batched_tokens,
            "max_tokens": 1,
            "logprobs": 100,
        }
        # https://github.com/python/mypy/issues/708
        results = (
            await self.client.completions_rate_limited(args)  # type: ignore
        ).json()
        return [d["logprobs"]["top_logprobs"][0] for d in results["choices"]]


@dataclass(frozen=True)
class CompletionsParams:
    max_tokens: int
    temperature: float
    top_p: float
    num_completions: int
    stop: Optional[str]


@dataclass(frozen=True)
class CompletionsBatchMaker(BatchMaker):
    client: GPT3Client = dataclasses.field(compare=False)
    params: CompletionsParams

    @property
    def max_batch_size(self) -> int:
        return 100

    @property
    def timeout(self) -> float:
        return 0.05

    async def execute(
        self, args: List[Tuple[Sequence[int], CompletionsParams]]
    ) -> List[List[Tuple[List[str], List[float]]]]:
        # Outermost List has length equal to the batch size
        # 2nd level List has length equal to `num_completions`
        # Each Tuple contains two (parallel) lists: tokens and their log probabilities
        batched_tokens = [x[0] for x in args]
        params = {
            "prompt": batched_tokens,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
            "n": self.params.num_completions,
            "stop": self.params.stop,
            "logprobs": 0,
        }
        response = (
            await self.client.completions_rate_limited(params)  # type: ignore
        ).json()

        result: List[List[Tuple[List[str], List[float]]]] = []
        for choices_per_prompt in more_itertools.chunked(
            response["choices"], self.params.num_completions
        ):
            result.append(
                [
                    (c["logprobs"]["tokens"], c["logprobs"]["token_logprobs"])
                    for c in choices_per_prompt
                ]
            )
        return result


def openai_token_to_bytes(token: str) -> bytes:
    if token.startswith("bytes:"):
        return ast.literal_eval(f"b'{token[6:]}'")
    else:
        return token.encode("utf-8")


def openai_token_to_id(tokenizer: ClampTokenizer, token: str) -> int:
    token_bytes = openai_token_to_bytes(token)
    return tokenizer.utf8_token_to_id_map[token_bytes]


@dataclass
class IncrementalOpenAIGPT3(IncrementalLanguageModel[OpenAIGPT3State]):
    engine: str = default_engine
    use_cache: bool = True

    client: GPT3Client = dataclasses.field(init=False)
    echo_batch_helper: BatchingHelper[
        Sequence[int], List[List[float]]
    ] = dataclasses.field(init=False)
    next_logprobs_batch_helper: BatchingHelper[
        Sequence[int], List[Dict[str, float]]
    ] = dataclasses.field(init=False)
    completions_batch_helper: BatchingHelper[
        Tuple[Sequence[int], CompletionsParams],
        List[List[Tuple[List[str], List[float]]]],
    ] = dataclasses.field(init=False)

    def __post_init__(self):
        client = GPT3Client(engine=self.engine)
        self.client = client
        self.echo_batch_helper = BatchingHelper(
            input_to_batch_maker=lambda _args: EchoBatchMaker(client),
        )
        self.next_logprobs_batch_helper = BatchingHelper(
            input_to_batch_maker=lambda _args: NextLogprobsBatchMaker(client),
        )
        self.completions_batch_helper = BatchingHelper(
            input_to_batch_maker=lambda args: CompletionsBatchMaker(client, args[1]),
        )
        if self.client.cache_client is None:
            self.use_cache = False

    async def __aenter__(self):
        await self.client.__aenter__()

    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)

    @cached_property
    def vocab_size(self):  # pylint: disable=invalid-overridden-method
        return self.tokenizer.vocab_size

    @cached_property
    def tokenizer(self) -> ClampTokenizer:  # pylint: disable=invalid-overridden-method
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        adjust_tokenizer(self.engine, gpt2_tokenizer)
        return GPT2ClampTokenizer(gpt2_tokenizer)

    @cached_property
    def max_length(self) -> int:
        if self.engine.startswith("davinci-codex"):
            return 4096
        return 2048

    async def execute(
        self,
        tokens: Sequence[int],
        hidden_state: Optional[OpenAIGPT3State] = None,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[OpenAIGPT3State]]:
        # In order to reduce network traffic, this function only returns the
        # logprobs for the last token. It also only returns the top 100 logprobs
        # due to limitations of the OpenAI API.
        if hidden_state is None:
            all_tokens = tuple(tokens)
        else:
            all_tokens = hidden_state.tokens + tuple(tokens)

        if self.use_cache and self.client.cache_client:
            cache_args = {
                "engine": self.engine,
                "prompt": all_tokens,
                "max_tokens": 1,
                "logprobs": 100,
            }
            cached = await self.client.cache_client.get(cache_args)
        else:
            cache_args = None
            cached = None

        if cached:
            next_logprobs = cached["choices"][0]["logprobs"]["top_logprobs"][0]
        else:
            batched_next_logprobs, i = await self.next_logprobs_batch_helper.execute(
                all_tokens
            )
            next_logprobs = batched_next_logprobs[i]
            if self.use_cache and self.client.cache_client:
                assert cache_args is not None
                asyncio.create_task(
                    self.client.cache_client.upload(
                        cache_args,
                        {"choices": [{"logprobs": {"top_logprobs": [next_logprobs]}}]},
                    )
                )

        result = torch.full(
            (max(1, len(tokens)), self.tokenizer.vocab_size), -float("inf")
        )
        for token, logprob in next_logprobs.items():
            token_id = openai_token_to_id(self.tokenizer, token)
            result[-1, token_id] = logprob

        return (
            result,
            None if drop_next_hidden_state else OpenAIGPT3State(all_tokens),
        )

    async def next_logprobs(self, hidden_state: OpenAIGPT3State) -> torch.Tensor:
        # First [0] is to get the logprobs, not the hidden state
        # Second [0] is to remove the length dimension
        return (await self.execute((), hidden_state, drop_next_hidden_state=True))[0][0]

    async def logprob_of_completion(
        self, prefix_tokens: Sequence[int], completion_tokens: Sequence[int]
    ) -> float:
        all_tokens = tuple(prefix_tokens) + tuple(completion_tokens)
        if self.use_cache and self.client.cache_client:
            assert self.client.cache_client is not None
            cache_args = {
                "prompt": all_tokens,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 0,
            }
            cached = await self.client.cache_client.get(cache_args)
        else:
            cache_args = None
            cached = None

        if cached:
            echoed_logprobs = cached["choices"][0]["logprobs"]["token_logprobs"]
        else:
            batched_echoed_logprobs, i = await self.echo_batch_helper.execute(
                all_tokens
            )
            echoed_logprobs = batched_echoed_logprobs[i]
            if self.use_cache and self.client.cache_client:
                assert cache_args is not None
                asyncio.create_task(
                    self.client.cache_client.upload(
                        cache_args,
                        {
                            "choices": [
                                {"logprobs": {"token_logprobs": echoed_logprobs}}
                            ]
                        },
                    )
                )

        return sum(echoed_logprobs[len(prefix_tokens) :])

    async def completions(
        self,
        tokens: Sequence[int],
        max_tokens: int,
        temperature: float = 1,
        top_p: float = 1,
        num_completions: int = 1,
        stop: Optional[str] = None,
        hidden_state: Optional[OpenAIGPT3State] = None,
    ) -> List[Tuple[TokensWithLogprobs, OpenAIGPT3State]]:
        """Corresponds to completions endpoint of OpenAI API.

        https://beta.openai.com/docs/api-reference/completions/create"""
        if hidden_state is None:
            all_tokens = tuple(tokens)
        else:
            all_tokens = hidden_state.tokens + tuple(tokens)

        all_completions, i = await self.completions_batch_helper.execute(
            (
                all_tokens,
                CompletionsParams(
                    max_tokens, temperature, top_p, num_completions, stop
                ),
            )
        )
        completions = all_completions[i]

        result: List[Tuple[TokensWithLogprobs, OpenAIGPT3State]] = []
        for completion_tokens, logprobs in completions:
            truncated_token_ids = []
            prev_was_stop = False
            for t in completion_tokens:
                if prev_was_stop:
                    break
                prev_was_stop = t == stop
                truncated_token_ids.append(openai_token_to_id(self.tokenizer, t))

            result.append(
                (
                    TokensWithLogprobs(
                        torch.tensor(truncated_token_ids),
                        torch.tensor(logprobs[: len(truncated_token_ids)]),
                    ),
                    OpenAIGPT3State(all_tokens + tuple(truncated_token_ids[:-1])),
                )
            )

        return result


@dataclass
class RequestInfo:
    num_prompts: int
    prompts: List[str]
    success: Optional[bool] = dataclasses.field(init=False, default=None)
    start_time: float = dataclasses.field(default_factory=time.time)
    end_time: Optional[float] = dataclasses.field(init=False, default=None)

    @staticmethod
    def create(args_without_engine: Dict[str, Any]) -> "RequestInfo":
        prompt = args_without_engine["prompt"]
        if isinstance(prompt, str):
            return RequestInfo(1, [prompt])
        else:
            return RequestInfo(len(prompt), prompt)

    def finish(self, success: bool) -> None:
        self.end_time = time.time()
        self.success = success


class Instrumentation:
    BUFFER_SIZE = 100
    AUTOMATIC_PRINTING_ENABLED = True
    PRINT_PROMPT_CONTENTS = False

    currently_pending_requests = 0
    dropped_requests = 0
    last_n_requests: Deque[RequestInfo] = collections.deque()
    requests_lock: threading.Lock = threading.Lock()
    last_printed_timestamp: float = time.time()

    @staticmethod
    def record_request(ri: RequestInfo) -> None:
        with Instrumentation.requests_lock:
            if len(Instrumentation.last_n_requests) == Instrumentation.BUFFER_SIZE:
                Instrumentation.last_n_requests.popleft()
                Instrumentation.dropped_requests += 1
            Instrumentation.last_n_requests.append(ri)

    @staticmethod
    def print_last_requests():
        Instrumentation.last_printed_timestamp = time.time()
        with Instrumentation.requests_lock:
            last_n_requests = list(Instrumentation.last_n_requests)
            dropped_requests = Instrumentation.dropped_requests

            Instrumentation.last_n_requests = collections.deque(
                [ri for ri in last_n_requests if not ri.end_time]
            )
            Instrumentation.dropped_requests = 0

        if not last_n_requests:
            return

        if dropped_requests:
            dropped_str = f" ({dropped_requests} not shown)"
        else:
            dropped_str = ""

        lines: List[str] = []
        for ri in last_n_requests:
            line_parts: List[str] = [
                f"- {ri.num_prompts} prompts, "
                f"started at {datetime.datetime.fromtimestamp(ri.start_time).strftime('%H:%M:%S.%f')}, "
            ]
            if ri.end_time:
                line_parts += [
                    f"elapsed {ri.end_time - ri.start_time:.3f}s, "
                    f"success {ri.success}"
                ]
            else:
                line_parts += ["pending"]
            lines.append("".join(line_parts))
            if Instrumentation.PRINT_PROMPT_CONTENTS:
                for prompt in ri.prompts:
                    lines.append(f"  * {prompt!r}")

        print(
            "*** GPT-3 API request report ***\n"
            f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {len(last_n_requests)} requests since last report{dropped_str}:\n"
            + "\n".join(lines)
            + "\n********************************",
            file=sys.stderr,
        )

    @staticmethod
    def print_loop():
        while Instrumentation.AUTOMATIC_PRINTING_ENABLED:
            time.sleep(1)
            if time.time() > Instrumentation.last_printed_timestamp + 10:
                Instrumentation.print_last_requests()


if Instrumentation.AUTOMATIC_PRINTING_ENABLED:
    threading.Thread(target=Instrumentation.print_loop, daemon=True).start()
