# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import dataclasses
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Tuple, TypeVar, Union

from semantic_parsing_with_constrained_lm.util.unit import UNIT, Unit
from semantic_parsing_with_constrained_lm.async_tools.limits import TimeoutBarrier

I = TypeVar("I")
O = TypeVar("O")


class BatchMaker(Generic[I, O], ABC):
    """Identifies which elements could be batched together, and specifies how to do the batched operation.

    This should also derive from collections.abc.Hashable, but then it is not
    possible to use a dataclass to create a BatchMaker.
    """

    @property
    @abc.abstractmethod
    def max_batch_size(self) -> int:
        """Maximum number of elements to batch together."""
        pass

    @property
    @abc.abstractmethod
    def timeout(self) -> float:
        """Time to wait before running `execute`."""
        pass

    @abc.abstractmethod
    async def execute(self, inputs: List[I]) -> O:
        """Batched operation on inputs."""
        pass


@dataclass
class PendingContainer(Generic[I, O]):
    """Used inside BatchingHelper."""

    batch_key: BatchMaker[I, O]
    batching_helper: "BatchingHelper[I, O]"

    inputs: List[I] = dataclasses.field(default_factory=list)
    barrier: TimeoutBarrier = dataclasses.field(init=False)
    result: Union[O, Unit] = UNIT

    def __post_init__(self):
        self.barrier = TimeoutBarrier(
            self.batch_key.max_batch_size, self.batch_key.timeout, self._execute
        )

    async def enqueue_and_wait(self, inp: I) -> Tuple[O, int]:
        i = len(self.inputs)
        self.inputs.append(inp)
        await self.barrier.arrive_and_wait()
        assert not isinstance(self.result, Unit)
        return self.result, i

    @property
    def closed(self) -> bool:
        return bool(self.barrier.currently_releasing or self.result is not UNIT)

    async def _execute(self) -> None:
        self.result = await self.batch_key.execute(self.inputs)
        self.batching_helper._del_pending_container(  # pylint: disable=protected-access
            self
        )


@dataclass
class BatchingHelper(Generic[I, O]):
    """Helper for running functions on batched inputs."""

    # Creates a BatchMaker from the input. Inputs with BatchMakers that compare equal are eligible for batching together.
    input_to_batch_maker: Callable[[I], BatchMaker[I, O]]

    # Pending operations per BatchMaker.
    pending: Dict[BatchMaker[I, O], PendingContainer[I, O]] = dataclasses.field(
        default_factory=dict
    )

    async def execute(self, inp: I) -> Tuple[O, int]:
        """Given an input of type I, this class uses `batch_key_fn` to create a BatchKey.
        Inputs with the same BatchKey are coalesced together.
        After a certain number of inputs are collected, or a timeout passes,
        we run BatchKey.execute to produce a batched output of type O.

        Returns the output O with a batch index to locate the result for the input within O."""

        batch_maker = self.input_to_batch_maker(inp)  # type: ignore[call-arg]
        pending_container = self.pending.get(batch_maker)

        if pending_container is None or pending_container.closed:
            self.pending[batch_maker] = pending_container = PendingContainer(
                batch_maker, self
            )

        return await pending_container.enqueue_and_wait(inp)

    def _del_pending_container(self, container: PendingContainer[I, O]):
        """When a PendingContainer is done executing, immediately remove it from `self.pending`"""
        if self.pending.get(container.batch_key) is container:
            del self.pending[container.batch_key]
