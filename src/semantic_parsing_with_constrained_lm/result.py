# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from semantic_parsing_with_constrained_lm.model import ModelResult


@dataclass(frozen=True, eq=True)
class DatumResult:
    """CLAMP predictions and results for a single datum"""

    # Test datum utterance
    test_datum_natural: str

    # Text and cost of each sequence in the final beam
    results: List[ModelResult]

    # Text of each sequence in the final beam
    # (Duplicated from `results`; maintained here only
    # for backwards compatibility. May be removed later.)
    outputs: List[str]

    # The metrics dictionary containing the main results
    metrics: Dict[str, Optional[str]]

    # Other (optional) test datum fields
    test_datum_id: Optional[str] = None
    test_datum_turn_part_index: Optional[int] = None
    test_datum_agent_context: Optional[str] = None
    test_datum_canonical: Optional[str] = None

    # Token-level log probabilities for each sequence in the final beam
    # (Not yet implemented)
    token_logprobs: Optional[List[List[Tuple[str, float]]]] = None
