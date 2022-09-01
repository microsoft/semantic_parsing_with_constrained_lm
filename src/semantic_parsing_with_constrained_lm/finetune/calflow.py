# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from typing import List, Optional, Set

from appdirs import user_cache_dir
from blobfile import BlobFile

from semantic_parsing_with_constrained_lm.datum import FullDatum
from semantic_parsing_with_constrained_lm.domains.calflow.write_data import (
    dialogues_from_calflow_textio,
)


def calflow_to_datum_format(
    calflow_data_file: str, whitelisted_dialogue_ids: Optional[Set[str]] = None
) -> List[FullDatum]:
    """
    Reads Calflow turns from CalflowTypesystem processed data into datum format.
    Agent context is set to a dict containing last turn user utterance and plan.
    """
    data_list = []
    smcalflow_cache_path = user_cache_dir("semantic_parsing_as_constrained_lm")
    for calflow_dialogue in dialogues_from_calflow_textio(
        BlobFile(calflow_data_file, streaming=False, cache_dir=smcalflow_cache_path)
    ):
        if (
            whitelisted_dialogue_ids is not None
            and calflow_dialogue.dialogue_id not in whitelisted_dialogue_ids
        ):
            continue
        for index, calflow_turn in enumerate(calflow_dialogue.turns):
            if calflow_turn.skip:
                continue
            prev_turn_user_utterance = (
                calflow_dialogue.turns[index - 1].user_utterance.original_text
                if index > 0
                else ""
            )
            prev_turn_agent_utterance = (
                calflow_dialogue.turns[index - 1].agent_utterance.original_text
                if index > 0
                else ""
            )
            prev_turn_plan = (
                calflow_dialogue.turns[index - 1].lispress if index > 0 else ""
            )
            agent_context = json.dumps(
                {
                    "user_utterance": prev_turn_user_utterance,
                    "agent_utterance": prev_turn_agent_utterance,
                    "plan": prev_turn_plan,
                }
            )
            data_list.append(
                FullDatum(
                    natural=calflow_turn.user_utterance.original_text,
                    canonical=calflow_turn.lispress,
                    agent_context=agent_context,
                    dialogue_id=calflow_dialogue.dialogue_id,
                    turn_part_index=calflow_turn.turn_index,
                )
            )

    return data_list
