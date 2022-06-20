# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import (
    OVERNIGHT_DOMAINS,
    BenchClampDataset,
)
from semantic_parsing_with_constrained_lm.domains.create_benchclamp_splits import (
    create_benchclamp_splits,
)
from semantic_parsing_with_constrained_lm.domains.overnight import OutputType, OvernightDataPieces
from semantic_parsing_with_constrained_lm.paths import (
    BENCH_CLAMP_PROCESSED_DATA_DIR,
    OVERNIGHT_DATA_DIR,
)


def main():
    for domain in OVERNIGHT_DOMAINS:
        overnight_pieces = OvernightDataPieces.from_dir(
            OVERNIGHT_DATA_DIR,
            is_dev=True,
            domain=domain,
            output_type=OutputType.MeaningRepresentation,
            simplify_logical_forms=True,
        )
        train_data = overnight_pieces.train_data
        dev_data = overnight_pieces.test_data
        overnight_pieces = OvernightDataPieces.from_dir(
            OVERNIGHT_DATA_DIR,
            is_dev=False,
            domain=domain,
            output_type=OutputType.MeaningRepresentation,
            simplify_logical_forms=True,
        )
        test_data = overnight_pieces.test_data

        train_benchclamp_data = []
        dev_benchclamp_data = []
        test_benchclamp_data = []
        for data, benchclamp_data in [
            (train_data, train_benchclamp_data),
            (dev_data, dev_benchclamp_data),
            (test_data, test_benchclamp_data),
        ]:
            for datum in data:
                benchclamp_data.append(
                    BenchClampDatum(
                        dialogue_id=datum.dialogue_id,
                        turn_part_index=datum.turn_part_index,
                        utterance=datum.natural,
                        plan=datum.canonical,
                    )
                )

        create_benchclamp_splits(
            train_benchclamp_data,
            dev_benchclamp_data,
            test_benchclamp_data,
            BENCH_CLAMP_PROCESSED_DATA_DIR / BenchClampDataset.Overnight.value / domain,
        )


if __name__ == "__main__":
    main()
