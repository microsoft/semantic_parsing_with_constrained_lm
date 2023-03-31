# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path



DOMAINS_DIR = Path(__file__).resolve().parent / "domains"

CALFLOW_EXAMPLES_DIR = DOMAINS_DIR / "calflow/data"

CALFLOW_GRAMMAR_DIR = DOMAINS_DIR / "calflow/grammar"


# Paths used in data preparation for BenchClamp
RUN_ON_AML = "AMLT_EXPERIMENT_NAME" in os.environ

CLAMP_PRETRAINED_MODEL_DIR = Path(os.environ.get("TRANSFORMERS_CACHE", "huggingface_models/")) 
#CLAMP_PRETRAINED_MODEL_DIR = (
#    Path("/mnt/default/huggingface_models/")
#    if RUN_ON_AML
#    else Path("huggingface_models/")
#)

# CLAMP_DATA_DIR = (
    # Path("/mnt/default/clamp_data/") if RUN_ON_AML else Path("data")
# )
CLAMP_DATA_DIR = Path("/brtx/601-nvme1/estengel/resources/data/")

OVERNIGHT_DATA_DIR = CLAMP_DATA_DIR / "overnight"

BENCH_CLAMP_DATA_DIR_ROOT = CLAMP_DATA_DIR / "benchclamp"

BENCH_CLAMP_RAW_DATA_DIR = BENCH_CLAMP_DATA_DIR_ROOT / "raw"

BENCH_CLAMP_PROCESSED_DATA_DIR = BENCH_CLAMP_DATA_DIR_ROOT / "processed"

BENCH_CLAMP_GRAMMAR_DATA_DIR = BENCH_CLAMP_DATA_DIR_ROOT / "grammar"


# Paths for users of BenchClamp. Kept as strings since Path does not work well with network paths.
CLAMP_DATA_DIR_AZURE = "https://benchclamp.blob.core.windows.net/benchclamp"

OVERNIGHT_DATA_DIR_AZURE = CLAMP_DATA_DIR_AZURE + "/overnight"

BENCH_CLAMP_DATA_DIR_ROOT_AZURE = CLAMP_DATA_DIR_AZURE + "/benchclamp"

BENCH_CLAMP_PROCESSED_DATA_DIR_AZURE = BENCH_CLAMP_DATA_DIR_ROOT_AZURE + "/processed"

BENCH_CLAMP_GRAMMAR_DATA_DIR_AZURE = BENCH_CLAMP_DATA_DIR_ROOT_AZURE + "/grammar"
