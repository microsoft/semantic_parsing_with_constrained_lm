# BenchCLAMP 

This is a benchmark allowing researchers to quickly use a pretrained language model
to perform semantic parsing using CLAMP framework (Constrained Language Model Parsing). 
We provide multiple semantic 
parsing datasets, with splits of different data sizes, as well as grammars which can be 
used to constrain decoding of semantic parses.

### BenchClamp Datasets
The current benchmark supports 6 datasets:
1. CalFlowV2
2. TreeDST (in LispressV2 format)
3. MTOP (all languages)
4. Overnight (all domains)
5. Spider
6. CoSQL

For each dataset, we create three low data train sets (`low_0`, `low_1`,
`low_2`) each containing 500 examples, one
medium data train set (`med_0`) with 5k examples, and a single full data 
split (`all`).

### Fine-tune a Language Model
1. You can edit `benchclamp_config.py` to add your LM to the `TRAIN_MODEL_CONFIGS` 
list. We have already added some popular LMs to the list.
    ```
    TRAIN_MODEL_CONFIGS: List[ClampModelConfig] = [
        T5ModelConfig(
            model_id="t5-base-lm-adapt",
            model_loc=HUGGINGFACE_MODEL_DIR / "t5-base-lm-adapt",
        ),
    ]
    ```  

2. Download pretrained models from Huggingface and place them in `HUGGINGFACE_MODEL_DIR`
directory.

3. You will need to set `TRAINED_MODEL_DIR` and `LOG_DIR` to save trained models
and logs from the experiments. For running evaluation on the SQL datasets, you
will also need to set the variables `TEST_SUITE_PATH`, 
`TEST_SUITE_DATABASE_PATH`, `SPIDER_DATABASE_PATH`,
`SPIDER_TABLES_FILE`, `COSQL_DATABASE_PATH` and `COSQL_TABLES_FILE`. 

4. To train the t5-base model on all CalflowV2 `low_0` train split with learning rate
`0.0001`, you can run the following:
   ```
   python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
   --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
   --exp-name-pattern 't5-base-lm-adapt_calflow_last_agent_low_0_0.0001'
   ```
   Training runs for 10000 steps, saving checkpoints at the end of 5000 and 10000 steps. 
   You can repeat the process for all learning rates listed in `LRS`. `last_agent` refers
   to the context being used in the input sequence.

5. Run evaluation on the trained models using the following command. First time you run this, it will run 
evaluation on the dev set for all checkpoints (corresponding to the different
learning rates and steps). Once all dev experiments are run, the same command 
will compare all dev performance and evaluate the best model on the test set.   
   ```
   python -m semantic_parsing_with_constrained_lm.run_exp \
   --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
   --exp-name-pattern 't5-base-lm-adapt_calflow_last_agent_low_0_.*'
   ```

6. We currently support running T5, Bart and GPT2 like models. If your language model
does not fall in these categories, you will need to implement a ClampModelConfig for your
model. 

### Few Shot Prompt GPT-3 or Codex

1. For few shot prompted experiments, you can use `benchclamp_gpt3_config.py`.
An example command below:
   ```
   python -m semantic_parsing_with_constrained_lm.run_exp \
   --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_gpt3_config \
   --exp-name-pattern 'text-davinci-001_calflow_no_context_low_0_2_dev_eval_constrained_bs_5'
   ```
where `text-davinci-001` refers to the OpenAI model being used, `bs_5` denotes beam size 5,
`2` denotes a best last prompt order (`0` for random shuffle and `1` for best first prompt order).


### BenchClamp Creation Steps 
(for documentation only, does not need to be run again)

1. Download the necessary Huggingface models by running. You can add models
of your choice to the script. 
    ```
    python src/semantic_parsing_with_constrained_lm/finetune/download_huggingface_lms.py
    ```

2. Create data and grammar for CalflowV2 and TreeDST:
   ```
   python src/semantic_parsing_with_constrained_lm/domains/lispress_v2/create_benchclamp_data.py
   ```
   
3. Create data and grammar for MTOP:
   ```
   python src/semantic_parsing_with_constrained_lm/domains/mtop/create_benchclamp_data.py
   ```
   
4. Create splits for Overnight:
   ```
   python src/semantic_parsing_with_constrained_lm/domains/overnight/create_benchclamp_data.py
   ```
   
5. Create splits for Spider and CoSQL:
   ```
   python src/semantic_parsing_with_constrained_lm/domains/sql/create_benchclamp_data.py
   ```
