# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from semantic_parsing_with_constrained_lm.paths import CLAMP_PRETRAINED_MODEL_DIR


def save_model_and_tokenizer(model, tokenizer, save_dir: Path) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():
    # T5
    if False:
        for model_id, huggingface_model_id in [
            ("t5-small-lm-adapt", "google/t5-small-lm-adapt"),
            ("t5-base-lm-adapt", "google/t5-base-lm-adapt"),
            ("t5-large-lm-adapt", "google/t5-large-lm-adapt"),
            ("t5-xl-lm-adapt", "google/t5-xl-lm-adapt"),
            # ("t5-xxl-lm-adapt", "google/t5-xxl-lm-adapt"),
        ]:
            print(f"Downloading {model_id} ...")
            model = T5ForConditionalGeneration.from_pretrained(huggingface_model_id)
            tokenizer = T5Tokenizer.from_pretrained(huggingface_model_id)
            save_model_and_tokenizer(
                model, tokenizer, CLAMP_PRETRAINED_MODEL_DIR / model_id
            )

        # CodeT5
        for model_id, huggingface_model_id in [
            ("codet5-base", "Salesforce/codet5-base"),
            ("codet5-base-multi-sum", "Salesforce/codet5-base-multi-sum"),
        ]:
            print(f"Downloading {model_id} ...")
            model = T5ForConditionalGeneration.from_pretrained(huggingface_model_id)
            tokenizer = RobertaTokenizer.from_pretrained(huggingface_model_id)
            save_model_and_tokenizer(
                model, tokenizer, CLAMP_PRETRAINED_MODEL_DIR / model_id
            )

        # Bart
        for model_id, huggingface_model_id in [
            ("bart-base", "facebook/bart-base"),
            ("bart-large", "facebook/bart-large"),
        ]:
            print(f"Downloading {model_id} ...")
            model = BartForConditionalGeneration.from_pretrained(huggingface_model_id)
            tokenizer = BartTokenizer.from_pretrained(huggingface_model_id)
            save_model_and_tokenizer(
                model, tokenizer, CLAMP_PRETRAINED_MODEL_DIR / model_id
            )

    # CodeGen
    for model_id, huggingface_model_id in [
        # ("codegen-350M", "Salesforce/codegen-350M-mono"),
        #  ("codegen-2B", "Salesforce/codegen-2B-mono"),
        #("codegen-6B", "Salesforce/codegen-6B-mono")
         ("codegen-16B", "Salesforce/codegen-16B-mono"),
    ]:
        print(f"Downloading {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(huggingface_model_id)
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id)
        save_model_and_tokenizer(
            model, tokenizer, CLAMP_PRETRAINED_MODEL_DIR / model_id
        )

if __name__ == "__main__":
    main()
