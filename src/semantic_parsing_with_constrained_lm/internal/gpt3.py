from transformers import GPT2Tokenizer

def add_merged_spaces(tokenizer: GPT2Tokenizer, n_merged_spaces: int = 24) -> None:
    """Adjust GPT2Tokenizer for Codex."""
    space_char = tokenizer.byte_encoder[ord(" ")]
    for i in range(1, n_merged_spaces):
        for j in range(1, n_merged_spaces):
            if i + j <= n_merged_spaces:
                tokenizer.bpe_ranks[space_char * i, space_char * j] = len(
                    tokenizer.bpe_ranks
                )

    for i in range(n_merged_spaces):
        next_id = len(tokenizer.encoder)
        tokenizer.encoder[space_char * (i + 2)] = next_id
        tokenizer.decoder[next_id] = space_char * (i + 2)


def adjust_tokenizer(engine: str, tokenizer: GPT2Tokenizer) -> None:
    if engine in (
        "codex-cushman-sm",
        "code-cushman-001",
        "code-davinci-001",
        "text-davinci-001",
        "code-davinci-002",
        "text-davinci-002",
        "code-davinci-003",
        "text-davinci-003",
    ):
        add_merged_spaces(tokenizer)


