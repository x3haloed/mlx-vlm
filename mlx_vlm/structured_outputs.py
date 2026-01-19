"""Helpers for constrained / structured generation.

This module intentionally does not encode OpenAI request semantics.
It provides small building blocks that servers can use to translate an API
request (e.g., `response_format`) into a logits processor compatible with
`mlx_vlm.generate(..., logits_processors=[...])`.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def build_outlines_json_schema_logits_processor(
    *,
    tokenizer: Any,
    schema: Dict[str, Any],
    eos_token_id: Any | None = None,
):
    """Build an Outlines (outlines_core) JSON-schema logits processor for MLX.

    Parameters
    ----------
    tokenizer
        A tokenizer that exposes `get_vocab()` and `eos_token_id`. For
        HuggingFace tokenizers this is typically `processor.tokenizer`.
    schema
        A JSON schema as a Python dict (Draft 7).

    Returns
    -------
    Any
        An Outlines logits processor compatible with MLX tensors.

    Notes
    -----
    This function imports `outlines`/`outlines_core` lazily and will raise an
    ImportError if they are not installed.
    """

    from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
    from outlines_core import Index, Vocabulary
    from outlines_core.json_schema import build_regex_from_schema
    import outlines_core.kernels.mlx  # noqa: F401

    if not hasattr(tokenizer, "get_vocab"):
        raise TypeError("Tokenizer must implement get_vocab().")
    eos_token_ids = eos_token_id
    if eos_token_ids is None:
        eos_token_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_token_ids is None:
        raise TypeError("Tokenizer must define eos_token_id (or pass eos_token_id=...).")

    if isinstance(eos_token_ids, (list, tuple, set)):
        eos_token_ids = list(eos_token_ids)
        if not eos_token_ids:
            raise TypeError("eos_token_id must not be an empty list.")
        primary_eos_token_id = int(eos_token_ids[0])
        eos_token_id_set = {int(x) for x in eos_token_ids if x is not None}
    else:
        primary_eos_token_id = int(eos_token_ids)
        eos_token_id_set = {primary_eos_token_id}

    if hasattr(tokenizer, "convert_token_to_string"):
        token_to_str = tokenizer.convert_token_to_string
    elif hasattr(tokenizer, "convert_tokens_to_string"):
        token_to_str = lambda token: tokenizer.convert_tokens_to_string([token])
    else:
        token_to_str = str

    formatted_vocab: Dict[str, list[int]] = {}
    for token, token_id in tokenizer.get_vocab().items():
        formatted_vocab[token_to_str(token)] = [int(token_id)]

    # Remove any entries that correspond to any EOS token ids. Outlines Core
    # expects EOS to be tracked separately via `Vocabulary(primary_eos_token_id, ...)`.
    for token, token_id in tokenizer.get_vocab().items():
        if int(token_id) in eos_token_id_set:
            formatted_vocab.pop(token_to_str(token), None)

    vocabulary = Vocabulary(primary_eos_token_id, formatted_vocab)
    json_schema_str = json.dumps(schema, ensure_ascii=True)
    regex = build_regex_from_schema(json_schema_str)
    index = Index(regex, vocabulary)
    return OutlinesCoreLogitsProcessor(index, "mlx")
