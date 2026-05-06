"""Token-aware sliding-window splitter.

Uses :mod:`tiktoken` (``cl100k_base``, OpenAI's GPT-3.5/4 vocabulary) for
counts. The choice of vocabulary matters far less than choosing one and
sticking with it — chunks are sized for embedding-model context windows,
not for any specific generator, and the differences across modern BPE
vocabularies are well within the headroom we leave at 512 tokens.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

import tiktoken

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` under ``cl100k_base``."""

    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))


class TokenAwareSplitter:
    """Sliding window over tokens, decoded back to character spans.

    Why decode back to character spans: the chunk's ``char_start`` /
    ``char_end`` are stable references into the source document. Storing
    token offsets instead would tie us to a specific tokenizer version
    forever, which is the kind of decision you regret at the wrong moment.
    """

    def __init__(self, *, target_tokens: int, overlap_tokens: int) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be in [0, target_tokens)")
        self._target = target_tokens
        self._overlap = overlap_tokens

    def split(self, text: str) -> Iterator[tuple[int, int, int]]:
        """Yield ``(char_start, char_end, token_count)`` triples.

        Spans are non-empty. Adjacent spans overlap by approximately
        ``overlap_tokens`` worth of text. The last span ends at
        ``len(text)``.
        """

        text = text or ""
        if not text:
            return

        encoding = _encoding()
        ids = encoding.encode(text, disallowed_special=())
        if not ids:
            return

        step = self._target - self._overlap
        # Pre-decode running prefixes so we can compute char offsets without
        # re-encoding the whole document for every chunk. Decode-of-prefix
        # is idempotent under any sane BPE.
        start = 0
        n = len(ids)
        while start < n:
            end = min(start + self._target, n)
            chunk_ids = ids[start:end]
            chunk_text = encoding.decode(chunk_ids)

            # Recover the byte span by decoding the prefix and the
            # prefix+chunk; the difference in length is the chunk's length
            # in characters.
            prefix_text = encoding.decode(ids[:start]) if start > 0 else ""
            char_start = len(prefix_text)
            char_end = char_start + len(chunk_text)

            yield char_start, char_end, end - start

            if end == n:
                break
            start += step
