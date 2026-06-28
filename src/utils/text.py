import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_+#.]+")


# ==========================================================
# Normalization
# ==========================================================

def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into a single space.
    """
    if not text:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(text: str) -> str:
    """
    Unicode-safe lowercase normalization.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    return normalize_whitespace(text).lower()


# ==========================================================
# Tokenization
# ==========================================================

@lru_cache(maxsize=50000)
def tokenize(text: str) -> tuple[str, ...]:
    """
    Fast cached tokenizer.
    """

    return tuple(
        TOKEN_PATTERN.findall(
            normalize_text(text)
        )
    )


def tokenize_list(text: str) -> List[str]:
    """
    Returns a mutable list of tokens.
    """

    return list(tokenize(text))


# ==========================================================
# Deduplication
# ==========================================================

def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """
    Removes duplicates while preserving order.
    """

    seen = set()

    result = []

    append = result.append

    for value in values:

        if value is None:
            continue

        value = normalize_whitespace(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)

        append(value)

    return result


# ==========================================================
# Helpers
# ==========================================================

def contains_phrase(text: str, phrase: str) -> bool:
    """
    Case-insensitive phrase search.
    """

    return normalize_text(phrase) in normalize_text(text)


def normalize_skill(skill: str) -> str:
    """
    Normalize skill names for matching.
    """

    return normalize_text(skill).replace("-", " ")


def join_non_empty(*parts: str) -> str:
    """
    Efficient string join.
    """

    return " ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def unique_tokens(text: str) -> List[str]:
    """
    Returns unique tokens preserving order.
    """

    return dedupe_preserve_order(tokenize_list(text))