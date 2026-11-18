"""Text and filename normalization (one implementation for every package of the project)."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase a string and strip diacritics/accents.

    Args:
        text: Input text.

    Returns:
        The lowercased, accent-stripped text.
    """
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def sanitize_filename(name: str) -> str:
    """Replace characters unsafe for filenames with underscores.

    Args:
        name: Raw string to sanitize (e.g. a feed or provider name).

    Returns:
        A filesystem-safe string containing only letters, digits,
        underscores, and hyphens.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", normalize_text(name))
