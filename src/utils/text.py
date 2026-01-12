from __future__ import annotations
import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def clean_text(text: str) -> str:
    """
    Sanitize text for embedding / tokenization stability.
    Handles common PDF extraction artifacts:
    - BOM / zero-width no-break space (U+FEFF)
    - ASCII control characters (except \\t, \\n, \\r)
    - Invalid unicode sequences (best-effort)
    - Excess whitespace normalization
    """
    if not isinstance(text, str):
        raise TypeError(f"sanitize_text expects str, got {type(text)!r}")

    text = text.replace("\ufeff", "")   # Remove BOM / zero-width no-break space
    text = _CONTROL_CHARS.sub(" ", text)    # Remove problematic control chars (keep \t \n \r)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")   # Force valid UTF-8 (drops problematic code points)
    text = " ".join(text.split())   # Normalize whitespace

    return text
