# src/rag/output_cleaner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import re

Decision = Literal["answer", "refuse"]

_LABEL_LINE_RE = re.compile(r"^\s*(Answer|Summary|Context|Question)\s*:\s*$", re.IGNORECASE)
_EXAMPLES_HEADER_RE = re.compile(r"^\s*Examples\s*:\s*$", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)
_PLACEHOLDER_LINE_RE = re.compile(r"^\s*(None|N/A)\s*$", re.IGNORECASE)
_EMPTY_BULLET_RE = re.compile(r"^\s*-\s*$")

_REFUSAL_PATTERNS = (
    "not enough information",
    "insufficient information",
    "provided context does not",
    "context does not contain",
    "cannot be determined from the context",
    "i don't have enough information",
    "i do not have enough information",
)

@dataclass(frozen=True)
class CleanResult:
    text: str
    decision: Decision
    flags: dict[str, bool]

def _remove_markdown(text: str) -> tuple[str, bool]:
    changed = False
    if "```" in text:
        new_text = _CODE_FENCE_RE.sub("", text).strip()
        if new_text != text:
            text = new_text
            changed = True
            
    if "#" in text:
        new_text = _MD_HEADING_RE.sub("", text).strip()
        if new_text != text:
            text = new_text
            changed = True

    return text, changed

def _remove_label_lines(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    
    for line in lines:
        if _LABEL_LINE_RE.match(line):
            changed = True
            continue
        
        out.append(line.rstrip())
    
    return "\n".join(out).strip(), changed

def _remove_placeholders(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    
    for line in lines:
        if _PLACEHOLDER_LINE_RE.match(line) or _EMPTY_BULLET_RE.match(line):
            changed = True
            continue
        
        out.append(line)
    
    return "\n".join(out).strip(), changed

def _normalize_whitespace(text: str) -> tuple[str, bool]:
    before = text
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text, (text != before)

def _prune_empty_examples(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        if _EXAMPLES_HEADER_RE.match(line):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines) or not lines[j].lstrip().startswith("-"):
                changed = True
                i = j
                continue

            out.append("Examples:")
            i += 1
            continue

        out.append(line)
        i += 1

    cleaned = "\n".join(out).strip()
    return cleaned, changed


def _is_refusal_response(text: str) -> bool:
    norm_text = text.lower()
    
    return any(pattern in norm_text for pattern in _REFUSAL_PATTERNS)

def _clean_text_pipeline(raw_text: str) -> tuple[str, dict[str, bool]]:
    text = (raw_text or "").strip()
    flags = {}
    
    steps = [
        (_remove_markdown, "removed_markdown"),
        (_remove_label_lines, "removed_labels"),
        (_remove_placeholders, "removed_placeholders"),
        (_normalize_whitespace, "normalized_whitespace"),
        (_prune_empty_examples, "fixed_examples"),
    ]
    
    for func, flag_name in steps:
        text, changed = func(text)
        
        if changed:
            flags[flag_name] = True
           
    return text, flags
    
def clean_rag_output(raw_text: str) -> CleanResult:
    """
    Handle output hygiene.
    """
    flags: dict[str, bool] = {
        "removed_labels": False,
        "removed_markdown": False,
        "removed_placeholders": False,
        "fixed_examples": False,
        "normalized_whitespace": False,
        "normalized_refusal": False,
    }   

    text, pipe_flags = _clean_text_pipeline(raw_text)
    flags.update(pipe_flags)
    
    decision: Decision = "answer"
    if _is_refusal_response(text) or text == "":
        decision = "refuse"

    if decision == "refuse":
        if len(text.split()) > 35 or text == "":
            text = "I don't have enough information in the provided context to answer this question."
            flags["normalized_refusal"] = True

    return CleanResult(
        text=text,
        decision=decision,
        flags=flags
    )
