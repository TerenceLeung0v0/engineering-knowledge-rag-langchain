from typing import TypedDict, NotRequired, TypeAlias, Literal
from enum import Enum
from dataclasses import dataclass

from langchain_core.documents import Document

RetrievalStatus: TypeAlias = Literal[
    "ok",
    "refuse",
    "ambiguous"
]

class RetrievalStatusEnum(str, Enum):
    OK = "ok"
    REFUSE = "refuse"
    AMBIGUOUS = "ambiguous"

@dataclass(frozen=True)
class SourceRef:
    filename: str
    page: int | str

@dataclass(frozen=True)
class ScoredDocument:
    doc: Document
    score: float    # L2 disatance

@dataclass(frozen=True)
class RetrievalOption:
    option_id: int
    docs: list[Document]
    sources: list[SourceRef]
    best_l2: float

class RetrievalState(TypedDict):
    # Chain entry
    input: str
    
    # Later states
    docs: NotRequired[list[Document]]
    context: NotRequired[str]
    answer: NotRequired[str]
    
    status: NotRequired[RetrievalStatus]
    refusal_reason: NotRequired[str | None]

    options: NotRequired[list[RetrievalOption]]
    selected_option: NotRequired[int | None]

    skip_llm: NotRequired[bool]
