# src/rag/identity_coverage.py
"""
R7: deterministic post-retrieval evidence-identity coverage backstop.

Purpose: if R6 (src/rag/document_identity.py) identifies one or more
explicit source/document requirements in the query, a successful answer path
must not continue unless the ALREADY-ADMITTED evidence -- produced entirely
by the existing, unmodified retrieval/gating/ambiguity/coverage pipeline --
satisfies those requirements.

R7 is NOT retrieval expansion, reranking, query rewriting, LLM source
interpretation, or a mechanism for manufacturing PASS by looking anything up
in the corpus. It only inspects the metadata already attached to documents
the pipeline already admitted.

Satisfaction is checked exclusively against the authoritative R1 identity
metadata fields (`logical_document_id`, `equivalence_group_id`) already
present on admitted `Document.metadata`. No inference from filename, domain,
doc_type, product, tag_signature, or entities is performed -- those are
different, coarser concepts (see src/rag/catalog.py) and must never be
treated as satisfying a named-source requirement.

DocumentIdentityConfigError (raised by R6's resolver on invalid alias/
authoritative-registry configuration) is intentionally NOT caught here. It
must propagate as a configuration/runtime error, never be reinterpreted as a
REFUSE outcome -- those are different failure classes with different
meanings (see docstring on DocumentIdentityConfigError).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.schemas import RetrievalState, RetrievalStatusEnum
from src.rag.policy import REFUSAL_TEXT
from src.rag.document_identity import IdentityRequirement, resolve_required_identities


@dataclass(frozen=True)
class IdentityCoverageConfig:
    enabled: bool = True
    project_root: Path = Path(".")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None, *, project_root: Path) -> "IdentityCoverageConfig":
        if not cfg:
            return cls(enabled=True, project_root=project_root)
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            project_root=project_root,
        )


def _refuse(state: RetrievalState, *, reason: str) -> RetrievalState:
    # Mirrors the established refusal-state shape used by
    # src/rag/coverage.py::_refuse and src/rag/policy.py::refuse_if_no_docs
    # (same fields, same REFUSAL_TEXT) -- no parallel/incompatible refusal
    # mechanism is introduced.
    return {
        **state,
        "skip_llm": True,
        "status": RetrievalStatusEnum.REFUSE.value,
        "docs": [],
        "context": "",
        "answer": REFUSAL_TEXT,
        "refusal_reason": reason,
    }


def _requirement_satisfied(requirement: IdentityRequirement, docs: list[Document]) -> bool:
    """
    Typed satisfaction predicate. The identity-kind namespaces are never
    cross-satisfied: a logical_document requirement only ever checks
    metadata["logical_document_id"]; an equivalence_group requirement only
    ever checks metadata["equivalence_group_id"]. A document lacking the
    relevant metadata field simply does not satisfy that requirement -- no
    field is inferred or fabricated.
    """
    field_name = {
        "logical_document": "logical_document_id",
        "equivalence_group": "equivalence_group_id",
    }.get(requirement.identity_kind)

    if field_name is None:
        return False

    for doc in docs:
        meta = doc.metadata or {}
        if meta.get(field_name) == requirement.identity_value:
            return True
    return False


def identity_coverage_gate(state: RetrievalState, *, cfg: IdentityCoverageConfig) -> RetrievalState:
    """
    Post-retrieval evidence-identity coverage check.

    True no-op (returns `state` unchanged) when:
      - the gate is disabled; or
      - the chain has already short-circuited (skip_llm); or
      - R6 resolves no requirements for this query (the overwhelming
        majority of ordinary topical queries) -- REGARDLESS of whether
        `docs` is empty; that generic empty-evidence case remains the
        downstream refuse_if_no_docs guard's responsibility; or
      - every resolved requirement is satisfied by the admitted evidence.

    Refuses (via the shared _refuse shape) when an explicit source contract
    exists (requirements is non-empty) and:
      - R6 returns one or more UNRESOLVED requirements (an explicit source
        reference the alias configuration could not deterministically
        settle -- fail closed, never treated as "no requirement"); or
      - one or more RESOLVED requirements are not satisfied by admitted
        evidence -- including the case where `docs` is empty, which is
        simply one possible (unsatisfying) evidence set once an explicit
        contract exists, not a special early-return.

    Never reorders, adds, or drops documents; never changes scores; never
    rewrites metadata; never alters the query; never touches status when
    every requirement is already met.

    Raises DocumentIdentityConfigError if the R6 alias/authoritative
    registry configuration itself is invalid -- this is deliberately NOT
    caught here.
    """
    if not cfg.enabled:
        return state
    if state.get("skip_llm", False):
        return state

    query = state.get("input", "")
    requirements = resolve_required_identities(query, project_root=cfg.project_root)

    if not requirements:
        # No explicit source contract exists for this query. An empty admitted
        # -evidence set in this case is NOT R7's concern -- it remains the
        # downstream refuse_if_no_docs guard's responsibility to produce the
        # generic "No relevant documents found" refusal. R7 stays a complete
        # no-op here regardless of `docs`.
        return state

    # From here on, an explicit source contract exists. docs=[] is simply one
    # possible evidence set to evaluate each requirement against -- it is NOT
    # a special early-return condition. An unresolved reference, or a
    # resolved-but-unsatisfied requirement, must still classify and REFUSE
    # with its specific, typed reason rather than falling through to the
    # generic empty-doc guard, which would erase that reason.
    docs = state.get("docs", [])

    unresolved = [r for r in requirements if r.status == "unresolved"]
    if unresolved:
        return _refuse(
            state,
            reason="Explicit source reference could not be resolved unambiguously",
        )

    missing = [r for r in requirements if not _requirement_satisfied(r, docs)]
    if missing:
        missing_labels = ", ".join(
            f"{r.identity_kind}:{r.identity_value}" for r in missing
        )
        return _refuse(
            state,
            reason=f"Missing required source-identity coverage: {missing_labels}",
        )

    return state
