# src/rag/document_identity.py
"""
R6: deterministic named-source / document-identity resolution.

Answers exactly one question: does a query contain an explicitly recognized
document/source reference, and if so, which evidence-identity requirement
does it impose?

This module is a PURE, catalog-driven function of the query string. It does
not perform retrieval, does not inspect candidates or admitted evidence, does
not call an LLM, and has no fuzzy/embedding matching. It has no wiring into
retriever.py / gating.py / coverage.py itself; its output is consumed only by
src/rag/identity_coverage.py as a post-retrieval evidence check, never used
to influence candidate retrieval or ranking.

Identity levels (see src/rag/catalog.py):
  - equivalence_group_id: whole-work evidence-equivalence (e.g. the MQTT
    v3.1.1 spec's PDF and HTML renderings -- verified interchangeable).
  - logical_document_id: identity of an authored work, with NO claim of
    evidence-equivalence across renderings/sections. Used here only for a
    document with exactly one physical rendering in the corpus, where
    matching on logical_document_id carries none of the multi-section
    over-broadening risk that a document assembled from several
    independently-titled sections (e.g. the AWS IoT Core Developer Guide)
    would introduce -- see docs_registry.json for the per-document identity
    assignments.

Configuration integrity: the alias registry (data/catalog/
document_identity_aliases.json) is validated at load time against the
*authoritative* identity namespaces declared in data/catalog/
docs_registry.json. A malformed alias file, an unreadable/malformed
authoritative registry, or an alias target that does not exist in the
correct authoritative namespace is a configuration error -- it raises
DocumentIdentityConfigError. It must never silently degrade to "no explicit
source references exist" (an empty query result), which is a legitimate
*resolution* outcome with an entirely different meaning.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias
from functools import lru_cache

import json
import re

_ALIASES_REL = Path("data/catalog/document_identity_aliases.json")
_DOCS_REGISTRY_REL = Path("data/catalog/docs_registry.json")

IdentityKind: TypeAlias = Literal["equivalence_group", "logical_document"]
ResolutionStatus: TypeAlias = Literal["resolved", "unresolved"]

_VALID_IDENTITY_KINDS: tuple[IdentityKind, ...] = ("equivalence_group", "logical_document")


class DocumentIdentityConfigError(Exception):
    """
    Raised for any R6 configuration/substrate problem: a missing, unreadable,
    or malformed alias registry; a structurally invalid alias entry; an
    invalid regex; duplicate/ambiguous alias configuration; or an alias
    target that does not exist in the authoritative docs_registry.json
    identity namespace it claims to belong to.

    This is deliberately NOT the same outcome as resolve_required_identities
    returning [] (which means "no explicit source reference in this query").
    A config error means R6 cannot be trusted to answer at all.
    """


@dataclass(frozen=True)
class AliasRule:
    requirement_id: str
    identity_kind: IdentityKind
    identity_value: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class IdentityRequirement:
    """
    One resolved-or-unresolved evidence-identity requirement extracted from a
    query. `status` is "resolved" or "unresolved". An "unresolved" entry
    means the query text matched an explicit source-reference pattern but the
    alias configuration itself could not deterministically settle on one
    identity (overlapping/conflicting alias rules) -- it is never a silent
    first-match guess.
    """
    status: ResolutionStatus
    matched_phrase: str | None = None
    identity_kind: IdentityKind | None = None
    identity_value: str | None = None
    requirement_id: str | None = None


# ---------------------------------------------------------------------------
# Authoritative identity namespaces (read from docs_registry.json)
# ---------------------------------------------------------------------------

def _load_authoritative_identity_namespaces(
    project_root: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    """
    Returns (valid_logical_document_ids, valid_equivalence_group_ids) read
    directly from the authoritative docs_registry.json. Raises
    DocumentIdentityConfigError if that registry cannot be loaded -- R6 must
    never validate alias targets against a source of truth it failed to read.

    This does not re-validate docs_registry.json's own internal schema
    beyond top-level structure (that is R1's responsibility, covered by
    tests/test_r1_metadata_model.py); it only extracts the two identity
    namespaces R6 needs, and fails loudly if the file itself is absent or
    unparseable.
    """
    path = project_root / _DOCS_REGISTRY_REL

    if not path.exists():
        raise DocumentIdentityConfigError(
            f"authoritative registry not found: {path}"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise DocumentIdentityConfigError(
            f"authoritative registry unreadable: {path}"
        ) from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DocumentIdentityConfigError(
            f"authoritative registry is malformed JSON: {path}"
        ) from e

    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise DocumentIdentityConfigError(
            f"authoritative registry has unexpected top-level structure: {path}"
        )

    logical_ids: set[str] = set()
    equivalence_ids: set[str] = set()
    for index, rule in enumerate(data["rules"]):
        lid, eid = _validate_and_extract_rule_identities(rule, index)
        if lid:
            logical_ids.add(lid)
        if eid:
            equivalence_ids.add(eid)

    return frozenset(logical_ids), frozenset(equivalence_ids)


def _validate_identity_field(tags: dict[str, Any], field_name: str, rule_index: int) -> str | None:
    """
    A field absent from `tags` is a legitimate "no claim" -- consistent with
    the existing docs_registry.json / catalog.py contract, where
    resolve_doc_tags() itself defaults missing tag keys to None and
    equivalence_group_id in particular is intentionally absent for documents
    without established whole-work evidence equivalence. A field that IS
    present but not a non-empty string is structural corruption, not a
    legitimate absence, and must raise.
    """
    if field_name not in tags:
        return None
    value = tags[field_name]
    if not isinstance(value, str) or not value.strip():
        raise DocumentIdentityConfigError(
            f"authoritative registry rule #{rule_index} has an invalid "
            f"{field_name!r} (must be a non-empty string when present): {value!r}"
        )
    return value


def _validate_and_extract_rule_identities(rule: Any, index: int) -> tuple[str | None, str | None]:
    """
    Validates one docs_registry.json rule's identity-bearing structure and
    returns (logical_document_id_or_None, equivalence_group_id_or_None).

    Raises DocumentIdentityConfigError on any structural corruption relevant
    to R6's authoritative identity model: a rule that isn't an object, or a
    'tags' value that is present but not an object. A rule with no 'tags'
    key at all is treated as making no identity claim (valid) -- this
    mirrors src/rag/catalog.py's own resolve_doc_tags(), which defaults
    rule.get("tags", {}) rather than treating an absent tags key as an
    error. Fields unrelated to identity (match, domain, doc_type, vendor,
    product, version) are intentionally NOT validated here -- they remain
    owned by src/rag/catalog.py; R6 only needs enough structure to safely
    read the two identity fields it consumes.
    """
    if not isinstance(rule, dict):
        raise DocumentIdentityConfigError(
            f"authoritative registry rule #{index} is not an object: {rule!r}"
        )

    if "tags" not in rule:
        return None, None

    tags = rule["tags"]
    if not isinstance(tags, dict):
        raise DocumentIdentityConfigError(
            f"authoritative registry rule #{index} has a malformed 'tags' "
            f"value (expected an object): {tags!r}"
        )

    logical_document_id = _validate_identity_field(tags, "logical_document_id", index)
    equivalence_group_id = _validate_identity_field(tags, "equivalence_group_id", index)
    return logical_document_id, equivalence_group_id


# ---------------------------------------------------------------------------
# Alias registry loading + validation
# ---------------------------------------------------------------------------

def _validate_and_compile_entry(entry: Any, index: int) -> tuple[str, IdentityKind, str, tuple[str, ...]]:
    """
    Structural validation of one raw alias entry. Returns
    (requirement_id, identity_kind, identity_value, raw_pattern_strings).
    Raises DocumentIdentityConfigError on any structural problem. Does not
    compile regexes or validate against the authoritative registry -- those
    are separate steps so each failure class is reported distinctly.
    """
    if not isinstance(entry, dict):
        raise DocumentIdentityConfigError(
            f"alias entry #{index} is not an object: {entry!r}"
        )

    requirement_id = entry.get("requirement_id")
    if not isinstance(requirement_id, str) or not requirement_id.strip():
        raise DocumentIdentityConfigError(
            f"alias entry #{index} has a missing/empty requirement_id"
        )

    identity_kind = entry.get("identity_kind")
    if identity_kind not in _VALID_IDENTITY_KINDS:
        raise DocumentIdentityConfigError(
            f"alias entry {requirement_id!r} has unsupported identity_kind "
            f"{identity_kind!r}; expected one of {_VALID_IDENTITY_KINDS}"
        )

    identity_value = entry.get("identity_value")
    if not isinstance(identity_value, str) or not identity_value.strip():
        raise DocumentIdentityConfigError(
            f"alias entry {requirement_id!r} has a missing/empty identity_value"
        )

    raw_patterns = entry.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise DocumentIdentityConfigError(
            f"alias entry {requirement_id!r} has a missing/empty patterns list"
        )
    pattern_strings: list[str] = []
    for p in raw_patterns:
        if not isinstance(p, str) or not p.strip():
            raise DocumentIdentityConfigError(
                f"alias entry {requirement_id!r} has a malformed pattern entry: {p!r}"
            )
        pattern_strings.append(p)

    return requirement_id, identity_kind, identity_value, tuple(pattern_strings)


def _compile_pattern(pattern_str: str, requirement_id: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern_str, re.IGNORECASE)
    except re.error as e:
        raise DocumentIdentityConfigError(
            f"alias entry {requirement_id!r} has an invalid regex pattern "
            f"{pattern_str!r}: {e}"
        ) from e


def _load_and_validate_alias_rules(project_root: Path) -> tuple[AliasRule, ...]:
    alias_path = project_root / _ALIASES_REL

    if not alias_path.exists():
        raise DocumentIdentityConfigError(
            f"alias registry not found: {alias_path}"
        )
    try:
        raw = alias_path.read_text(encoding="utf-8")
    except OSError as e:
        raise DocumentIdentityConfigError(
            f"alias registry unreadable: {alias_path}"
        ) from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DocumentIdentityConfigError(
            f"alias registry is malformed JSON: {alias_path}"
        ) from e

    if not isinstance(data, dict) or not isinstance(data.get("aliases"), list):
        raise DocumentIdentityConfigError(
            f"alias registry has unexpected top-level structure: {alias_path}"
        )

    # Structural validation of every entry.
    parsed: list[tuple[str, IdentityKind, str, tuple[str, ...]]] = [
        _validate_and_compile_entry(entry, i) for i, entry in enumerate(data["aliases"])
    ]

    # Duplicate requirement_id -> reject.
    seen_ids: set[str] = set()
    for requirement_id, _, _, _ in parsed:
        if requirement_id in seen_ids:
            raise DocumentIdentityConfigError(
                f"duplicate requirement_id in alias registry: {requirement_id!r}"
            )
        seen_ids.add(requirement_id)

    # Exact duplicate pattern string anywhere in the registry -> reject.
    # This applies uniformly whether the duplicate targets the same identity
    # (redundant configuration) or a different identity (a genuine
    # configuration hazard, since it would make any query matching that
    # phrase permanently ambiguous). Rejecting both at load time keeps the
    # rule single and simple, and forces the alias file to stay unambiguous
    # by construction rather than relying on runtime overlap resolution to
    # paper over avoidable duplication.
    pattern_owner: dict[str, str] = {}
    for requirement_id, _, _, pattern_strings in parsed:
        for p in pattern_strings:
            if p in pattern_owner:
                raise DocumentIdentityConfigError(
                    f"duplicate exact pattern {p!r} in alias registry, used by "
                    f"both {pattern_owner[p]!r} and {requirement_id!r}"
                )
            pattern_owner[p] = requirement_id

    # Every alias target must exist in the authoritative registry's matching
    # identity namespace -- an alias may never assert an identity the
    # registry itself does not declare.
    valid_logical_ids, valid_equivalence_ids = _load_authoritative_identity_namespaces(project_root)

    rules: list[AliasRule] = []
    for requirement_id, identity_kind, identity_value, pattern_strings in parsed:
        namespace = valid_equivalence_ids if identity_kind == "equivalence_group" else valid_logical_ids
        if identity_value not in namespace:
            raise DocumentIdentityConfigError(
                f"alias entry {requirement_id!r} targets {identity_kind}="
                f"{identity_value!r}, which does not exist in the authoritative "
                f"{identity_kind}_id namespace declared by docs_registry.json"
            )

        compiled = tuple(_compile_pattern(p, requirement_id) for p in pattern_strings)
        rules.append(AliasRule(
            requirement_id=requirement_id,
            identity_kind=identity_kind,
            identity_value=identity_value,
            patterns=compiled,
        ))

    return tuple(rules)


@lru_cache(maxsize=8)
def _load_and_validate_alias_rules_cached(
    alias_abs_path: Path,
    alias_mtime_ns: int,
    registry_abs_path: Path,
    registry_mtime_ns: int,
    project_root: Path,
) -> tuple[AliasRule, ...]:
    # alias_abs_path/mtime and registry_abs_path/mtime are cache-key
    # components only (both files' identity must be part of the key, since
    # validation depends on both); the actual loading re-derives paths from
    # project_root for simplicity and to avoid duplicating path-join logic.
    return _load_and_validate_alias_rules(project_root)


def _load_alias_rules(project_root: Path) -> tuple[AliasRule, ...]:
    alias_path = project_root / _ALIASES_REL
    registry_path = project_root / _DOCS_REGISTRY_REL

    # Resolve mtimes for cache-key purposes. If either file is missing, fall
    # through to the uncached loader, which raises a precise
    # DocumentIdentityConfigError for that exact condition rather than a
    # generic stat failure here.
    try:
        alias_mtime_ns = alias_path.stat().st_mtime_ns
    except OSError:
        return _load_and_validate_alias_rules(project_root)

    try:
        registry_mtime_ns = registry_path.stat().st_mtime_ns
    except OSError:
        return _load_and_validate_alias_rules(project_root)

    return _load_and_validate_alias_rules_cached(
        alias_path.resolve(),
        alias_mtime_ns,
        registry_path.resolve(),
        registry_mtime_ns,
        project_root,
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve_against_rules(query: str, rules: tuple[AliasRule, ...]) -> list[IdentityRequirement]:
    """
    Pure matching core, parameterized on an explicit rule set so it can be
    exercised directly (e.g. with a deliberately conflicting synthetic rule
    set) without touching the real alias registry file.
    """
    q = query or ""
    if not q.strip() or not rules:
        return []

    # Every (start, end, matched_text, rule) hit across all rules/patterns.
    raw_hits: list[tuple[int, int, str, AliasRule]] = []
    for rule in rules:
        for pattern in rule.patterns:
            for m in pattern.finditer(q):
                raw_hits.append((m.start(), m.end(), m.group(0), rule))

    if not raw_hits:
        return []

    # Group hits whose spans overlap (transitively) -- these compete for the
    # same mention. Non-overlapping hits are separate, independent
    # requirements. Sorting by start position (not rule declaration order)
    # makes grouping independent of alias-registry ordering.
    raw_hits.sort(key=lambda h: h[0])
    groups: list[list[tuple[int, int, str, AliasRule]]] = [[raw_hits[0]]]
    group_end = raw_hits[0][1]
    for hit in raw_hits[1:]:
        if hit[0] < group_end:
            groups[-1].append(hit)
            group_end = max(group_end, hit[1])
        else:
            groups.append([hit])
            group_end = hit[1]

    requirements: list[IdentityRequirement] = []
    seen_resolved: set[tuple[str, str]] = set()

    for group in groups:
        distinct_identities = {(h[3].identity_kind, h[3].identity_value) for h in group}

        if len(distinct_identities) > 1:
            # The same/overlapping mention matches more than one distinct
            # identity: the alias configuration itself is ambiguous here.
            # Never arbitrarily pick the first match.
            requirements.append(IdentityRequirement(
                status="unresolved",
                matched_phrase=group[0][2],
            ))
            continue

        rule = group[0][3]
        key = (rule.identity_kind, rule.identity_value)
        if key in seen_resolved:
            # Same identity already resolved from an earlier, separate
            # mention in this query -- do not emit a redundant duplicate
            # requirement.
            continue
        seen_resolved.add(key)

        requirements.append(IdentityRequirement(
            status="resolved",
            matched_phrase=group[0][2],
            identity_kind=rule.identity_kind,
            identity_value=rule.identity_value,
            requirement_id=rule.requirement_id,
        ))

    return requirements


def resolve_required_identities(
    query: str,
    *,
    project_root: Path,
) -> list[IdentityRequirement]:
    """
    Resolve explicit source/document references in `query` against the
    catalog-owned, authoritative-registry-validated alias registry.

    Activation is defined entirely by the curated alias patterns: a query
    with no matching phrase returns []. There is no generic "this looks like
    it might be naming a document" heuristic -- only exact, deterministic
    regex matches against explicitly authored, validated aliases activate a
    requirement.

    Raises DocumentIdentityConfigError if the alias registry or the
    authoritative docs_registry.json cannot be loaded or fails validation --
    this is never conflated with a legitimate empty ([]) resolution result.
    """
    rules = _load_alias_rules(project_root)
    return _resolve_against_rules(query, rules)
