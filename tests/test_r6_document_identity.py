import json
import os
import re
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.rag.catalog import resolve_doc_tags
from src.rag.document_identity import (
    AliasRule,
    DocumentIdentityConfigError,
    resolve_required_identities,
    _resolve_against_rules,
    _load_alias_rules,
    _ALIASES_REL,
    _DOCS_REGISTRY_REL,
)


def _resolve(query: str):
    return resolve_required_identities(query, project_root=PROJECT_ROOT)


# ============================================================================
# Synthetic-project-root helpers (isolated tmp_path trees; never touch real
# repository data).
# ============================================================================

def _write_registry(root: Path, rules: list[dict]) -> Path:
    catalog_dir = root / "data" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "docs_registry.json"
    path.write_text(json.dumps({"rules": rules}), encoding="utf-8")
    return path


def _write_aliases_raw(root: Path, raw_text: str) -> Path:
    catalog_dir = root / "data" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "document_identity_aliases.json"
    path.write_text(raw_text, encoding="utf-8")
    return path


def _write_aliases(root: Path, aliases: list[dict]) -> Path:
    return _write_aliases_raw(root, json.dumps({"aliases": aliases}))


def _bump_mtime(path: Path, mtime_ns: int) -> None:
    os.utime(path, ns=(mtime_ns, mtime_ns))


_VALID_MIN_REGISTRY = [
    {"match": {"filename": "doc-a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
]
_VALID_MIN_ALIASES = [
    {
        "requirement_id": "doc_a_reference",
        "identity_kind": "logical_document",
        "identity_value": "doc_a",
        "patterns": [r"\bthe\s+doc\s+a\b"],
    }
]


# ============================================================================
# Bare topic mentions must NOT activate; explicit references resolve
# ============================================================================

def test_the_mqtt_specification_itself_resolves():
    reqs = _resolve("the MQTT specification itself")
    assert len(reqs) == 1
    assert reqs[0].status == "resolved"
    assert reqs[0].identity_kind == "equivalence_group"
    assert reqs[0].identity_value == "mqtt_v3_1_1_spec"


def test_the_mqtt_protocol_specification_resolves():
    reqs = _resolve("the MQTT protocol specification")
    assert reqs[0].identity_kind == "equivalence_group"
    assert reqs[0].identity_value == "mqtt_v3_1_1_spec"


def test_how_does_mqtt_work_does_not_resolve():
    assert _resolve("How does MQTT work?") == []


def test_bare_mqtt_does_not_resolve():
    assert _resolve("MQTT") == []


def test_bare_aws_iot_jobs_does_not_resolve():
    assert _resolve("AWS IoT jobs") == []


def test_bare_iot_connectivity_does_not_resolve():
    assert _resolve("IoT connectivity") == []


def test_the_general_iot_whitepaper_resolves_as_logical_document():
    reqs = _resolve("the general IoT whitepaper")
    assert len(reqs) == 1
    assert reqs[0].identity_kind == "logical_document"
    assert reqs[0].identity_value == "general_iot_whitepaper_2018"


def test_general_iot_whitepaper_white_paper_two_words_variant_resolves():
    reqs = _resolve("the general IoT white paper")
    assert reqs[0].identity_kind == "logical_document"
    assert reqs[0].identity_value == "general_iot_whitepaper_2018"


# ============================================================================
# Historical query regression
# ============================================================================

_XQ_001_QUERY = (
    "How does AWS IoT Core's guidance on MQTT topic design relate to the "
    "wildcard rules defined in the MQTT specification itself?"
)
_XQ_004_QUERY = (
    "What common ground exists between the general IoT whitepaper's "
    "discussion of connectivity and the MQTT protocol specification?"
)
_CQ_003_QUERY = (
    "What is the general relationship between a cloud IoT platform and the "
    "devices it manages, based on this corpus?"
)


def test_xq001_resolves_only_mqtt_source_requirement():
    reqs = _resolve(_XQ_001_QUERY)
    assert len(reqs) == 1
    assert reqs[0].identity_kind == "equivalence_group"
    assert reqs[0].identity_value == "mqtt_v3_1_1_spec"


def test_xq004_resolves_two_independent_source_requirements():
    reqs = _resolve(_XQ_004_QUERY)
    assert len(reqs) == 2
    resolved = {(r.identity_kind, r.identity_value) for r in reqs}
    assert resolved == {
        ("logical_document", "general_iot_whitepaper_2018"),
        ("equivalence_group", "mqtt_v3_1_1_spec"),
    }


def test_cq003_resolves_no_requirements():
    assert _resolve(_CQ_003_QUERY) == []


def test_multiple_independent_references_produce_multiple_requirements_not_ambiguity():
    query = "Compare the MQTT specification itself against the general IoT whitepaper's findings."
    reqs = _resolve(query)
    assert len(reqs) == 2
    assert all(r.status == "resolved" for r in reqs)


# ============================================================================
# Configuration integrity
# ============================================================================

def test_A_real_alias_registry_loads_successfully():
    rules = _load_alias_rules(PROJECT_ROOT)
    assert len(rules) == 2
    kinds_values = {(r.identity_kind, r.identity_value) for r in rules}
    assert ("equivalence_group", "mqtt_v3_1_1_spec") in kinds_values
    assert ("logical_document", "general_iot_whitepaper_2018") in kinds_values


def test_B_logical_document_target_exists_in_authoritative_namespace():
    tags = resolve_doc_tags(project_root=PROJECT_ROOT, source="white-paper-iot-july-2018.pdf")
    assert tags.logical_document_id == "general_iot_whitepaper_2018"


def test_C_equivalence_group_target_exists_in_authoritative_namespace():
    tags = resolve_doc_tags(project_root=PROJECT_ROOT, source="mqtt-v3.1.1-os.pdf")
    assert tags.equivalence_group_id == "mqtt_v3_1_1_spec"


def test_D_nonexistent_logical_document_target_raises_config_error(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)  # only declares "doc_a"
    _write_aliases(tmp_path, [
        {
            "requirement_id": "bad_ref",
            "identity_kind": "logical_document",
            "identity_value": "doc_that_does_not_exist",
            "patterns": [r"\bmissing\s+doc\b"],
        }
    ])
    with pytest.raises(DocumentIdentityConfigError, match="does not exist"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_E_nonexistent_equivalence_group_target_raises_config_error(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)  # no equivalence_group_id declared at all
    _write_aliases(tmp_path, [
        {
            "requirement_id": "bad_ref",
            "identity_kind": "equivalence_group",
            "identity_value": "group_that_does_not_exist",
            "patterns": [r"\bmissing\s+group\b"],
        }
    ])
    with pytest.raises(DocumentIdentityConfigError, match="does not exist"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_F_cross_namespace_misuse_raises_config_error(tmp_path):
    # doc_a exists ONLY as a logical_document_id (mirrors the real
    # general_iot_whitepaper_2018 case: no equivalence_group_id declared).
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases(tmp_path, [
        {
            "requirement_id": "cross_namespace_ref",
            "identity_kind": "equivalence_group",  # WRONG: doc_a has no equivalence group
            "identity_value": "doc_a",
            "patterns": [r"\bthe\s+doc\s+a\b"],
        }
    ])
    with pytest.raises(DocumentIdentityConfigError, match="does not exist"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_G_missing_alias_registry_raises_clearly(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    # No aliases file written at all.
    with pytest.raises(DocumentIdentityConfigError, match="not found"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_H_malformed_alias_json_raises_clearly(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases_raw(tmp_path, "{not valid json,,,")
    with pytest.raises(DocumentIdentityConfigError, match="malformed JSON"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_I_structurally_invalid_alias_entry_raises_clearly(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases(tmp_path, [
        {"requirement_id": "no_patterns_field", "identity_kind": "logical_document", "identity_value": "doc_a"}
    ])
    with pytest.raises(DocumentIdentityConfigError, match="patterns"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_I_wrong_top_level_structure_raises_clearly(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases_raw(tmp_path, json.dumps({"not_aliases_key": []}))
    with pytest.raises(DocumentIdentityConfigError, match="structure"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_J_invalid_regex_raises_clearly(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases(tmp_path, [
        {
            "requirement_id": "bad_regex",
            "identity_kind": "logical_document",
            "identity_value": "doc_a",
            "patterns": ["(unclosed"],
        }
    ])
    with pytest.raises(DocumentIdentityConfigError, match="invalid regex"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_K_duplicate_requirement_id_is_rejected(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases(tmp_path, [
        {"requirement_id": "dup", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bfoo\b"]},
        {"requirement_id": "dup", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bbar\b"]},
    ])
    with pytest.raises(DocumentIdentityConfigError, match="duplicate requirement_id"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_L_conflicting_exact_duplicate_pattern_different_identity_is_rejected(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        {"match": {"filename": "b.pdf"}, "tags": {"logical_document_id": "doc_b"}},
    ])
    _write_aliases(tmp_path, [
        {"requirement_id": "ref_a", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+doc\b"]},
        {"requirement_id": "ref_b", "identity_kind": "logical_document", "identity_value": "doc_b", "patterns": [r"\bthe\s+doc\b"]},
    ])
    with pytest.raises(DocumentIdentityConfigError, match="duplicate exact pattern"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_M_same_target_duplicate_pattern_is_also_rejected_deterministically(tmp_path):
    # Policy choice: exact duplicate patterns are rejected at load time
    # uniformly, whether they target the same or different identities --
    # not silently deduplicated. Pinned here explicitly.
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _write_aliases(tmp_path, [
        {"requirement_id": "ref_1", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+doc\b"]},
        {"requirement_id": "ref_2", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+doc\b"]},
    ])
    with pytest.raises(DocumentIdentityConfigError, match="duplicate exact pattern"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_N_missing_authoritative_registry_prevents_resolver_configuration(tmp_path):
    # Valid alias file, but no docs_registry.json at all.
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="authoritative registry not found"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_N_malformed_authoritative_registry_prevents_resolver_configuration(tmp_path):
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    (tmp_path / "data" / "catalog").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "catalog" / "docs_registry.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentIdentityConfigError, match="malformed JSON"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_O_alias_file_change_invalidates_cache(tmp_path):
    _write_registry(tmp_path, _VALID_MIN_REGISTRY)
    _bump_mtime(tmp_path / "data" / "catalog" / "docs_registry.json", 1_000_000_000)

    aliases_path = _write_aliases(tmp_path, [
        {"requirement_id": "ref_1", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\boriginal\s+phrase\b"]},
    ])
    _bump_mtime(aliases_path, 1_000_000_000)

    reqs1 = resolve_required_identities("original phrase", project_root=tmp_path)
    assert len(reqs1) == 1

    # Rewrite the alias file with a different pattern and a later mtime.
    aliases_path = _write_aliases(tmp_path, [
        {"requirement_id": "ref_1", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bnew\s+phrase\b"]},
    ])
    _bump_mtime(aliases_path, 2_000_000_000)

    reqs2 = resolve_required_identities("original phrase", project_root=tmp_path)
    assert reqs2 == [], "stale cached alias rules were used instead of reloading"

    reqs3 = resolve_required_identities("new phrase", project_root=tmp_path)
    assert len(reqs3) == 1


def test_P_registry_identity_change_invalidates_and_revalidates_cache(tmp_path):
    registry_path = _write_registry(tmp_path, _VALID_MIN_REGISTRY)  # declares "doc_a"
    _bump_mtime(registry_path, 1_000_000_000)

    aliases_path = _write_aliases(tmp_path, _VALID_MIN_ALIASES)  # targets "doc_a"
    _bump_mtime(aliases_path, 1_000_000_000)

    reqs1 = resolve_required_identities("the doc a", project_root=tmp_path)
    assert len(reqs1) == 1

    # Rename the authoritative identity underneath the (unchanged) alias
    # file. Only the registry's mtime changes.
    registry_path = _write_registry(tmp_path, [
        {"match": {"filename": "doc-a.pdf"}, "tags": {"logical_document_id": "doc_a_renamed"}},
    ])
    _bump_mtime(registry_path, 2_000_000_000)

    with pytest.raises(DocumentIdentityConfigError, match="does not exist"):
        resolve_required_identities("the doc a", project_root=tmp_path)


# ============================================================================
# Authoritative-rule structural validation: malformed rules inside
# docs_registry.json must never be silently skipped while a
# partially-corrupted registry remains usable for R6.
# ============================================================================

def test_A_non_object_rule_raises_config_error(tmp_path):
    _write_registry(tmp_path, ["not_a_rule_object"])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="not an object"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_B_malformed_non_object_tags_raises_config_error(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": ["not", "a", "dict"]},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="malformed 'tags'"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_C_missing_tags_key_is_valid_no_identity_claim(tmp_path):
    # A rule with no 'tags' key at all is a legitimate shape under the
    # existing docs_registry.json / catalog.py contract -- resolve_doc_tags()
    # itself defaults rule.get("tags", {}) rather than treating an absent
    # tags key as an error. Such a rule simply makes no identity claim; it
    # must not fail registry loading, and the alias registry must still
    # validate successfully against whatever OTHER rules do carry identities.
    _write_registry(tmp_path, [
        {"match": {"filename": "no-tags.pdf"}},  # no "tags" key at all
        {"match": {"filename": "doc-a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    reqs = resolve_required_identities("the doc a", project_root=tmp_path)
    assert len(reqs) == 1
    assert reqs[0].identity_value == "doc_a"


def test_D_logical_document_id_wrong_type_raises_config_error(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": 123}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="logical_document_id"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_E_logical_document_id_empty_string_raises_config_error(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": ""}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="logical_document_id"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_F_logical_document_id_whitespace_only_raises_config_error(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "   "}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="logical_document_id"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_G_equivalence_group_id_wrong_type_raises_config_error(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a", "equivalence_group_id": 456}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="equivalence_group_id"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_H_equivalence_group_id_empty_or_whitespace_raises_config_error(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a", "equivalence_group_id": "  "}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="equivalence_group_id"):
        resolve_required_identities("anything", project_root=tmp_path)


def test_I_absent_equivalence_group_id_remains_valid(tmp_path):
    # Mirrors the real R1 model: a document with no established whole-work
    # evidence equivalence simply has no equivalence_group_id key at all.
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    reqs = resolve_required_identities("the doc a", project_root=tmp_path)
    assert len(reqs) == 1
    assert reqs[0].identity_value == "doc_a"


def test_J_mixed_valid_and_structurally_malformed_rule_fails_entirely(tmp_path):
    # One valid rule + one structurally malformed rule (not an object) must
    # fail the WHOLE authoritative load -- no partial-success extraction from
    # the valid rule while silently ignoring the corrupted one.
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        "this_is_not_a_rule_object",
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="not an object"):
        resolve_required_identities("the doc a", project_root=tmp_path)


def test_K_mixed_valid_and_invalid_identity_bearing_rule_fails_entirely(tmp_path):
    # One valid logical identity + one rule with a malformed identity value
    # must fail the whole namespace load, even though the first rule alone
    # would have been perfectly sufficient to satisfy the alias registry.
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        {"match": {"filename": "b.pdf"}, "tags": {"logical_document_id": ""}},
    ])
    _write_aliases(tmp_path, _VALID_MIN_ALIASES)
    with pytest.raises(DocumentIdentityConfigError, match="logical_document_id"):
        resolve_required_identities("the doc a", project_root=tmp_path)


def test_L_current_real_docs_registry_still_validates():
    # Uses the authoritative production registry as-is: it must load and
    # validate cleanly under the structural checks with no synthetic
    # mutation of repository registry data required.
    reqs = _resolve("the MQTT specification itself")
    assert len(reqs) == 1
    assert reqs[0].identity_value == "mqtt_v3_1_1_spec"


# ============================================================================
# Ambiguity edge cases (algorithm pinned, not redesigned, by these tests)
# ============================================================================

def test_overlapping_same_identity_patterns_resolve_to_one_requirement():
    rules = (
        AliasRule("r1", "logical_document", "doc_a", (re.compile(r"\bthe\s+doc\b", re.I),)),
        AliasRule("r2", "logical_document", "doc_a", (re.compile(r"\bdoc\s+itself\b", re.I),)),
    )
    reqs = _resolve_against_rules("the doc itself", rules)
    assert len(reqs) == 1
    assert reqs[0].status == "resolved"
    assert reqs[0].identity_value == "doc_a"


def test_nested_same_identity_patterns_resolve_to_one_requirement():
    rules = (
        AliasRule("r1", "logical_document", "doc_a", (re.compile(r"\bthe\s+special\s+doc\b", re.I),)),
        AliasRule("r2", "logical_document", "doc_a", (re.compile(r"\bspecial\b", re.I),)),
    )
    reqs = _resolve_against_rules("the special doc", rules)
    assert len(reqs) == 1
    assert reqs[0].status == "resolved"
    assert reqs[0].identity_value == "doc_a"


def test_nested_different_identity_patterns_are_unresolved():
    rules = (
        AliasRule("r1", "logical_document", "doc_a", (re.compile(r"\bthe\s+special\s+doc\b", re.I),)),
        AliasRule("r2", "equivalence_group", "group_b", (re.compile(r"\bspecial\b", re.I),)),
    )
    reqs = _resolve_against_rules("the special doc", rules)
    assert len(reqs) == 1
    assert reqs[0].status == "unresolved"


def test_adjacent_non_overlapping_references_are_independent_requirements():
    rules = (
        AliasRule("r1", "logical_document", "doc_a", (re.compile(r"\bdoc\s*a\b", re.I),)),
        AliasRule("r2", "logical_document", "doc_b", (re.compile(r"\bdoc\s*b\b", re.I),)),
    )
    reqs = _resolve_against_rules("doc a doc b", rules)
    assert len(reqs) == 2
    assert {r.identity_value for r in reqs} == {"doc_a", "doc_b"}
    assert all(r.status == "resolved" for r in reqs)


def test_three_way_transitive_overlap_forms_one_conflict_cluster():
    # query = "abcdefghijklmnopqrst" (20 distinct-position characters)
    # A = "abcdefghij" -> span [0,10)
    # B = "fghijklmno" -> span [5,15)   (overlaps A)
    # C = "mnopqrst"   -> span [12,20)  (overlaps B, does NOT overlap A directly)
    query = "abcdefghijklmnopqrst"
    rules = (
        AliasRule("rA", "logical_document", "doc_a", (re.compile(re.escape("abcdefghij")),)),
        AliasRule("rB", "logical_document", "doc_b", (re.compile(re.escape("fghijklmno")),)),
        AliasRule("rC", "logical_document", "doc_c", (re.compile(re.escape("mnopqrst")),)),
    )
    reqs = _resolve_against_rules(query, rules)
    # One deterministic conflict cluster, not three separate unresolved entries.
    assert len(reqs) == 1
    assert reqs[0].status == "unresolved"


def test_no_rule_order_dependent_outcome():
    query = "the special doc"
    rules_forward = (
        AliasRule("r1", "logical_document", "doc_a", (re.compile(r"\bthe\s+special\s+doc\b", re.I),)),
        AliasRule("r2", "equivalence_group", "group_b", (re.compile(r"\bspecial\b", re.I),)),
    )
    rules_reversed = tuple(reversed(rules_forward))

    reqs_forward = _resolve_against_rules(query, rules_forward)
    reqs_reversed = _resolve_against_rules(query, rules_reversed)

    to_tuple = lambda reqs: [(r.status, r.identity_kind, r.identity_value) for r in reqs]
    assert to_tuple(reqs_forward) == to_tuple(reqs_reversed)


def test_overlapping_alias_configuration_resolves_as_unresolved_not_first_match():
    conflicting_rules = (
        AliasRule("rule_a", "equivalence_group", "mqtt_v3_1_1_spec", (re.compile(r"\bthe\s+mqtt\s+specification\b", re.I),)),
        AliasRule("rule_b", "equivalence_group", "mqtt_v5_spec", (re.compile(r"\bmqtt\s+specification\b", re.I),)),
    )
    reqs = _resolve_against_rules("Tell me about the MQTT specification.", conflicting_rules)
    assert len(reqs) == 1
    assert reqs[0].status == "unresolved"
    assert reqs[0].identity_kind is None
    assert reqs[0].identity_value is None


def test_non_overlapping_hits_from_different_rules_are_not_treated_as_conflicting():
    rules = (
        AliasRule("rule_a", "equivalence_group", "mqtt_v3_1_1_spec", (re.compile(r"\bthe\s+mqtt\s+spec\b", re.I),)),
        AliasRule("rule_b", "logical_document", "general_iot_whitepaper_2018", (re.compile(r"\bthe\s+whitepaper\b", re.I),)),
    )
    reqs = _resolve_against_rules("Compare the MQTT spec against the whitepaper.", rules)
    assert len(reqs) == 2
    assert all(r.status == "resolved" for r in reqs)


# ============================================================================
# Compatibility with the R1 identity model
# ============================================================================

def test_general_iot_whitepaper_has_no_equivalence_group_in_r1_and_r6_does_not_assume_one():
    tags = resolve_doc_tags(project_root=PROJECT_ROOT, source="white-paper-iot-july-2018.pdf")
    assert tags.equivalence_group_id is None
    assert tags.logical_document_id == "general_iot_whitepaper_2018"

    reqs = _resolve("the general IoT whitepaper")
    assert reqs[0].identity_kind == "logical_document"


def test_r6_does_not_resolve_aws_developer_guide_as_a_whole_file_requirement():
    aliases_path = PROJECT_ROOT / _ALIASES_REL
    data = json.loads(aliases_path.read_text(encoding="utf-8"))
    targeted_values = {a["identity_value"] for a in data.get("aliases", [])}
    assert "aws_iot_core_developer_guide" not in targeted_values


# ============================================================================
# R6 is not directly coupled into retrieval/gating/ranking and does not
# itself alter retrieval behavior. Its output is consumed only through the
# separate identity-coverage path (src/rag/identity_coverage.py) -- none of
# chain.py, retriever.py, gating.py, or coverage.py reference this module
# directly.
# ============================================================================

def test_r6_module_not_directly_referenced_by_chain_retriever_gating_coverage():
    for rel in ("src/rag/chain.py", "src/rag/retriever.py", "src/rag/gating.py", "src/rag/coverage.py"):
        source = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert "document_identity" not in source, f"{rel} unexpectedly references document_identity"


def test_resolving_identities_does_not_touch_retrieval_or_docs():
    import inspect
    from src.rag import document_identity as di

    sig = inspect.signature(di.resolve_required_identities)
    assert list(sig.parameters) == ["query", "project_root"]
