import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import PROJECT_ROOT, RETRIEVAL_CONFIG
from src.rag.catalog import enrich_metadata
from src.rag.document_identity import DocumentIdentityConfigError, IdentityRequirement, resolve_required_identities
from src.rag.identity_coverage import (
    IdentityCoverageConfig,
    identity_coverage_gate,
    _requirement_satisfied,
)
from src.rag.chain import _guard_identity_coverage_gate


_CFG = IdentityCoverageConfig(enabled=True, project_root=PROJECT_ROOT)

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
_AQ_003_QUERY = "What is the MQTT specification about?"


def _real_doc(source_basename: str, page: int | None = None) -> Document:
    """
    Build a Document with metadata grounded in the real production R1
    registry (via the actual enrich_metadata()), not hand-invented values --
    keeps every fixture traceable to production identity data.
    """
    meta = enrich_metadata(
        project_root=PROJECT_ROOT,
        source=source_basename,
        metadata={"source": source_basename, "page": page},
    )
    return Document(page_content=f"chunk from {source_basename}", metadata=meta)


def _ok_state(query: str, docs: list[Document]) -> dict:
    return {
        "input": query,
        "docs": docs,
        "status": "ok",
        "skip_llm": False,
        "context": "",
        "answer": "",
        "refusal_reason": None,
        "options": [],
        "selected_option": None,
    }


def _write_registry(root: Path, rules: list[dict]) -> Path:
    catalog_dir = root / "data" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "docs_registry.json"
    path.write_text(json.dumps({"rules": rules}), encoding="utf-8")
    return path


def _write_aliases(root: Path, aliases: list[dict]) -> Path:
    catalog_dir = root / "data" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "document_identity_aliases.json"
    path.write_text(json.dumps({"aliases": aliases}), encoding="utf-8")
    return path


# ============================================================================
# Unit tests
# ============================================================================

def test_A_no_requirements_state_unchanged():
    docs = [_real_doc("iot-dg.pdf")]
    state = _ok_state("How does MQTT work?", docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_B_logical_document_requirement_satisfied_state_unchanged():
    docs = [_real_doc("white-paper-iot-july-2018.pdf")]
    state = _ok_state("the general IoT whitepaper", docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_C_equivalence_group_requirement_satisfied_state_unchanged():
    docs = [_real_doc("mqtt-v3.1.1-os.pdf", page=0)]
    state = _ok_state("the MQTT specification itself", docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_D_logical_requirement_not_satisfied_by_equivalence_field_only():
    # Document explicitly carries equivalence_group_id but NOT
    # logical_document_id -- must not satisfy a logical_document requirement.
    doc = Document(page_content="x", metadata={"equivalence_group_id": "mqtt_v3_1_1_spec"})
    req = IdentityRequirement(
        status="resolved", identity_kind="logical_document",
        identity_value="mqtt_v3_1_1_spec", requirement_id="r",
    )
    assert _requirement_satisfied(req, [doc]) is False


def test_E_equivalence_requirement_not_satisfied_by_logical_field_only():
    # Converse of D: logical_document_id alone must not satisfy an
    # equivalence_group requirement, even with the identical value.
    doc = Document(page_content="x", metadata={"logical_document_id": "mqtt_v3_1_1_spec"})
    req = IdentityRequirement(
        status="resolved", identity_kind="equivalence_group",
        identity_value="mqtt_v3_1_1_spec", requirement_id="r",
    )
    assert _requirement_satisfied(req, [doc]) is False


def test_F_one_missing_resolved_requirement_refuses():
    docs = [_real_doc("iot-dg.pdf"), _real_doc("designing-mqtt-topics-aws-iot-core.pdf")]
    state = _ok_state("the MQTT specification itself", docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["skip_llm"] is True
    assert result["docs"] == []
    assert "mqtt_v3_1_1_spec" in result["refusal_reason"]


def test_G_two_requirements_both_satisfied_state_unchanged():
    docs = [_real_doc("white-paper-iot-july-2018.pdf"), _real_doc("mqtt-v3.1.1-os.pdf")]
    state = _ok_state(_XQ_004_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_H_two_requirements_one_missing_refuses():
    docs = [_real_doc("white-paper-iot-july-2018.pdf")]  # whitepaper present, MQTT spec absent
    state = _ok_state(_XQ_004_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert "mqtt_v3_1_1_spec" in result["refusal_reason"]
    assert "general_iot_whitepaper_2018" not in result["refusal_reason"]


def test_I_two_requirements_both_missing_refuses():
    docs = [_real_doc("iot-dg.pdf"), _real_doc("designing-mqtt-topics-aws-iot-core.pdf")]
    state = _ok_state(_XQ_004_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert "mqtt_v3_1_1_spec" in result["refusal_reason"]
    assert "general_iot_whitepaper_2018" in result["refusal_reason"]


def test_J_mqtt_html_rendering_satisfies_equivalence_group_requirement():
    docs = [_real_doc("mqtt-v3-1-1-spec.html")]
    state = _ok_state("the MQTT specification itself", docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_K_wrong_document_sharing_topic_does_not_satisfy():
    # iot-dg.pdf shares the MQTT topic/entity extensively but is not the
    # MQTT specification -- must not satisfy the equivalence_group requirement.
    docs = [_real_doc("iot-dg.pdf")]
    state = _ok_state("the MQTT specification itself", docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"


def test_L_unresolved_requirement_refuses(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        {"match": {"filename": "b.pdf"}, "tags": {"logical_document_id": "doc_b"}},
    ])
    _write_aliases(tmp_path, [
        {"requirement_id": "ref_a", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+special\s+doc\b"]},
        {"requirement_id": "ref_b", "identity_kind": "logical_document", "identity_value": "doc_b", "patterns": [r"\bspecial\b"]},
    ])
    cfg = IdentityCoverageConfig(enabled=True, project_root=tmp_path)
    docs = [Document(page_content="x", metadata={"logical_document_id": "doc_a"})]
    state = _ok_state("the special doc", docs)
    result = identity_coverage_gate(state, cfg=cfg)
    assert result["status"] == "refuse"
    assert result["skip_llm"] is True
    assert "could not be resolved" in result["refusal_reason"].lower()


def test_M_config_error_propagates_not_refuse(tmp_path):
    # A valid authoritative registry but NO alias registry file at all ->
    # DocumentIdentityConfigError must propagate, never be caught/converted.
    _write_registry(tmp_path, [{"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}}])
    cfg = IdentityCoverageConfig(enabled=True, project_root=tmp_path)
    docs = [Document(page_content="x", metadata={"logical_document_id": "doc_a"})]
    state = _ok_state("anything", docs)
    with pytest.raises(DocumentIdentityConfigError):
        identity_coverage_gate(state, cfg=cfg)


def test_N_admitted_document_missing_identity_metadata_does_not_satisfy():
    # Simulates a document whose metadata was never enriched (e.g. a
    # remediation index that hasn't been rebuilt yet) -- must refuse, not
    # crash and not falsely pass.
    doc = Document(page_content="x", metadata={"source": "mqtt-v3.1.1-os.pdf"})
    state = _ok_state("the MQTT specification itself", [doc])
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"


def test_O_refusal_reason_distinguishes_unresolved_from_missing(tmp_path):
    # Unresolved-reference reason.
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        {"match": {"filename": "b.pdf"}, "tags": {"logical_document_id": "doc_b"}},
    ])
    _write_aliases(tmp_path, [
        {"requirement_id": "ref_a", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+special\s+doc\b"]},
        {"requirement_id": "ref_b", "identity_kind": "logical_document", "identity_value": "doc_b", "patterns": [r"\bspecial\b"]},
    ])
    cfg = IdentityCoverageConfig(enabled=True, project_root=tmp_path)
    unresolved_state = _ok_state("the special doc", [Document(page_content="x", metadata={"logical_document_id": "doc_a"})])
    unresolved_result = identity_coverage_gate(unresolved_state, cfg=cfg)

    # Missing-coverage reason (real production config).
    missing_result = identity_coverage_gate(
        _ok_state("the MQTT specification itself", [_real_doc("iot-dg.pdf")]),
        cfg=_CFG,
    )

    assert unresolved_result["status"] == "refuse"
    assert missing_result["status"] == "refuse"
    assert unresolved_result["refusal_reason"] != missing_result["refusal_reason"]
    assert "could not be resolved" in unresolved_result["refusal_reason"].lower()
    assert "missing" in missing_result["refusal_reason"].lower()


# ============================================================================
# Empty-evidence identity semantics.
#
# An empty admitted-evidence set is simply one possible (unsatisfying)
# evidence set to check each requirement against once an explicit source
# contract exists -- it is never a special early-return that bypasses
# unresolved/missing classification. Bypassing that classification would
# collapse a specific identity-contract failure into the generic downstream
# "No relevant documents found" refusal, discarding the specific reason.
# ============================================================================

def test_7A_unresolved_requirement_with_empty_docs_refuses_with_unresolved_reason(tmp_path):
    _write_registry(tmp_path, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        {"match": {"filename": "b.pdf"}, "tags": {"logical_document_id": "doc_b"}},
    ])
    _write_aliases(tmp_path, [
        {"requirement_id": "ref_a", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+special\s+doc\b"]},
        {"requirement_id": "ref_b", "identity_kind": "logical_document", "identity_value": "doc_b", "patterns": [r"\bspecial\b"]},
    ])
    cfg = IdentityCoverageConfig(enabled=True, project_root=tmp_path)
    state = _ok_state("the special doc", [])  # docs=[]
    result = identity_coverage_gate(state, cfg=cfg)
    assert result["status"] == "refuse"
    assert result["skip_llm"] is True
    assert "could not be resolved" in result["refusal_reason"].lower()


def test_7B_resolved_logical_document_requirement_with_empty_docs_refuses_typed():
    state = _ok_state("the general IoT whitepaper", [])  # docs=[]
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["skip_llm"] is True
    assert result["refusal_reason"] == "Missing required source-identity coverage: logical_document:general_iot_whitepaper_2018"


def test_7C_resolved_equivalence_group_requirement_with_empty_docs_refuses_typed():
    state = _ok_state("the MQTT specification itself", [])  # docs=[]
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["skip_llm"] is True
    assert result["refusal_reason"] == "Missing required source-identity coverage: equivalence_group:mqtt_v3_1_1_spec"


def test_7D_two_resolved_requirements_with_empty_docs_refuses_listing_both():
    state = _ok_state(_XQ_004_QUERY, [])  # docs=[]
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["refusal_reason"] == (
        "Missing required source-identity coverage: "
        "logical_document:general_iot_whitepaper_2018, equivalence_group:mqtt_v3_1_1_spec"
    )


def test_7E_no_requirements_with_empty_docs_is_r7_noop_then_downstream_generic_refusal():
    # R7 itself must be a true no-op here (state identity-preserved) --
    # never any REFUSE emitted by R7.
    state = _ok_state("How does MQTT work?", [])  # docs=[]
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state

    # Only when routed through the SEPARATE, downstream, unmodified generic
    # guard does this become the generic no-doc refusal -- proving R7 did
    # not duplicate or preempt that responsibility. (refuse_if_no_docs's own
    # established contract signals refusal via skip_llm + refusal_reason,
    # not by overwriting `status`; this test follows that existing contract
    # rather than asserting behavior on policy.py it does not have.)
    from src.rag.policy import refuse_if_no_docs
    downstream_result = refuse_if_no_docs(result)
    assert downstream_result["skip_llm"] is True
    assert downstream_result["refusal_reason"] == "No relevant documents found"


# ----------------------------------------------------------------------------
# Four-way outcome distinction pin: these must remain mechanically
# distinguishable from one another, not just individually correct.
# ----------------------------------------------------------------------------

def test_four_way_outcome_distinction_is_mechanically_preserved(tmp_path):
    # 1. Invalid R6 configuration -> DocumentIdentityConfigError (never REFUSE).
    _write_registry(tmp_path, [{"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}}])
    # (no alias file written at all)
    bad_cfg = IdentityCoverageConfig(enabled=True, project_root=tmp_path)
    with pytest.raises(DocumentIdentityConfigError):
        identity_coverage_gate(_ok_state("anything", []), cfg=bad_cfg)

    # 2. Unresolved explicit source -> REFUSE / unresolved reason.
    tmp_path2 = tmp_path / "unresolved_case"
    _write_registry(tmp_path2, [
        {"match": {"filename": "a.pdf"}, "tags": {"logical_document_id": "doc_a"}},
        {"match": {"filename": "b.pdf"}, "tags": {"logical_document_id": "doc_b"}},
    ])
    _write_aliases(tmp_path2, [
        {"requirement_id": "ref_a", "identity_kind": "logical_document", "identity_value": "doc_a", "patterns": [r"\bthe\s+special\s+doc\b"]},
        {"requirement_id": "ref_b", "identity_kind": "logical_document", "identity_value": "doc_b", "patterns": [r"\bspecial\b"]},
    ])
    unresolved_cfg = IdentityCoverageConfig(enabled=True, project_root=tmp_path2)
    unresolved_result = identity_coverage_gate(_ok_state("the special doc", []), cfg=unresolved_cfg)
    assert unresolved_result["status"] == "refuse"
    assert "could not be resolved" in unresolved_result["refusal_reason"].lower()

    # 3. Resolved explicit source, no satisfying evidence -> REFUSE / typed reason.
    missing_result = identity_coverage_gate(_ok_state("the MQTT specification itself", []), cfg=_CFG)
    assert missing_result["status"] == "refuse"
    assert missing_result["refusal_reason"] == "Missing required source-identity coverage: equivalence_group:mqtt_v3_1_1_spec"

    # 4. No explicit source + no evidence -> R7 no-op; downstream generic refusal.
    noref_state = _ok_state("How does MQTT work?", [])
    noref_result = identity_coverage_gate(noref_state, cfg=_CFG)
    assert noref_result is noref_state
    from src.rag.policy import refuse_if_no_docs
    generic = refuse_if_no_docs(noref_result)
    assert generic["refusal_reason"] == "No relevant documents found"

    # All four outcomes are mutually distinguishable.
    reasons_or_markers = {
        "config_error",
        unresolved_result["refusal_reason"],
        missing_result["refusal_reason"],
        generic["refusal_reason"],
    }
    assert len(reasons_or_markers) == 4


# ============================================================================
# Direct terminal-state chain-guard tests
# ============================================================================

def test_chain_guard_incoming_refuse_state_unchanged():
    state = {"input": "the MQTT specification itself", "docs": [], "status": "refuse", "skip_llm": True, "refusal_reason": "some prior reason"}
    result = _guard_identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_chain_guard_incoming_ambiguous_state_unchanged():
    state = {"input": "the MQTT specification itself", "docs": [], "status": "ambiguous", "skip_llm": False, "options": []}
    result = _guard_identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_chain_guard_incoming_ood_or_other_non_ok_status_unchanged():
    state = {"input": "the MQTT specification itself", "docs": [], "status": "ood", "skip_llm": True}
    result = _guard_identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_chain_guard_skip_llm_true_state_unchanged_even_with_ok_status():
    # skip_llm already true (short-circuited upstream) must bypass R7
    # regardless of status.
    docs = [_real_doc("iot-dg.pdf")]  # would otherwise fail coverage
    state = _ok_state("the MQTT specification itself", docs)
    state["skip_llm"] = True
    result = _guard_identity_coverage_gate(state, cfg=_CFG)
    assert result is state


# ============================================================================
# Enable/disable policy
# ============================================================================

def test_default_config_has_identity_coverage_enabled():
    assert RETRIEVAL_CONFIG["identity_coverage"]["enabled"] is True


def test_disabled_gate_is_complete_noop_even_with_missing_coverage():
    docs = [_real_doc("iot-dg.pdf")]  # would normally refuse when enabled
    state = _ok_state("the MQTT specification itself", docs)
    cfg = IdentityCoverageConfig(enabled=False, project_root=PROJECT_ROOT)
    result = identity_coverage_gate(state, cfg=cfg)
    assert result is state


# ============================================================================
# Pipeline / historical regression
# ============================================================================

def test_xq001_historical_admitted_evidence_now_refuses():
    docs = [
        _real_doc("iot-dg.pdf", page=637),
        _real_doc("designing-mqtt-topics-aws-iot-core.pdf", page=28),
        _real_doc("iot-dg.pdf", page=636),
        _real_doc("iot-dg.pdf", page=652),
    ]
    state = _ok_state(_XQ_001_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["skip_llm"] is True
    assert "mqtt_v3_1_1_spec" in result["refusal_reason"]


def test_xq004_historical_admitted_evidence_now_refuses():
    docs = [
        _real_doc("designing-mqtt-topics-aws-iot-core.pdf", page=32),
        _real_doc("designing-mqtt-topics-aws-iot-core.pdf", page=6),
        _real_doc("iot-dg.pdf", page=350),
        _real_doc("iot-dg.pdf", page=351),
    ]
    state = _ok_state(_XQ_004_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert "mqtt_v3_1_1_spec" in result["refusal_reason"]
    assert "general_iot_whitepaper_2018" in result["refusal_reason"]
    # Deterministic order -- both missing requirements appear in
    # R6's own mention order (whitepaper is named before "MQTT protocol
    # specification" in the query text), and the exact same query/evidence
    # produces the exact same string on repeat evaluation.
    assert result["refusal_reason"] == (
        "Missing required source-identity coverage: "
        "logical_document:general_iot_whitepaper_2018, equivalence_group:mqtt_v3_1_1_spec"
    )
    repeat = identity_coverage_gate(_ok_state(_XQ_004_QUERY, docs), cfg=_CFG)
    assert repeat["refusal_reason"] == result["refusal_reason"]


def test_xq004_partial_A_whitepaper_satisfied_mqtt_missing_refuses_for_mqtt_only():
    docs = [_real_doc("white-paper-iot-july-2018.pdf")]  # satisfies logical_document only
    state = _ok_state(_XQ_004_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["refusal_reason"] == "Missing required source-identity coverage: equivalence_group:mqtt_v3_1_1_spec"
    assert "general_iot_whitepaper_2018" not in result["refusal_reason"]


def test_xq004_partial_B_mqtt_satisfied_whitepaper_missing_refuses_for_whitepaper_only():
    docs = [_real_doc("mqtt-v3.1.1-os.pdf")]  # satisfies equivalence_group only
    state = _ok_state(_XQ_004_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result["status"] == "refuse"
    assert result["refusal_reason"] == "Missing required source-identity coverage: logical_document:general_iot_whitepaper_2018"
    assert "mqtt_v3_1_1_spec" not in result["refusal_reason"]


def test_cq003_r7_is_complete_noop_because_r6_returns_nothing():
    assert resolve_required_identities(_CQ_003_QUERY, project_root=PROJECT_ROOT) == []

    state = {
        "input": _CQ_003_QUERY,
        "docs": [],
        "status": "ambiguous",
        "skip_llm": False,
        "options": [],
    }
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_cq003_chain_guard_skips_non_ok_status_without_calling_resolver():
    state = {"input": _CQ_003_QUERY, "docs": [], "status": "ambiguous", "skip_llm": False}
    result = _guard_identity_coverage_gate(state, cfg=_CFG)
    assert result is state


def test_aq003_r6_recognizes_requirement_under_current_aliases():
    reqs = resolve_required_identities(_AQ_003_QUERY, project_root=PROJECT_ROOT)
    assert len(reqs) == 1
    assert reqs[0].status == "resolved"
    assert reqs[0].identity_kind == "equivalence_group"
    assert reqs[0].identity_value == "mqtt_v3_1_1_spec"


def test_aq003_r1_repaired_evidence_satisfies_requirement_no_regression():
    docs = [
        _real_doc("mqtt-v3-1-1-spec.html"),
        _real_doc("mqtt-v3.1.1-os.pdf", page=65),
    ]
    state = _ok_state(_AQ_003_QUERY, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


@pytest.mark.parametrize("query", [
    "How does MQTT work?",
    "AWS IoT jobs",
    "IoT connectivity",
    "What is firmware update?",
])
def test_ordinary_non_named_queries_are_complete_noop_through_gate(query):
    docs = [_real_doc("iot-dg.pdf")]
    state = _ok_state(query, docs)
    result = identity_coverage_gate(state, cfg=_CFG)
    assert result is state


@pytest.mark.parametrize("query", [
    "How does MQTT work?",
    "AWS IoT jobs",
    "IoT connectivity",
    "What is firmware update?",
])
def test_ordinary_non_named_queries_are_complete_noop_through_chain_guard(query):
    docs = [_real_doc("iot-dg.pdf")]
    state = _ok_state(query, docs)
    result = _guard_identity_coverage_gate(state, cfg=_CFG)
    assert result is state
