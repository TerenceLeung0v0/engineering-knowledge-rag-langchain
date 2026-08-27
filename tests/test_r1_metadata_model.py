from pathlib import Path

from langchain_core.documents import Document

from src.config import PROJECT_ROOT, RAW_DOCS_PDF_DIR, RAW_DOCS_HTML_DIR
from src.ingest.loaders.pdf_loader import load_pdfs_documents
from src.ingest.loaders.html_loader import load_htmls_documents
from src.ingest.loaders.md_loader import _read_markdown_file
from src.rag.catalog import DocTags, resolve_doc_tags, enrich_metadata, tag_signature
from src.rag.gating import _validate_confidence_gap_gate
from src.schemas import ScoredDocument


# --- Loaders preserve physical source_format, no longer pre-populate doc_type ---

def test_pdf_loader_sets_source_format_not_doc_type():
    docs = load_pdfs_documents(RAW_DOCS_PDF_DIR, glob_pattern="mqtt-v3.1.1-os.pdf")
    assert docs, "expected at least one loaded PDF document"
    for d in docs:
        assert d.metadata.get("source_format") == "pdf"
        assert "doc_type" not in d.metadata


def test_html_loader_sets_source_format_not_doc_type():
    docs = load_htmls_documents(RAW_DOCS_HTML_DIR, glob_pattern="mqtt-v3-1-1-spec.html")
    assert docs, "expected at least one loaded HTML document"
    for d in docs:
        assert d.metadata.get("source_format") == "html"
        assert "doc_type" not in d.metadata


def test_md_loader_sets_source_format_not_doc_type(tmp_path):
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\n\nSome content.\n")
    doc = _read_markdown_file(md_file)
    assert doc.metadata.get("source_format") == "md"
    assert "doc_type" not in doc.metadata


# --- Registry semantic doc_type is reachable; physical source_format survives enrichment ---

def test_registry_semantic_doc_type_applied_after_enrichment():
    meta = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3.1.1-os.pdf",
        metadata={"source_format": "pdf"},
    )
    assert meta["doc_type"] == "spec"


def test_source_format_not_overwritten_by_registry_enrichment():
    meta = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3.1.1-os.pdf",
        metadata={"source_format": "pdf"},
    )
    assert meta["source_format"] == "pdf"


def test_registry_tags_never_define_source_format():
    # source_format is loader-owned; the registry must never be able to assert it.
    tags = resolve_doc_tags(project_root=PROJECT_ROOT, source="mqtt-v3.1.1-os.pdf").to_metadata()
    assert "source_format" not in tags


# --- Same-physical-format documents that previously collided now separate on semantic doc_type ---

def test_previously_colliding_same_format_docs_now_distinguished():
    meta_guide = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="iot-dg.pdf",
        metadata={"source_format": "pdf"},
    )
    meta_whitepaper = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="designing-mqtt-topics-aws-iot-core.pdf",
        metadata={"source_format": "pdf"},
    )
    # Same physical format and same domain/product -- under the old (physical-format)
    # doc_type these collided into one tag_signature. They must not collide now.
    assert meta_guide["source_format"] == meta_whitepaper["source_format"] == "pdf"
    assert meta_guide["domain"] == meta_whitepaper["domain"] == "aws_iot"
    assert meta_guide["product"] == meta_whitepaper["product"] == "iot_core"
    assert meta_guide["doc_type"] != meta_whitepaper["doc_type"]
    assert tag_signature(meta_guide) != tag_signature(meta_whitepaper)


# --- Registry completeness for the currently ingested HTML corpus ---

def test_all_ingested_html_documents_are_registered():
    html_sources = [
        "aws-iot-core-topics.html",
        "aws-iot-jobs-overview.html",
        "aws-iot-jobs-workflows.html",
        "aws-iot-job-execution-states.html",
        "aws-iot-thing-groups.html",
        "mqtt-v3-1-1-spec.html",
    ]
    for src in html_sources:
        tags = resolve_doc_tags(project_root=PROJECT_ROOT, source=src)
        assert tags.domain is not None, f"{src} missing domain after registry completion"
        assert tags.doc_type is not None, f"{src} missing doc_type after registry completion"


# ============================================================================
# tag_signature equality must never be read as logical-document identity.
# This is a structural property of tag_signature() itself -- it is computed
# from (domain, doc_type, product) only and cannot see logical_document_id.
# Proven directly at the function level with an explicit synthetic fixture
# (not asserted as corpus fact): under the current AWS Developer Guide
# identity model (see the tests below), no two *distinct* logical works in
# the corpus happen to share a tag_signature, so the invariant is
# demonstrated structurally rather than via a real corpus pair.
# ============================================================================

def test_same_tag_signature_does_not_imply_same_logical_document():
    shared_family = {"domain": "aws_iot", "doc_type": "guide", "product": "iot_core"}

    meta_a = {**shared_family, "source": "hypothetical-doc-a.pdf", "logical_document_id": "doc_a"}
    meta_b = {**shared_family, "source": "hypothetical-doc-b.pdf", "logical_document_id": "doc_b"}

    # Same broad semantic family ...
    assert tag_signature(meta_a) == tag_signature(meta_b) == ("aws_iot", "guide", "iot_core")
    # ... is not evidence of, and must never be treated as, the same authored work.
    assert meta_a["logical_document_id"] != meta_b["logical_document_id"]


def test_corrected_aws_family_shares_tag_signature_and_logical_identity_but_distinct_physical_sources():
    # The corpus-grounded case: iot-dg.pdf and its five HTML section
    # renderings legitimately share BOTH tag_signature AND logical_document_id
    # (they are the same authored work), while remaining physically distinct
    # sources. This demonstrates the converse situation from the test above --
    # shared tag_signature CAN coincide with shared logical identity when that
    # identity is independently, evidentially justified by corpus evidence --
    # it is simply never *inferred* from tag_signature alone.
    sources = [
        "iot-dg.pdf",
        "aws-iot-core-topics.html",
        "aws-iot-jobs-overview.html",
        "aws-iot-jobs-workflows.html",
        "aws-iot-job-execution-states.html",
        "aws-iot-thing-groups.html",
    ]
    metas = [enrich_metadata(project_root=PROJECT_ROOT, source=s, metadata={"source": s}) for s in sources]

    assert {tag_signature(m) for m in metas} == {("aws_iot", "guide", "iot_core")}
    assert {m["logical_document_id"] for m in metas} == {"aws_iot_core_developer_guide"}
    assert len({Path(m["source"]).name for m in metas}) == len(sources)  # physically distinct


# ============================================================================
# logical_document_id and equivalence_group_id are independent fields.
# Constructed directly via DocTags (bypassing the real registry file) so no
# fictional document is added to docs_registry.json.
# ============================================================================

def test_logical_document_id_and_equivalence_group_id_are_independent_fields():
    tags = DocTags(
        domain="mqtt",
        doc_type="spec",
        product="mqtt",
        logical_document_id="compatible_summary",
        equivalence_group_id="mqtt_v3_1_1_spec",
    )
    meta = tags.to_metadata()

    assert meta["logical_document_id"] == "compatible_summary"
    assert meta["equivalence_group_id"] == "mqtt_v3_1_1_spec"
    assert meta["logical_document_id"] != meta["equivalence_group_id"]


# ============================================================================
# AWS Developer Guide logical-document-model tests
# ============================================================================

_AWS_GUIDE_SOURCES = [
    "iot-dg.pdf",
    "aws-iot-core-topics.html",
    "aws-iot-jobs-overview.html",
    "aws-iot-jobs-workflows.html",
    "aws-iot-job-execution-states.html",
    "aws-iot-thing-groups.html",
]


def test_aws_guide_family_shares_logical_document_id():
    # iot-dg.pdf and all five HTML section renderings share one logical
    # identity -- corroborated directly against the corpus: each HTML file
    # contains a "download PDF" link of the form
    # href=".../iot-dg.pdf#<anchor>", confirming they are section renderings
    # of iot-dg.pdf, not independently-authored works.
    metas = [enrich_metadata(project_root=PROJECT_ROOT, source=s, metadata={}) for s in _AWS_GUIDE_SOURCES]
    logical_ids = {m["logical_document_id"] for m in metas}
    assert logical_ids == {"aws_iot_core_developer_guide"}


def test_aws_guide_family_physical_sources_remain_distinct():
    # Shared logical identity must never collapse physical identity.
    metas = [
        enrich_metadata(project_root=PROJECT_ROOT, source=s, metadata={"source": s})
        for s in _AWS_GUIDE_SOURCES
    ]
    physical_ids = {Path(m["source"]).name for m in metas}
    assert len(physical_ids) == len(_AWS_GUIDE_SOURCES) == 6


def test_aws_guide_family_source_format_distinguishes_pdf_from_html():
    # source_format still distinguishes the physical rendering: the one
    # PDF differs from the five HTML pages.
    meta_pdf = enrich_metadata(
        project_root=PROJECT_ROOT, source="iot-dg.pdf", metadata={"source_format": "pdf"}
    )
    html_metas = [
        enrich_metadata(project_root=PROJECT_ROOT, source=s, metadata={"source_format": "html"})
        for s in _AWS_GUIDE_SOURCES[1:]
    ]
    assert meta_pdf["source_format"] == "pdf"
    assert all(m["source_format"] == "html" for m in html_metas)
    assert {meta_pdf["source_format"]} != {m["source_format"] for m in html_metas}


def test_aws_guide_html_sections_have_no_fabricated_equivalence_group():
    # Sharing logical_document_id must NOT imply a fabricated whole-guide
    # equivalence_group_id. Each HTML page corresponds to one specific anchor
    # within iot-dg.pdf, not the entire document -- R1 has no section/anchor
    # -level equivalence mechanism, so equivalence_group_id must be absent.
    for s in _AWS_GUIDE_SOURCES[1:]:
        tags = resolve_doc_tags(project_root=PROJECT_ROOT, source=s)
        assert tags.equivalence_group_id is None, f"{s} must not have a fabricated equivalence_group_id"


def test_iot_dg_pdf_does_not_establish_whole_file_evidence_equivalence():
    # iot-dg.pdf itself must not claim file-level evidence-equivalence with
    # every subsection either, absent section-level machinery.
    tags = resolve_doc_tags(project_root=PROJECT_ROOT, source="iot-dg.pdf")
    assert tags.equivalence_group_id is None


# ============================================================================
# MQTT identity/equivalence tests
# ============================================================================

def test_mqtt_pdf_and_html_share_logical_document_id():
    # A.
    meta_pdf = enrich_metadata(project_root=PROJECT_ROOT, source="mqtt-v3.1.1-os.pdf", metadata={})
    meta_html = enrich_metadata(project_root=PROJECT_ROOT, source="mqtt-v3-1-1-spec.html", metadata={})
    assert meta_pdf["logical_document_id"] == "mqtt_v3_1_1_spec"
    assert meta_html["logical_document_id"] == "mqtt_v3_1_1_spec"


def test_mqtt_pdf_and_html_share_equivalence_group_id():
    # Valid because they are complete renderings of the same authored work
    # (whole-document equivalence, unlike the AWS Guide sections).
    meta_pdf = enrich_metadata(project_root=PROJECT_ROOT, source="mqtt-v3.1.1-os.pdf", metadata={})
    meta_html = enrich_metadata(project_root=PROJECT_ROOT, source="mqtt-v3-1-1-spec.html", metadata={})
    assert meta_pdf["equivalence_group_id"] == "mqtt_v3_1_1_spec"
    assert meta_html["equivalence_group_id"] == "mqtt_v3_1_1_spec"


def test_mqtt_pdf_and_html_physical_identity_and_source_format_remain_distinct():
    # C.
    meta_pdf = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3.1.1-os.pdf",
        metadata={"source": "mqtt-v3.1.1-os.pdf", "source_format": "pdf"},
    )
    meta_html = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3-1-1-spec.html",
        metadata={"source": "mqtt-v3-1-1-spec.html", "source_format": "html"},
    )
    assert Path(meta_pdf["source"]).name != Path(meta_html["source"]).name
    assert meta_pdf["source_format"] != meta_html["source_format"]


def test_mqtt_pdf_and_html_share_tag_signature_after_enrichment():
    # AQ-003 precondition: equal semantic tag_signature after R1.
    meta_pdf = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3.1.1-os.pdf",
        metadata={"source": "mqtt-v3.1.1-os.pdf", "source_format": "pdf"},
    )
    meta_html = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3-1-1-spec.html",
        metadata={"source": "mqtt-v3-1-1-spec.html", "source_format": "html"},
    )
    assert tag_signature(meta_pdf) == ("mqtt", "spec", "mqtt")
    assert tag_signature(meta_html) == ("mqtt", "spec", "mqtt")
    assert tag_signature(meta_pdf) == tag_signature(meta_html)


# ============================================================================
# Standalone documents: logical_document_id retained, equivalence_group_id
# absent (no whole-work evidence-equivalent partner currently exists).
# ============================================================================

def test_standalone_documents_retain_logical_document_id_but_no_equivalence_group():
    for src, expected_logical_id in [
        ("designing-mqtt-topics-aws-iot-core.pdf", "designing_mqtt_topics_for_aws_iot_core_whitepaper"),
        ("white-paper-iot-july-2018.pdf", "general_iot_whitepaper_2018"),
    ]:
        tags = resolve_doc_tags(project_root=PROJECT_ROOT, source=src)
        assert tags.logical_document_id == expected_logical_id
        assert tags.equivalence_group_id is None


def test_logical_document_id_distinct_for_unrelated_documents():
    meta_a = enrich_metadata(project_root=PROJECT_ROOT, source="iot-dg.pdf", metadata={})
    meta_b = enrich_metadata(
        project_root=PROJECT_ROOT, source="designing-mqtt-topics-aws-iot-core.pdf", metadata={}
    )
    assert meta_a["logical_document_id"] != meta_b["logical_document_id"]


# --- AQ-003 mechanical diagnostic: gap-gate exemption after the R1 fix (no threshold change) ---

def test_aq003_gap_gate_recognizes_same_sig_after_r1_fix():
    # Exact observed AQ-003 top-two candidates/scores from the frozen Phase 5 raw
    # execution record (held_out_raw_execution.json), reconstructed with the
    # corrected metadata model. gating.py itself is untouched.
    meta_html = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3-1-1-spec.html",
        metadata={"source": "mqtt-v3-1-1-spec.html", "source_format": "html"},
    )
    meta_pdf = enrich_metadata(
        project_root=PROJECT_ROOT,
        source="mqtt-v3.1.1-os.pdf",
        metadata={"source": "mqtt-v3.1.1-os.pdf", "source_format": "pdf", "page": 65},
    )

    assert tag_signature(meta_html) == tag_signature(meta_pdf) == ("mqtt", "spec", "mqtt")

    scored = [
        ScoredDocument(doc=Document(page_content="mqtt spec html chunk", metadata=meta_html), score=0.4495995044708252),
        ScoredDocument(doc=Document(page_content="mqtt spec pdf chunk", metadata=meta_pdf), score=0.45132720470428467),
    ]

    # Real observed gap (~0.0017) is far below the real min_gap (0.015); before R1
    # this combination returned False (blocked -> AMBIGUOUS). It must now return
    # True via the same_sig exemption, using the unmodified gap-gate function.
    assert _validate_confidence_gap_gate(scored, min_gap=0.015) is True
