from langchain_core.documents import Document
from src.schemas import ScoredDocument
from src.rag.retriever import _resolve_tag_ambiguity
from src.rag.ambiguity import AmbiguityConfig
from src.rag.entity_extract import EntityExtractor

import re

def test_entity_aware_resolve_prefers_jobs_group():
    entity_patterns = {
        "aws_iot": (re.compile(r"\baws\s*iot\b", re.I),),
        "aws_iot_jobs": (re.compile(r"\baws\s*iot\s*jobs?\b|\biot\s*jobs?\b", re.I),),
    }
    extractor = EntityExtractor(entity_patterns=entity_patterns)

    cfg = AmbiguityConfig(
        entity_extractor=extractor,
        enable_entity_resolve=True,
        require_full_entity_coverage=True,
        # disable embedding tiebreak for deterministic unit test
        embed_docs=None,
        enable_sig_tiebreak=False,
        enable_anchor_tiebreak=False,
    )

    # Group A: aws_iot only
    d1 = Document(page_content="iot core", metadata={"source":"a.pdf","entities":["aws_iot"], "domain":"aws_iot", "doc_type":"pdf", "product":"iot_core"})
    d2 = Document(page_content="iot core 2", metadata={"source":"a.pdf","entities":["aws_iot"], "domain":"aws_iot", "doc_type":"pdf", "product":"iot_core"})
    # Group B: aws_iot + aws_iot_jobs
    d3 = Document(page_content="jobs", metadata={"source":"b.pdf","entities":["aws_iot","aws_iot_jobs"], "domain":"aws_iot", "doc_type":"pdf", "product":"iot_jobs"})
    d4 = Document(page_content="jobs 2", metadata={"source":"b.pdf","entities":["aws_iot_jobs"], "domain":"aws_iot", "doc_type":"pdf", "product":"iot_jobs"})

    scored = [
        ScoredDocument(doc=d1, score=0.30),
        ScoredDocument(doc=d3, score=0.31),
        ScoredDocument(doc=d2, score=0.32),
        ScoredDocument(doc=d4, score=0.33),
    ]
    scored.sort(key=lambda x: x.score)

    _, auto, docs = _resolve_tag_ambiguity(scored, query="AWS IoT Jobs", final_k=2, cfg=cfg)

    assert auto is True
    assert len(docs) >= 1
    # should pick jobs group anchor first
    assert "aws_iot_jobs" in (docs[0].metadata or {}).get("entities", [])
