from langchain_core.documents import Document

from src.schemas import ScoredDocument
from src.rag.gating import _validate_confidence_gap_gate

def _sd(score: float, meta: dict) -> ScoredDocument:
    return ScoredDocument(doc=Document(page_content="x", metadata=meta), score=score)


def test_gap_gate_pass_when_gap_is_large():
    scored = [
        _sd(0.10, {"domain": "mqtt", "doc_type": "spec", "product": "mqtt"}),
        _sd(0.30, {"domain": "aws_iot", "doc_type": "guide", "product": "iot_core"}),
    ]
    assert _validate_confidence_gap_gate(scored, min_gap=0.05) is True


def test_gap_gate_blocks_when_gap_small_and_different_family():
    scored = [
        _sd(0.10, {"domain": "mqtt", "doc_type": "spec", "product": "mqtt"}),
        _sd(0.12, {"domain": "aws_iot", "doc_type": "guide", "product": "iot_core"}),
    ]
    assert _validate_confidence_gap_gate(scored, min_gap=0.05) is False


def test_gap_gate_pass_when_gap_small_but_same_family():
    scored = [
        _sd(0.10, {"domain": "aws_iot", "doc_type": "guide", "product": "iot_core"}),
        _sd(0.12, {"domain": "aws_iot", "doc_type": "guide", "product": "iot_core"}),
    ]
    assert _validate_confidence_gap_gate(scored, min_gap=0.05) is True


def test_gap_gate_pass_when_gap_small_but_same_file_pages_close():
    scored = [
        _sd(0.10, {"source": "/x/iot-dg.pdf", "page": 10}),
        _sd(0.12, {"source": "/x/iot-dg.pdf", "page": 11}),
    ]
    assert _validate_confidence_gap_gate(scored, min_gap=0.05) is True
