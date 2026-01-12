from src.rag.output_cleaner import clean_rag_output

def test_removes_answer_label_line():
    raw = "Answer:\nThis is the output."
    res = clean_rag_output(raw)
    assert "Answer:" not in res.text

def test_examples_header_removed_if_no_bullets():
    raw = "Some summary.\n\nExamples:\n\nNo example here."
    res = clean_rag_output(raw)
    assert "Examples:" not in res.text

def test_examples_header_kept_if_bullets_exist():
    raw = "Some summary.\n\nExamples:\n- Do X\n- Error Y"
    res = clean_rag_output(raw)
    assert "Examples:" in res.text

def test_remove_empty_bullets_and_placeholders():
    raw = "Summary.\n-\nN/A\nNone\n- Real bullet"
    res = clean_rag_output(raw)
    assert "\n-\n" not in res.text
    assert "N/A" not in res.text
    assert "None" not in res.text
    assert "- Real bullet" in res.text

def test_refusal_normalization_for_long_refusal():
    raw = "The provided context does not contain enough information. " * 10
    res = clean_rag_output(raw)
    assert res.decision == "refuse"
    assert len(res.text.split()) <= 20
