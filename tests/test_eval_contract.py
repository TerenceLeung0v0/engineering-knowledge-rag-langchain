from src.eval.schemas import parse_case

def test_parse_case_valid():
    c = parse_case({"id": "t1", "query": "q", "expect_status": "ok", "expect_sources": []})
    assert c.id == "t1"
    assert c.expect_status == "ok"

def test_parse_case_invalid_status():
    try:
        parse_case({"id": "t1", "query": "q", "expect_status": "oops"})
        assert False, "should raise"
    except ValueError as e:
        assert "Invalid expect_status" in str(e)

def test_parse_case_min_sources():
    c = parse_case({"id": "t2", "query": "q", "expect_status": "ok", "min_sources": 1})
    assert c.min_sources == 1
