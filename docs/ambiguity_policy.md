# Ambiguity Policy

This document defines when the retriever returns `ok`, `refuse`, or `ambiguous`, and how ambiguity is resolved.

## Why ambiguity exists

In domain RAG, multiple document clusters can look relevant for a query, especially when the query asks for “overall”, “architecture”, or spans multiple subsystems.

Instead of guessing (hallucination risk), the system treats ambiguity as a first-class outcome:
- return `ambiguous` with explicit options, OR
- auto-resolve only when confidence is strong.

## Status definitions

### OK
Return `ok` when:
- relevant documents are retrieved AND pass all gates, AND
- the system can pick a single best cluster (or only one cluster exists).

### REFUSE
Return `refuse` when:
- query is out-of-domain by explicit deny/allow policy, OR
- retrieval finds no relevant evidence, OR
- coverage gate fails (retrieved docs do not support key entities in the query).

A refusal must include a `refusal_reason`.

### AMBIGUOUS
Return `ambiguous` when:
- there are multiple distinct candidate clusters (e.g., different source sets),
- and no auto-resolve rule can confidently select one,
- OR the query is an overview-style query and multiple signatures exist.

When `ambiguous`, the system returns a list of options, each option includes:
- a small doc set (anchor + distinct supporting docs)
- sources (filename, page/page_label)
- best_l2 score for the option’s anchor

User can select an option by providing `selected_option`.

## Pipeline summary

1) OOD gate (allow/deny patterns)
2) Vectorstore retrieval (FAISS) -> scored docs with L2 distance
3) L2 gating (absolute hard/soft, density, confidence gap)
4) If ambiguous -> tag-based grouping + ambiguity resolution policy
5) Output hygiene enforcement

## Tag grouping concept

Retrieved docs are grouped by a tag signature derived from metadata.

Goal:
- treat each signature group as a “topic cluster”
- avoid mixing unrelated clusters in a single answer

If a doc has no tag signature, a safe fallback signature is used
(e.g., `__file__:filename`) to keep grouping stable.

## Ambiguity resolution order (retriever policy)

Given tag groups sorted by best L2 (lower is better), attempt in order:

### (1) Single group
If only one group exists -> `ok`.

### (2) Generic overview query -> force ambiguous
If the query matches “generic/overview style” AND does not include enough facets
(or entities), do NOT auto-resolve.
Return `ambiguous` with options.

### (3) Entity-aware resolve (if enabled)
If entity extractor finds entities in the query, score each group by entity hits:
- anchor entity hits
- docs entity hits
- group coverage
If a single winner emerges -> `ok`.

If tied winners remain, keep ranked candidates and continue.

### (4) Group score gap resolve
If the best group is sufficiently better than the second best
(i.e., L2 gap >= `min_group_gap`) -> `ok`.

### (5) Query-aware tie-break (if enabled)
Use embeddings to compare the query against:
- group tag signatures (signature embedding)
- or anchor content (anchor embedding)

If similarity is strong enough and gap is large enough -> `ok`.

### (6) Otherwise -> user options
Return `ambiguous` with `max_options` options.

## User selection behavior

If `selected_option` is provided and matches an option id:
- return `ok` with the selected option’s docs
If invalid selection:
- return `refuse` with reason `Invalid selection: <id>`

## Determinism guarantees

- Retrieval returns scored docs sorted by L2 distance.
- Grouping is based on stable metadata-derived signatures.
- Options are de-duplicated by source signature (filename + page).
- All rules are threshold-driven and versionable via config.

## Testing contract

Each QA case checks:
- `expect_status`: ok/refuse/ambiguous
- expected sources (if applicable)
- hygiene constraints (no empty answers, refusal reason present, ambiguity explicit)

Evaluation is designed to catch regressions when tuning thresholds or changing
ambiguity logic.
