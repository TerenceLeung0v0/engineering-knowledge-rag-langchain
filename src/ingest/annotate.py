from __future__ import annotations
from typing import Iterable

from langchain_core.documents import Document

from src.ingest.templates import DocMetadataTemplate
from src.ingest.entities.registry import build_entity_registry
from src.ingest.entities.tagger import tag_entities_for_docs

def annotate_docs(
    docs: Iterable[Document],
    *,
    cfg_entities: dict | None,
    template: DocMetadataTemplate | None = None,
) -> list[Document]:
    template = template or DocMetadataTemplate()
    registry = build_entity_registry(cfg_entities)

    standardized_docs: list[Document] = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        doc_id = f"{source}::{idx}"
        standardized_docs.append(
            template.apply(
                doc,
                source=source,
                doc_id=doc_id
            )
        )

    return tag_entities_for_docs(
        standardized_docs,
        registry=registry
    )
