from __future__ import annotations
from typing import Protocol, Sequence, Any

class EmbeddingModel(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

def build_embeddings(cfg: dict[str, Any]) -> EmbeddingModel:
    """
    Build an embeddings backend based on configuration.
    Expected cfg keys:
      - backend (required): "huggingface" | "openai" | ...
      - model_name (optional depending on backend)
    """    
    backend = cfg["backend"]
    
    match backend:
        case "huggingface":
            try:
                import torch
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as e:
                raise ImportError(
                    "Missing Hugging Face embedding dependencies."
                ) from e
            
            model_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = int(cfg.get("batch_size", 64))
            show_progress = bool(cfg.get("show_progress", True))
            
            return  HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "batch_size": batch_size
                },
                show_progress=show_progress
            )
        
        case "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as e:
                raise ImportError(
                    "Missing OpenAI embedding dependencies."
                ) from e
            
            model_name = cfg.get("model_name", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model_name)
        
        case _:
            raise ValueError(f"Unsupported embedding backend: {backend!r}")
