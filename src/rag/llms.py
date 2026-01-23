from __future__ import annotations
from typing import Any
from langchain_core.runnables import Runnable

def build_llm(cfg: dict[str, Any]) -> Runnable:
    """
    Build an embeddings backend based on configuration.
    Expected cfg keys:
      - backend (required): "huggingface" | "openai" | ...
      - model_name (optional depending on backend)
    """    
    backend = cfg["backend"]
    
    match backend:
        case "ollama":
            try:
                from langchain_ollama import OllamaLLM
            except ImportError as e:
                raise ImportError(
                    "Missing Ollama dependencies."
                ) from e
            
            model_name = cfg.get("model_name", "llama3")
            return OllamaLLM(
                model=model_name,
                temperature=cfg.get("temperature", 0)
            )
        
        case "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise ImportError(
                    "Missing OpenAI embedding dependencies."
                ) from e
            
            model_name = cfg.get("model_name", "gpt-4o-mini")
            return ChatOpenAI(
                model=model_name,
                temperature=cfg.get("temperature", 0)
            )
        
        case _:
            raise ValueError(f"Unsupported LLM backend: {backend!r}")
