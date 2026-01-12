# src/utils/diagnostics.py
from __future__ import annotations
from typing import Any, Iterable, Callable, TypeAlias
from src.utils.console import debug as _debug_out, warn as _warn_out, info as _info_out

Logger: TypeAlias = Callable[[str], None]

def _noop(*args, **kwargs) -> None:
    pass

def _resolve_domain_path(
    cfg: dict[str, Any],
    domain_path: Iterable[str]
) -> dict[str, Any] | None:
    """
    Check domain_path
    - e.g. domain_path=("rag", "chain")'
    - return None if not a dict
    """
    node = cfg
    for path in domain_path:
        if not isinstance(node, dict) or path not in node:
            return None
        node = node[path]
    
    return node if isinstance(node, dict) else None

def is_enabled(
    cfg: dict[str, Any],
    domain_path: Iterable[str],
    key: str
) -> bool:
    """
    Check key with a given domain_path
    """
    node = _resolve_domain_path(cfg, domain_path)
    if node is None:
        return False
    
    return bool(node.get(key, False))

def build_debug_logger(
    *,
    domain_path: Iterable[str] | str,
    key: str,
    cfg: dict[str, Any],
) -> Logger:
    """
    Returns pre-configured logger with a specific domain.
    Example:
    - dbg = make_debug_logger(
        path=("rag.gating"),
        key="print_absolute",
        cfg=DEBUG_CONFIG
      )
    - dbg("Here is the message")
    """
    if isinstance(domain_path, str):
        domain_path = domain_path.split(".")
    
    enabled = is_enabled(
        cfg=cfg,
        domain_path=domain_path,
        key=key
    )
    
    if enabled:
        domain = ".".join(domain_path)
        return lambda msg: _debug_out(domain, msg)
    
    return _noop

def warn(domain: str, message: str) -> None:
    _warn_out(domain, message)

def info(domain: str, message: str) -> None:
    _info_out(domain, message)
