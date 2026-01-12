from __future__ import annotations

def debug(
    domain: str,
    message: str,
) -> None:
    print(f"[DEBUG][{domain}] {message}")


def warn(
    domain: str,
    message: str
) -> None:
    print(f"[WARN][{domain}] {message}")


def info(
    domain: str,
    message: str
) -> None:
    print(f"[INFO][{domain}] {message}")
