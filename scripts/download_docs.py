
from __future__ import annotations
from pathlib import Path

from src.config import RAW_DOCS_DIR
from src.startup_checks import ensure_required_dirs

import hashlib
import shutil
import sys
import urllib.request

PDFS: list[tuple[str, str]] = [
    (
        "mqtt-v3.1.1-os.pdf",
        "https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.pdf",
    ),
    (
        "iot-dg.pdf",
        "https://docs.aws.amazon.com/iot/latest/developerguide/iot-dg.pdf",
    ),
    (
        "designing-mqtt-topics-aws-iot-core.pdf",
        "https://docs.aws.amazon.com/whitepapers/latest/designing-mqtt-topics-aws-iot-core/designing-mqtt-topics-aws-iot-core.pdf",
    ),
    (
        "white-paper-iot-july-2018.pdf",
        "https://portail-qualite.public.lu/dam-assets/publications/normalisation/2018/white-paper-iot-july-2018.pdf",
    ),
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)

def main() -> int:
    ensure_required_dirs([RAW_DOCS_DIR])
    print(f"[INFO] Download target: {RAW_DOCS_DIR}")

    for filename, url in PDFS:
        dest = RAW_DOCS_DIR / filename

        if dest.exists() and dest.stat().st_size > 0:
            print(f"[SKIP] {filename} already exists ({dest.stat().st_size} bytes)")
            continue

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.unlink(missing_ok=True)

        try:
            print(f"[DOWN] {filename}")
            download(url, tmp)

            if tmp.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")

            tmp.replace(dest)
            print(f"[OK]  {filename} ({dest.stat().st_size} bytes) sha256={sha256_file(dest)[:12]}...")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"[ERR] {filename}: {type(e).__name__}: {e}")
            return 1

    print("[DONE] All downloads finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
