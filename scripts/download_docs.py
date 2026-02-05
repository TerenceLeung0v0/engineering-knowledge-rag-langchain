
from __future__ import annotations
from pathlib import Path

from src.config import RAW_DOCS_PDF_DIR, RAW_DOCS_HTML_DIR
from src.startup_checks import ensure_project_dirs

import hashlib
import shutil
import sys
import urllib.request
import argparse

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

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

HTMLS: list[tuple[str, str]] = [
    (
        "aws-iot-core-mqtt.html",
        "https://docs.aws.amazon.com/iot/latest/developerguide/iot-mqtt.html",
    ),
    (
        "aws-iot-core-topics.html",
        "https://docs.aws.amazon.com/iot/latest/developerguide/topics.html",
    ),
    (
        "aws-iot-jobs-overview.html",
        "https://docs.aws.amazon.com/iot/latest/developerguide/iot-jobs.html",
    ),
    (
        "aws-iot-jobs-workflows.html",
        "https://docs.aws.amazon.com/iot/latest/developerguide/jobs-workflow-jobs-online.html"
    ),
    (
        "aws-iot-job-execution-states.html",
        "https://docs.aws.amazon.com/iot/latest/developerguide/iot-jobs-lifecycle.html",
    ),
    (
        "aws-iot-thing-groups.html",
        "https://docs.aws.amazon.com/iot/latest/developerguide/thing-groups.html",
    ),
    (
        "mqtt-v3-1-1-spec.html",
        "https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html",
    ),
]

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)

def _download_file(
    *,
    filename: str,
    url: str,
    out_dir: Path | str,
    force: bool=False,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename

    if not force:
        if dest.exists() and dest.stat.st_size > 0:
            print(f"[SKIP] {dest.name} already exists ({dest.stat().st_size} bytes)")
            return 0

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    try:
        print(f"[DOWN] {dest.name}")
        _download(url, tmp)

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("downloaded file is empty")

        tmp.replace(dest)
        print(f"[OK]  {dest.name} ({dest.stat().st_size} bytes) sha256={_sha256_file(dest)[:12]}...")
        return 0
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"[ERR] {dest.name}: {type(e).__name__}: {e}")
        return 1

def _download_files(
    *,
    items: list[tuple[str, str]],
    out_dir: Path | str,
    label: str,
    force: bool=False
) -> int:
    if not items:
        print(f"[INFO] No {label} entries configured. Skipping.")
        return 0

    print(f"[INFO] {label} download taget: {out_dir}")
    errors = 0

    for filename, url in items:
        errors += _download_file(
            filename=filename,
            url=url,
            out_dir=out_dir,
            force=force
        )

    return 0 if not errors else errors

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download raw documents (PDF/HTML) into data/raw_docs/*")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Download both PDFs and HTMLs (default)")
    g.add_argument("--pdf", action="store_true", help="Download PDFs only")
    g.add_argument("--html", action="store_true", help="Download HTMLs only")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists")
    return p.parse_args(argv)

def main(argv: list[str] | None=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    download_all = args.all or (not args.pdf and not args.html)
    download_pdf = download_all or args.pdf
    download_html = download_all or args.html

    ensure_project_dirs()
    
    if args.force:
        print("[INFO] Re-download existing files")

    pdf_errors = 0
    if download_pdf:
        print(f"[INFO] Download target: {RAW_DOCS_PDF_DIR}")
        pdf_errors = _download_files(
            items=PDFS,
            out_dir=RAW_DOCS_PDF_DIR,
            force=args.force,
            label="PDF"
        )
        print(f"[INFO] PDF errors: {pdf_errors}")

    html_errors = 0
    if download_html:
        print(f"[INFO] Download target: {RAW_DOCS_HTML_DIR}")
        html_errors = _download_files(
            items=HTMLS,
            out_dir=RAW_DOCS_HTML_DIR,
            force=args.force,
            label="HTML"
        )
        print(f"[INFO] HTML errors: {html_errors}")

    errors = pdf_errors + html_errors
    result = f"{RED}FAIL{RESET}" if errors else f"{GREEN}OK{RESET}"
    print(f"[{result}] Completed with {errors} errors")

    return errors

if __name__ == "__main__":
    raise SystemExit(main())
