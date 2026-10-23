"""Ingest the CGPDTM 'Manual of Designs Practice & Procedure' into a local ChromaDB (./db).

Steps: download PDF if missing -> extract text + tables per page -> chunk -> enrich metadata
(page, chapter, citations) -> embed in batches of 25 with 429 backoff -> persist to ./db.
"""
import re
import shutil
import sys
import time
from pathlib import Path

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

PDF_URLS = [
    "https://ipindia.gov.in/storage/uploads/docs-operator/1ebf65d0-3b78-499a-a9a8-ba28863962f2.pdf",
    "https://www.ipindia.gov.in/writereaddata/Portal/IPOGuidelinesManuals/1_30_1_manual-designs-practice-and-procedure.pdf",
    "https://www.jetro.go.jp/ext_images/world/asia/in/ip/pdf/2011_ishou_02.pdf",
]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
EXPECTED_PAGES = (80, 95)  # accept "~87 pages"

CHAPTER_BY_PAGE = [
    # Verified against the Contents page (p.2) of the 87-page IP India PDF.
    (1, 1, "Cover Page"),
    (2, 2, "Contents"),
    (3, 4, "Preface"),
    (5, 6, "Chapter 1 – Introduction"),
    (7, 10, "Chapter 2 – Important Definitions"),
    (11, 23, "Chapter 3 – Filing of Design Application"),
    (24, 33, "Chapter 4 – Examination and Registration of Designs"),
    (34, 34, "Chapter 5 – Patent Office Journal"),
    (35, 36, "Chapter 6 – Piracy of Registered Designs and Consequences Thereof"),
    (37, 44, "Chapter 7 – Post Registration Procedures"),
    (45, 46, "Chapter 8 – Transfer of Rights"),
    (47, 49, "Chapter 9 – General Services"),
    (50, 50, "Chapter 10 – Use and Acquisition of Registered Design by Government"),
    (51, 53, "Chapter 11 – General Powers of Controller"),
    (54, 57, "Chapter 12 – Evidences etc."),
    (58, 59, "Chapter 13 – Restrictive Conditions in Contract etc."),
    (60, 60, "Chapter 14 – Appeals"),
    (61, 62, "Chapter 15 – Requirements Before Delivery on Sales"),
    (63, 64, "Chapter 16 – Timelines"),
    (65, 85, "Annexure I – Important Case Laws"),
    (86, 87, "Annexure II – Specimen Novelty Statements & Disclaimers"),
]


def get_chapter_for_page(pdf_page_1_indexed: int) -> str:
    for start, end, title in CHAPTER_BY_PAGE:
        if start <= pdf_page_1_indexed <= end:
            return title
    return "Manual of Designs Practice & Procedure"


def extract_citations(text: str) -> str:
    """Extract manual decimal headings (e.g., 03.06.02.02) and statutory margin refs."""
    manual_secs = sorted(set(re.findall(r"\b\d{2}\.\d{2}(?:\.\d{2}){0,2}\b", text)))
    stat_refs = sorted(set(re.findall(
        r"\b(?:Section\s+\d+[A-Za-z]*(?:\s*,\s*\d+)*|Rule\s+\d+[A-Za-z]*|(?:First|Second|Third|Fourth)\s+Schedule|Annexure\s+[IVX0-9]+|Form-\d+)\b",
        text,
        flags=re.IGNORECASE,
    )))
    parts = []
    if manual_secs:
        parts.append("Manual Para " + ", ".join(manual_secs[:3]))
    if stat_refs:
        parts.append(", ".join(stat_refs[:4]))
    return " | ".join(parts) if parts else "General Text"


# --------------------------------------------------------------------------- download
def _pdf_page_count(path: Path):
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001
        return None


def ensure_pdf() -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf = config.PDF_PATH
    if pdf.exists():
        n = _pdf_page_count(pdf)
        if n:
            print(f"[ingest] Using existing PDF: {pdf} ({n} pages)")
            if not (EXPECTED_PAGES[0] <= n <= EXPECTED_PAGES[1]):
                print(f"[ingest] WARNING: expected ~87 pages, found {n}; chapter/page metadata may be off.")
            return pdf
        print("[ingest] Existing PDF is invalid; re-downloading.")
        pdf.unlink()

    tmp = pdf.with_suffix(".tmp")
    for url in PDF_URLS:
        print(f"[ingest] Downloading {url}")
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90)
            r.raise_for_status()
            tmp.write_bytes(r.content)
        except Exception as exc:  # noqa: BLE001 (timeouts, SSL errors, HTTP errors)
            print(f"  failed: {exc}")
            continue
        n = _pdf_page_count(tmp)
        if n and EXPECTED_PAGES[0] <= n <= EXPECTED_PAGES[1]:
            tmp.replace(pdf)
            print(f"  OK: valid PDF with {n} pages -> {pdf}")
            return pdf
        print(f"  rejected: not a valid ~87-page PDF (pages={n})")
    if tmp.exists():
        tmp.unlink()
    raise SystemExit(
        "Could not download the manual from any URL. Download it manually and save it as "
        f"{pdf} (see README), then re-run ingest.py."
    )


# --------------------------------------------------------------------------- extraction
def _clean(txt: str) -> str:
    txt = re.sub(r"[ \t ]+", " ", txt)
    txt = re.sub(r"\s*\n\s*", " ", txt)
    return txt.strip()


def _extract_fitz(path: Path):
    import fitz  # pymupdf

    pages = []
    doc = fitz.open(str(path))
    for i, page in enumerate(doc, start=1):
        items, table_rects = [], []
        try:
            for t in page.find_tables().tables:
                rows = t.extract()
                if not rows or len(rows) < 2 or len(rows[0]) < 2:
                    continue
                lines = [" | ".join(_clean(c or "") for c in r) for r in rows]
                lines = [l for l in lines if l.strip(" |")]
                if not lines:
                    continue
                rect = fitz.Rect(t.bbox)
                table_rects.append(rect)
                items.append((rect.y0, rect.x0, "[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]"))
        except Exception:  # noqa: BLE001 - table detection is best-effort
            pass
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, txt, _no, btype = b[:7]
            if btype != 0:
                continue
            centre = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(r.contains(centre) for r in table_rects):
                continue
            txt = _clean(txt)
            if txt:
                items.append((y0, x0, txt))
        items.sort(key=lambda t: (round(t[0]), t[1]))
        pages.append((i, "\n".join(t[2] for t in items)))
    return pages


def _extract_pdfplumber(path: Path):
    import pdfplumber

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=False) or ""
            for tbl in page.extract_tables() or []:
                rows = [" | ".join(_clean(c or "") for c in r) for r in tbl if r]
                if rows:
                    text += "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"
            pages.append((i, text))
    return pages


def _extract_pypdf(path: Path):
    from pypdf import PdfReader

    return [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages, start=1)]


def extract_pages(path: Path):
    for name, fn in (("pymupdf", _extract_fitz), ("pdfplumber", _extract_pdfplumber), ("pypdf", _extract_pypdf)):
        try:
            pages = fn(path)
            if sum(len(t) for _, t in pages) > 1000:
                print(f"[ingest] Extracted {len(pages)} pages with {name}")
                return pages
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest] {name} unavailable/failed: {exc}")
    raise SystemExit("No PDF extractor produced usable text (scanned PDF?).")


# --------------------------------------------------------------------------- chunking
def build_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n[/TABLE]", "\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for page_no, text in pages:
        text = text.strip()
        if len(text) < 30:
            continue
        chapter = get_chapter_for_page(page_no)
        for idx, piece in enumerate(splitter.split_text(text)):
            citation = extract_citations(piece)
            header = f"[Page {page_no} | {chapter} | {citation}]"
            docs.append(
                Document(
                    page_content=f"{header}: {piece}",
                    metadata={
                        "source": "Manual of Designs Practice & Procedure (CGPDTM, IP India)",
                        "page": page_no,
                        "chapter": chapter,
                        "citation": citation,
                        "chunk": idx,
                    },
                )
            )
    return docs


# --------------------------------------------------------------------------- indexing
def build_db(docs):
    if Path(config.DB_DIR).exists():
        shutil.rmtree(config.DB_DIR)
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.get_embeddings(),
        persist_directory=config.DB_DIR,
    )
    bs = config.EMBED_BATCH_SIZE
    total = len(docs)
    for start in range(0, total, bs):
        batch = docs[start : start + bs]
        config.with_backoff(store.add_documents, batch)
        print(f"[ingest] embedded {min(start + bs, total)}/{total} chunks")
        time.sleep(16)  # free tier ~100 embed requests/min -> stay under it
    return store


def main():
    pdf = ensure_pdf()
    pages = extract_pages(pdf)
    docs = build_chunks(pages)
    print(f"[ingest] {len(docs)} chunks from {len(pages)} pages")
    build_db(docs)
    print(f"[ingest] Done. Vector store saved in {config.DB_DIR}")


if __name__ == "__main__":
    sys.exit(main())
