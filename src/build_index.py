from __future__ import annotations

from pathlib import Path
import re
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TEXT_DIR = Path("data/text")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


SECTION_RE = re.compile(
    r"(?im)^\s*(section|sec\.|s\.)\s*(\d+[A-Za-z]?)\s*[\:\.\-]?\s*(.*)$"
)


def detect_section(paragraph: str) -> str | None:
    """
    Best-effort: if a paragraph looks like a section header, capture it.
    Examples:
      "Section 15. Grounds for ejectment"
      "S. 15 - ..."
    """
    m = SECTION_RE.match(paragraph.strip())
    if not m:
        return None
    num = m.group(2).strip()
    title = (m.group(3) or "").strip()
    if title:
        return f"Section {num} — {title}"
    return f"Section {num}"


def split_into_paragraphs(text: str) -> List[str]:
    # Normalize newlines, split on blank lines
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return parts


def chunk_paragraphs(
    paragraphs: List[str],
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
) -> List[str]:
    """
    Build chunks by concatenating paragraphs up to a char budget.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for p in paragraphs:
        add_len = len(p) + 2
        if buf and (buf_len + add_len > chunk_chars):
            chunk = "\n\n".join(buf).strip()
            if chunk:
                chunks.append(chunk)

            # overlap: keep last overlap_chars from chunk as starter
            if overlap_chars > 0:
                tail = chunk[-overlap_chars:]
                buf = [tail]
                buf_len = len(tail)
            else:
                buf = []
                buf_len = 0

        buf.append(p)
        buf_len += add_len

    final = "\n\n".join(buf).strip()
    if final:
        chunks.append(final)

    return chunks


def load_documents() -> Tuple[List[str], List[Dict]]:
    documents: List[str] = []
    metadatas: List[Dict] = []

    for txt_file in sorted(TEXT_DIR.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8", errors="ignore")
        paragraphs = split_into_paragraphs(raw)

        # Track the most recent detected section header as we walk
        current_section = ""

        # First pass: update current_section when we see a section header
        annotated_paras: List[Tuple[str, str]] = []
        for p in paragraphs:
            sec = detect_section(p)
            if sec:
                current_section = sec
            annotated_paras.append((p, current_section))

        # Second pass: chunk while carrying section labels (best-effort)
        # We'll chunk by text but assign section = most common/last section in chunk.
        paras_only = [p for (p, _) in annotated_paras]
        chunks = chunk_paragraphs(paras_only, chunk_chars=1800, overlap_chars=250)

        # For each chunk, pick the last seen section that appears within it
        for ci, chunk in enumerate(chunks):
            section_for_chunk = ""
            # try to find the last section header present in this chunk
            chunk_paras = split_into_paragraphs(chunk)
            for cp in chunk_paras:
                sec = detect_section(cp)
                if sec:
                    section_for_chunk = sec

            documents.append(chunk)
            metadatas.append(
                {
                    "source": txt_file.name,
                    "section": section_for_chunk,
                    "text": chunk,
                    "chunk_id": f"{txt_file.stem}__{ci}",
                }
            )

    return documents, metadatas


def main():
    print("Loading documents...")
    docs, metadata = load_documents()

    print("Total chunks:", len(docs))
    if len(docs) == 0:
        raise SystemExit("No .txt files found in data/text/")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("✅ FAISS index built successfully.")
    print("FAISS ntotal:", index.ntotal, "| dim:", index.d)
    print("Example metadata keys:", list(metadata[0].keys()))


if __name__ == "__main__":
    main()
