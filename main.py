import os
import time
import sqlite3
from pathlib import Path
from typing import List, Tuple

import hnswlib
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

SUPPORTED_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".py", ".java", ".js", ".go", ".rs", ".c", ".cpp", ".cs", ".html", ".css"}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "hnsw.bin"
DB_FILE = "index.sqlite"


def ensure_repo(repo: Path):
    repo.mkdir(parents=True, exist_ok=True)
    dbp = repo / DB_FILE
    if not dbp.exists():
        with sqlite3.connect(dbp) as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS items(
                    id INTEGER PRIMARY KEY,
                    path TEXT,
                    chunk_idx INTEGER,
                    text TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_items_path ON items(path);
                CREATE TABLE IF NOT EXISTS files(
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER
                );
                """
            )


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= chunk_size:
            chunks.append(" ".join(cur))
            # overlap
            if overlap > 0:
                tail = " ".join(cur)[-overlap:]
                cur = tail.split()
                cur_len = len(" ".join(cur))
            else:
                cur = []
                cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def read_text_from_file(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".pdf":
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(str(p))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def iter_docs(folder: Path) -> List[Path]:
    for root, _, files in os.walk(folder):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in SUPPORTED_EXT or p.suffix.lower() == ".pdf":
                yield p


def get_ann_index(repo: Path, dim: int, target_size: int) -> hnswlib.Index:
    idx_path = repo / INDEX_FILE
    if idx_path.exists():
        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.load_index(str(idx_path))
        try:
            idx.resize_index(max(idx.get_max_elements(), target_size))
        except Exception:
            pass
        idx.set_ef(64)
        return idx
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=max(target_size, 1000), ef_construction=200, M=16)
    idx.set_ef(64)
    return idx


def save_ann_index(repo: Path, idx: hnswlib.Index):
    idx.save_index(str(repo / INDEX_FILE))


def current_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(1) FROM items").fetchone()
    return int(row[0] or 0)


def index_folder(repo: Path, data_path: Path):
    ensure_repo(repo)
    model = load_model()
    dim = model.get_sentence_embedding_dimension()

    dbp = repo / DB_FILE
    with sqlite3.connect(dbp) as conn:
        conn.execute("PRAGMA synchronous=NORMAL")
        before = current_count(conn)

        # gather files and determine changes
        to_index: List[Tuple[Path, float, int]] = []
        for p in iter_docs(data_path):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            rec = conn.execute("SELECT mtime,size FROM files WHERE path=?", (str(p),)).fetchone()
            if not rec or rec[0] != st.st_mtime or rec[1] != st.st_size:
                to_index.append((p, st.st_mtime, st.st_size))

        if not to_index:
            print("No changes detected.")
            return

        idx = get_ann_index(repo, dim, target_size=before + len(to_index) * 16)

        t0 = time.time()
        add_vecs = []
        add_ids = []
        added_chunks = 0

        for p, mtime, size in to_index:
            # remove existing chunks for file (keep labels gap; OK)
            conn.execute("DELETE FROM items WHERE path=?", (str(p),))
            conn.execute(
                "INSERT OR REPLACE INTO files(path,mtime,size) VALUES(?,?,?)",
                (str(p), mtime, size),
            )

            text = read_text_from_file(p)
            if not text.strip():
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue

            # embed in small batches
            batch_size = 64
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                for j, (emb, t) in enumerate(zip(embs, batch)):
                    cur = conn.execute(
                        "INSERT INTO items(path,chunk_idx,text) VALUES(?,?,?)",
                        (str(p), i + j, t),
                    )
                    label = cur.lastrowid
                    add_vecs.append(emb)
                    add_ids.append(label)
                    added_chunks += 1
            conn.commit()

        if add_vecs:
            import numpy as np
            vecs = np.vstack(add_vecs)
            ids = np.array(add_ids)
            idx.add_items(vecs, ids)
            save_ann_index(repo, idx)

        dt = time.time() - t0
        print(f"Indexed {len(to_index)} files, {added_chunks} chunks in {dt:.2f}s")


def query(repo: Path, q: str, k: int = 5):
    ensure_repo(repo)
    model = load_model()
    dim = model.get_sentence_embedding_dimension()

    idx_path = repo / INDEX_FILE
    if not idx_path.exists():
        print("Index not found. Run 'index' first.")
        return

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(str(idx_path))
    idx.set_ef(max(64, k))

    t0 = time.time()
    qv = model.encode([q], normalize_embeddings=True)
    import numpy as np
    labels, dists = idx.knn_query(np.asarray(qv), k=k)
    latency = (time.time() - t0) * 1000

    labels = labels[0]
    dists = dists[0]

    with sqlite3.connect(repo / DB_FILE) as conn:
        for rank, (lab, dist) in enumerate(zip(labels, dists), start=1):
            row = conn.execute("SELECT path, chunk_idx, text FROM items WHERE id=?", (int(lab),)).fetchone()
            if not row:
                continue
            path, chunk_idx, text = row
            print(f"[{rank}] score={1-dist:.4f} path={path}#chunk{chunk_idx}")
            preview = text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print("    ", preview)

    print(f"Latency: {latency:.1f} ms")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Local RAG CLI (HNSW + sentence-transformers)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("repo")
    p_index.add_argument("path")

    p_query = sub.add_parser("query")
    p_query.add_argument("repo")
    p_query.add_argument("query")
    p_query.add_argument("--k", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "index":
        index_folder(Path(args.repo), Path(args.path))
    elif args.cmd == "query":
        query(Path(args.repo), args.query, k=args.k)


if __name__ == "__main__":
    main()
