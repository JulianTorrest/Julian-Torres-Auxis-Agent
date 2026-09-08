import json
import os
import sqlite3
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    def __init__(self, max_features=256):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    def fit(self, texts):
        self.vectorizer.fit(texts)
    def embed(self, text):
        return self.vectorizer.transform([text]).toarray().astype("float32")[0]
    def __call__(self, texts):
        return self.vectorizer.transform(texts).toarray().astype("float32")

class SQLiteVectorStore:
    def __init__(self, db_path="vector_db/vectors.sqlite"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.embedder = Embedder()
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, text TEXT, metadata TEXT, embedding BLOB)")
        self._conn.commit()
    def add_texts(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        self.embedder.fit(texts)
        for t, m, i in zip(texts, metadatas, ids):
            emb = self.embedder.embed(t)
            self._conn.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)", (i, t, json.dumps(m, ensure_ascii=False), pickle.dumps(emb)))
        self._conn.commit()
    def search(self, query: str, k: int = 5) -> List[Tuple[float, str, Dict]]:
        q = self.embedder.embed(query)
        rows = self._conn.execute("SELECT text, metadata, embedding FROM chunks").fetchall()
        results = []
        for text, meta, emb_blob in rows:
            emb = pickle.loads(emb_blob)
            sim = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-10))
            results.append((sim, text, json.loads(meta)))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

class FAISSVectorStore:
    def __init__(self, index_path="vector_db/faiss.index"):
        import faiss
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        self.index_path = index_path
        self.embedder = Embedder()
        self._texts = []
        self._metas = []
        self._index = None
    def add_texts(self, texts, metadatas, ids):
        self.embedder.fit(texts)
        vecs = self.embedder(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-10)
        import faiss
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        self._texts = texts
        self._metas = metadatas
        faiss.write_index(self._index, self.index_path)
    def search(self, query, k=5):
        if self._index is None:
            return []
        q = self.embedder.embed(query)
        q = q / (np.linalg.norm(q) + 1e-10)
        import faiss
        scores, indices = self._index.search(np.array([q], dtype="float32"), k)
        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._texts):
                results.append((float(sc), self._texts[idx], self._metas[idx]))
        return results

class ChromaVectorStore:
    def __init__(self, persist_path="vector_db/chroma"):
        os.makedirs(persist_path, exist_ok=True)
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection("corpus")
        self.embedder = Embedder()
    def add_texts(self, texts, metadatas, ids):
        self.embedder.fit(texts)
        embeddings = self.embedder(texts).tolist()
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    def search(self, query, k=5):
        q = self.embedder.embed(query).tolist()
        out = self.collection.query(query_embeddings=[q], n_results=k)
        results = []
        for i in range(len(out["documents"][0])):
            score = 1.0 - float(out["distances"][0][i])
            results.append((score, out["documents"][0][i], out["metadatas"][0][i]))
        return results

class HybridVectorStore:
    def __init__(self, backend_order: Optional[List[str]] = None, base_dir="vector_db"):
        if backend_order is None:
            backend_order = ["chroma", "faiss", "sqlite"]
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.backend_name = None
        self.backend = None
        self.errors = {}
        for name in backend_order:
            try:
                if name == "chroma":
                    self.backend = ChromaVectorStore(os.path.join(base_dir, "chroma"))
                elif name == "faiss":
                    self.backend = FAISSVectorStore(os.path.join(base_dir, "faiss.index"))
                elif name == "sqlite":
                    self.backend = SQLiteVectorStore(os.path.join(base_dir, "vectors.sqlite"))
                self.backend_name = name
                break
            except Exception as e:
                self.errors[name] = str(e)
    def add_texts(self, texts, metadatas, ids):
        return self.backend.add_texts(texts, metadatas, ids)
    def search(self, query, k=5):
        return self.backend.search(query, k)

def index_corpus(index_path="corpus_index.jsonl", base_dir="vector_db", backend_order=None):
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    texts = [e["text"] for e in entries]
    metadatas = [{k: v for k, v in e.items() if k != "text"} for e in entries]
    ids = [e["id"] for e in entries]
    store = HybridVectorStore(backend_order, base_dir)
    store.add_texts(texts, metadatas, ids)
    return store, len(entries)
