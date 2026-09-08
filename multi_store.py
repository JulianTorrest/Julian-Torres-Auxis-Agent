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
        self.ollama_model = os.environ.get("OLLAMA_EMBED_MODEL")
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_emb = None
        if self.ollama_model:
            try:
                from langchain_ollama import OllamaEmbeddings
                self.ollama_emb = OllamaEmbeddings(model=self.ollama_model, base_url=self.ollama_base_url)
            except Exception:
                pass
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._use_ollama = self.ollama_emb is not None
    def fit(self, texts):
        if not self._use_ollama:
            self.vectorizer.fit(texts)
    def embed(self, text):
        if self._use_ollama:
            try:
                return np.array(self.ollama_emb.embed_query(text), dtype="float32")
            except Exception:
                pass
        return self.vectorizer.transform([text]).toarray().astype("float32")[0]
    def __call__(self, texts):
        if self._use_ollama:
            try:
                return np.array(self.ollama_emb.embed_documents(texts), dtype="float32")
            except Exception:
                pass
        return self.vectorizer.transform(texts).toarray().astype("float32")

class SQLiteVectorStore:
    def __init__(self, db_path="vector_db/embeddings.sqlite"):
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
    def get(self, id: str) -> Tuple[str, Dict]:
        row = self._conn.execute("SELECT text, metadata FROM chunks WHERE id=?", (id,)).fetchone()
        if not row:
            return "", {}
        return row[0], json.loads(row[1])
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
    def __init__(self, index_path="vector_db/vectors.faiss"):
        import faiss
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        self.index_path = index_path
        self.embedder = Embedder()
        self._texts = {}
        self._metas = {}
        self._ids = []
        self._index = None
    def add_texts(self, texts, metadatas, ids):
        import faiss
        self.embedder.fit(texts)
        vecs = self.embedder(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-10)
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        for i, t, m in zip(ids, texts, metadatas):
            self._ids.append(i)
            self._texts[i] = t
            self._metas[i] = m
        faiss.write_index(self._index, self.index_path)
    def get(self, id):
        return self._texts.get(id, ""), self._metas.get(id, {})
    def search(self, query, k=5):
        if self._index is None:
            return []
        import faiss
        q = self.embedder.embed(query)
        q = q / (np.linalg.norm(q) + 1e-10)
        scores, indices = self._index.search(np.array([q], dtype="float32"), k)
        results = []
        for sc, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._ids):
                doc_id = self._ids[idx]
                results.append((float(sc), self._texts[doc_id], self._metas[doc_id]))
        return results

class ChromaVectorStore:
    def __init__(self, persist_path="vector_db/documents"):
        import chromadb
        os.makedirs(persist_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection("corpus")
        self.embedder = Embedder()
    def add_texts(self, texts, metadatas, ids):
        self.embedder.fit(texts)
        embeddings = self.embedder(texts).tolist()
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    def get(self, id):
        out = self.collection.get(ids=[id], include=["documents", "metadatas"])
        if not out["documents"]:
            return "", {}
        return out["documents"][0], out["metadatas"][0]
    def search(self, query, k=5):
        q = self.embedder.embed(query).tolist()
        out = self.collection.query(query_embeddings=[q], n_results=k)
        results = []
        for i in range(len(out["documents"][0])):
            score = 1.0 - float(out["distances"][0][i])
            results.append((score, out["documents"][0][i], out["metadatas"][0][i]))
        return results

class TriadStore:
    def __init__(self, base_dir="vector_db"):
        os.makedirs(base_dir, exist_ok=True)
        self.sqlite = None
        self.faiss = None
        self.chroma = None
        self.errors = {}
        try:
            self.sqlite = SQLiteVectorStore(os.path.join(base_dir, "embeddings.sqlite"))
        except Exception as e:
            self.errors["sqlite"] = str(e)
        try:
            self.faiss = FAISSVectorStore(os.path.join(base_dir, "vectors.faiss"))
        except Exception as e:
            self.errors["faiss"] = str(e)
        try:
            self.chroma = ChromaVectorStore(os.path.join(base_dir, "documents"))
        except Exception as e:
            self.errors["chroma"] = str(e)
        self.active = [n for n in ["sqlite", "faiss", "chroma"] if getattr(self, n) is not None]
    def add_texts(self, texts, metadatas, ids):
        if self.sqlite: self.sqlite.add_texts(texts, metadatas, ids)
        if self.faiss: self.faiss.add_texts(texts, metadatas, ids)
        if self.chroma: self.chroma.add_texts(texts, metadatas, ids)
    def search(self, query, k=5):
        if self.faiss:
            return self.faiss.search(query, k)
        if self.sqlite:
            return self.sqlite.search(query, k)
        if self.chroma:
            return self.chroma.search(query, k)
        return []
    def status(self):
        return self.active

def index_corpus(index_path="corpus_index.jsonl", base_dir="vector_db"):
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    texts = [e["text"] for e in entries]
    metadatas = [{k: v for k, v in e.items() if k != "text"} for e in entries]
    ids = [e["id"] for e in entries]
    store = TriadStore(base_dir)
    store.add_texts(texts, metadatas, ids)
    return store, len(entries)
