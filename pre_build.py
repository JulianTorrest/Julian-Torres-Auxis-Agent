import json
import os
import pickle
import build_corpus
from retriever import GraphRAG
import multi_store
import data_catalog
from llm_clients import get_llm

def ensure_built(base_dir="prebuilt"):
    os.makedirs(base_dir, exist_ok=True)
    rag_path = os.path.join(base_dir, "graph_rag.pkl")
    if not os.path.exists(rag_path):
        pdf_count, index_path, full_index_path, chunk_count = build_corpus.build()
        fulls = []
        with open(full_index_path, "r", encoding="utf-8") as f:
            for line in f:
                fulls.append(json.loads(line)["text"])
        rag = GraphRAG(fulls, chunk_size=300, chunk_overlap=30)
        with open(rag_path, "wb") as f:
            pickle.dump(rag, f)
        multi_store.index_corpus(index_path)
        llm = get_llm(["fake"], {}, 0)
        data_catalog.build_all(rag, llm=llm)
    return rag_path

def load_rag(base_dir="prebuilt"):
    ensure_built(base_dir)
    with open(os.path.join(base_dir, "graph_rag.pkl"), "rb") as f:
        return pickle.load(f)
