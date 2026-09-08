import json
import os
import networkx as nx
from sklearn.cluster import KMeans
from pyvis.network import Network
from text_chunker import chunk_by_sections
from summarizer import summarize
import build_corpus

def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def _build_pyvis_graph(g, entries, path, max_nodes=200):
    topic_colors = {
        "economia": "#1f77b4",
        "legal": "#ff7f0e",
        "estrategia": "#2ca02c",
        "organizacional": "#d62728",
        "datos": "#9467bd",
    }
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    nodes = list(g.nodes())[:max_nodes]
    sub_g = g.subgraph(nodes)
    for n in sub_g.nodes():
        e = entries[n]
        net.add_node(
            n,
            label=e["topic"],
            color=topic_colors.get(e["topic"], "#999999"),
            title=f"{e['subtopic']} ({e['lang']})<br>{e['text'][:150]}...",
        )
    for a, b in sub_g.edges():
        net.add_edge(a, b)
    net.write_html(path)

def build_all(rag, output_dir="data_catalog", llm=None):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists("corpus_index.jsonl") or not os.path.exists("corpus_full.jsonl"):
        build_corpus.build()
    entries = _load_jsonl("corpus_index.jsonl")
    fulls = _load_jsonl("corpus_full.jsonl")

    doc_index = []
    for doc in fulls:
        sections = chunk_by_sections(doc["text"])
        chunk_count = sum(1 for e in entries if e["source"] == doc["source"])
        summary = summarize(doc["text"], llm) if llm else "Resumen no generado"
        doc_index.append({
            "source": doc["source"],
            "topic": doc["topic"],
            "subtopic": doc["subtopic"],
            "lang": doc["lang"],
            "section_count": len(sections),
            "chunk_count": chunk_count,
            "summary": summary,
        })
    with open(os.path.join(output_dir, "document_index.jsonl"), "w", encoding="utf-8") as f:
        for d in doc_index:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    data_dictionary = [
        {"field": "id", "type": "string", "description": "Identificador unico del chunk", "source": "corpus_index"},
        {"field": "source", "type": "string", "description": "Ruta del PDF origen", "source": "corpus_index/corpus_full"},
        {"field": "topic", "type": "string", "description": "Tema del documento", "source": "corpus_index/corpus_full"},
        {"field": "subtopic", "type": "string", "description": "Subtema del documento", "source": "corpus_index/corpus_full"},
        {"field": "lang", "type": "string", "description": "Idioma del documento", "source": "corpus_index/corpus_full"},
        {"field": "chunk_index", "type": "integer", "description": "Indice del chunk dentro del documento", "source": "corpus_index"},
        {"field": "text", "type": "string", "description": "Contenido textual", "source": "corpus_index/corpus_full"},
        {"field": "embedding", "type": "float32[]", "description": "Vector TF-IDF del chunk", "source": "vector_db/embeddings.sqlite"},
        {"field": "summary", "type": "string", "description": "Resumen generado por LLM", "source": "document_index"},
    ]
    with open(os.path.join(output_dir, "data_dictionary.json"), "w", encoding="utf-8") as f:
        json.dump(data_dictionary, f, ensure_ascii=False, indent=2)

    glossary = []
    for topic, subtopics in build_corpus.TOPICS.items():
        glossary.append({"term": topic, "category": "tema", "definition": f"Area tematica de {topic}."})
        for sub in subtopics:
            glossary.append({"term": sub.replace("_", " "), "category": "subtema", "definition": f"Concepto especifico dentro de {topic}: {sub.replace('_', ' ')}."})
    with open(os.path.join(output_dir, "glossary.json"), "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    field_counts = {}
    for e in entries:
        for k in e.keys():
            field_counts[k] = field_counts.get(k, 0) + 1
    metadata_inventory = [{"field": k, "count": v} for k, v in field_counts.items()]
    with open(os.path.join(output_dir, "metadata_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_inventory, f, ensure_ascii=False, indent=2)

    hierarchy = {}
    for doc in fulls:
        topic = doc["topic"]
        sub = doc["subtopic"]
        lang = doc["lang"]
        hierarchy.setdefault(topic, {}).setdefault(sub, {}).setdefault(lang, []).append(doc["source"])
    with open(os.path.join(output_dir, "hierarchy.json"), "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)

    n_clusters = min(5, len(entries))
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(rag.tfidf)
    else:
        labels = [0] * len(entries)
    cluster_index = {}
    for i, (e, label) in enumerate(zip(entries, labels)):
        cluster_index.setdefault(int(label), {}).setdefault(e["topic"], []).append(e["subtopic"])
    with open(os.path.join(output_dir, "topic_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_index, f, ensure_ascii=False, indent=2)

    g = nx.Graph()
    for i, node in enumerate(rag.nodes):
        e = entries[i]
        g.add_node(i, topic=e["topic"], subtopic=e["subtopic"], lang=e["lang"])
    for i, neighbors in rag.adj.items():
        for j in neighbors:
            if i < j:
                g.add_edge(i, j)
    nx.write_gml(g, os.path.join(output_dir, "corpus_graph.gml"))
    _build_pyvis_graph(g, entries, os.path.join(output_dir, "corpus_graph.html"), max_nodes=200)

    return output_dir
