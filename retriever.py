from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chunker import chunk_text

class GraphRAG:
    def __init__(self, documents, chunk_size=200, chunk_overlap=20, top_k=3, max_hops=1):
        self.documents = documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_hops = max_hops
        self.nodes = []
        self.adj = {}
        self._build()
        self._vectorize()

    def _build(self):
        for doc in self.documents:
            chunks = chunk_text(doc, self.chunk_size, self.chunk_overlap)
            base = len(self.nodes)
            for i, ch in enumerate(chunks):
                node_id = base + i
                self.nodes.append({"id": node_id, "text": ch})
                self.adj[node_id] = set()
                if i > 0:
                    self.adj[node_id - 1].add(node_id)
                    self.adj[node_id].add(node_id - 1)

    def _vectorize(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf = self.vectorizer.fit_transform([n["text"] for n in self.nodes])
        sim = cosine_similarity(self.tfidf)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if sim[i, j] > 0.4:
                    self.adj[i].add(j)
                    self.adj[j].add(i)

    def retrieve(self, query: str) -> list[str]:
        if not self.nodes:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf)[0]
        top = sims.argsort()[::-1][:self.top_k]
        visited = set()
        frontier = set(top)
        for _ in range(self.max_hops + 1):
            next_frontier = set()
            for n in frontier:
                if n not in visited:
                    visited.add(n)
                    next_frontier.update(self.adj.get(n, set()))
            frontier = next_frontier
        return [self.nodes[i]["text"] for i in sorted(visited)]
