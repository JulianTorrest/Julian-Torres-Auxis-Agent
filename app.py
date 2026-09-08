import streamlit as st
import streamlit.components.v1 as components
import os
import time
import json
import pandas as pd
from llm_clients import get_llm
from orchestrator import create_workflow
from retriever import GraphRAG
from pii_guard import redact_pii
from prompt_guard import is_injection
from governance import check_policy
from pdf_generator import generate_pdfs
from observability import ObservabilityManager
import build_corpus
import multi_store
import pre_build
import mermaid_diagrams

st.set_page_config(page_title="Agentic AI - Auxis", layout="wide")

DEFAULT_DOCS = [
    "LangGraph permite construir aplicaciones agenticas como grafos de estados con ciclos y persistencia.",
    "La gobernanza de IA define politicas de uso etico, seguro y responsable incluyendo proteccion de datos y control de acceso.",
    "TF-IDF pondera terminos segun su frecuencia local en un documento y su rareza en la coleccion.",
    "El chunking divide textos largos en fragmentos solapados para indexacion y recuperacion eficiente.",
    "Un orquestador dirige la ejecucion entre modulos de sanitizacion, validacion, recuperacion y generacion.",
]

DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "gemini": "gemini-1.5-flash",
    "groq": "llama3-8b-8192",
    "mistral": "mistral-small-latest",
    "ollama": "llama3.2",
    "deepseek": "deepseek-chat",
}

if "documents" not in st.session_state:
    st.session_state.documents = DEFAULT_DOCS

with st.sidebar:
    st.header("Perfil del agente")
    profiles = {
        "Fast (Ollama router)": {"selected_providers": ["ollama"], "mode": "router", "max_retries": 0},
        "Calidad (MoE Ollama+OpenAI)": {"selected_providers": ["ollama", "openai"], "mode": "moe", "max_retries": 2},
        "Redundante (Parallel)": {"selected_providers": ["ollama", "openai", "gemini"], "mode": "parallel", "max_retries": 0},
        "Seguro (fake)": {"selected_providers": ["fake"], "mode": "router", "max_retries": 0},
    }
    profile = st.selectbox("Selecciona un perfil", ["Personalizado"] + list(profiles.keys()), key="profile")
    if st.button("Aplicar perfil"):
        if profile in profiles:
            st.session_state["selected_providers"] = profiles[profile]["selected_providers"]
            st.session_state["mode"] = profiles[profile]["mode"]
            st.session_state["max_retries"] = profiles[profile]["max_retries"]
            st.success(f"Perfil '{profile}' aplicado. Haz clic en 'Actualizar LLM' para activarlo.")

    st.divider()
    st.header("Configuracion del LLM")
    st.session_state.setdefault("selected_providers", ["fake"])
    st.session_state.setdefault("mode", "router")
    st.session_state.setdefault("max_retries", 1)
    all_providers = ["fake", "openai", "gemini", "groq", "mistral", "ollama", "deepseek"]
    selected_providers = st.multiselect("Proveedores activos", all_providers, key="selected_providers")

    configs = {}
    for p in selected_providers:
        if p == "fake":
            continue
        with st.expander(f"Configuracion {p.upper()}"):
            model = st.text_input(f"Modelo de {p}", value=DEFAULT_MODELS.get(p, ""), key=f"{p}_model")
            if p == "ollama":
                base_url = st.text_input(f"Base URL de {p}", value="http://localhost:11434", key=f"{p}_base")
                configs[p] = {"model": model, "base_url": base_url}
            else:
                api_key = st.text_input(f"API key de {p}", type="password", key=f"{p}_key")
                configs[p] = {"api_key": api_key, "model": model}

    modes = ["router", "parallel", "moe", "retry"]
    mode = st.selectbox("Modo de generacion", modes, key="mode")
    if mode == "router" and len(selected_providers) > 1:
        st.session_state.setdefault("router_provider", selected_providers[0])
        if st.session_state["router_provider"] not in selected_providers:
            st.session_state["router_provider"] = selected_providers[0]
        router_provider = st.selectbox("Proveedor para router", selected_providers, key="router_provider")
    else:
        router_provider = selected_providers[0] if selected_providers else "fake"
    max_retries = st.slider("Reintentos MoE / retry", 0, 3, key="max_retries")

    if st.button("Actualizar LLM"):
        st.session_state.providers = selected_providers
        st.session_state.configs = configs
        st.session_state.mode = mode
        st.session_state.selected_provider = router_provider
        st.session_state.max_retries = max_retries
        st.session_state.llm = get_llm(selected_providers, configs, max_retries)
        st.success("LLM actualizado")

    st.divider()
    st.subheader("Ollama local")
    ollama_host = st.text_input("Ollama host (embeddings/listado)", value="http://localhost:11434", key="ollama_host")
    ollama_embed_model = st.text_input("Modelo de embeddings (Ollama)", value="", help="Ej: nomic-embed-text; dejar vacio para usar TF-IDF", key="ollama_embed_model")
    os.environ["OLLAMA_BASE_URL"] = ollama_host
    if ollama_embed_model:
        os.environ["OLLAMA_EMBED_MODEL"] = ollama_embed_model
    if st.button("Listar modelos Ollama"):
        try:
            import ollama
            client = ollama.Client(host=ollama_host)
            models = client.list()
            names = [m.get("name") or m.get("model") for m in models.get("models", [])]
            st.write("Modelos locales encontrados:", names)
        except Exception as e:
            st.error(f"No se pudo conectar con Ollama en {ollama_host}: {e}")

    st.divider()
    st.header("Gobernanza y Observabilidad")
    user = st.text_input("Usuario", value="anon")
    st.caption("Configura LANGSMITH_API_KEY y/o LANGFUSE_* como variables de entorno para trazas externas.")

    st.divider()
    st.header("Corpus del RAG")
    docs_text = st.text_area(
        "Documentos (uno por linea)",
        value="\n".join(st.session_state.documents),
        height=150,
    )
    if st.button("Actualizar corpus"):
        st.session_state.documents = [d.strip() for d in docs_text.split("\n") if d.strip()]
        st.success("Corpus actualizado")

llm = st.session_state.get("llm", get_llm(["fake"], {}, 1))
with st.spinner("Cargando RAG pre-generado desde el corpus..."):
    rag = pre_build.load_rag()
tracer = ObservabilityManager()
workflow = create_workflow(rag, llm, tracer, session_id=user, token_budget=2000)

st.title("Prueba Tecnica - Agente IA con LangGraph + Router/MoE")

tab_query, tab_rag, tab_gov, tab_trace, tab_vector, tab_temas, tab_archi, tab_pdf = st.tabs([
    "Consulta", "RAG y Grafo", "Gobernanza", "Observabilidad", "Base Vectorial", "Temas y Red", "Arquitectura", "PDFs"
])

with tab_query:
    query = st.text_area("Escribe tu consulta", height=80)
    if st.button("Ejecutar agente"):
        final = workflow.invoke({
            "query": query,
            "user": user,
            "start_time": time.time(),
            "mode": st.session_state.get("mode", "router"),
            "selected_provider": st.session_state.get("selected_provider", "fake"),
            "max_retries": st.session_state.get("max_retries", 1),
        })
        st.session_state.result = final
        st.subheader("Respuesta")
        st.write(final["answer"])
        st.caption(f"Proveedor seleccionado: {final.get('answer_provider', 'N/A')} | Latencia: {final.get('latency_ms', 0):.1f} ms")

with tab_rag:
    st.subheader("Chunks y grafo")
    st.json({
        "nodos": len(rag.nodes),
        "aristas_por_nodo": {k: list(v) for k, v in rag.adj.items()},
        "forma_tfidf": rag.tfidf.shape,
    })
    st.subheader("Top chunks recuperados (consulta de ejemplo)")
    example = st.text_input("Consulta de prueba", value="que es LangGraph")
    if example:
        chunks = rag.retrieve(example)
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}:** {c}")

with tab_gov:
    st.subheader("Input Guardian (encoding, SQL, topicos, moderacion, rate limit)")
    test = st.text_input("Texto a validar", value="", key="guard_test")
    if test:
        from guardian import InputGuardian
        g = InputGuardian().check(user, test)
        st.json(g)
    st.subheader("Output Guardian (PII, credenciales, moderacion)")
    out = st.text_input("Texto de salida a validar", value="", key="guard_out")
    if out:
        from guardian import output_guard
        st.json({"output_guard": output_guard(out)})
    st.subheader("Cache semantico")
    from semantic_cache import SemanticCache
    st.json(SemanticCache().stats())

with tab_trace:
    st.subheader("Trazas del agente actual")
    if "result" in st.session_state:
        st.json(st.session_state.result)
    else:
        st.info("Ejecuta una consulta para ver el trazo.")
    st.subheader("Historial de consultas")
    user_filter = st.text_input("Filtrar por usuario", value="", key="trace_user")
    limit = st.slider("Mostrar", 5, 100, 20, key="trace_limit")
    for row in tracer.local.list(user=user_filter or None, limit=limit):
        with st.expander(f"{row['timestamp']} | {row['user']} | {row['query'][:60]}..."):
            st.json(row)
    st.subheader("Reporte de auditoria")
    if st.button("Generar audit report"):
        report = tracer.get_audit_report(user=user_filter or None)
        st.json({"total": len(report), "muestra": report[:5]})
    st.subheader("Estadisticas de moderacion")
    if st.button("Generar moderation stats"):
        st.json(tracer.get_moderation_stats())

with tab_vector:
    st.header("Corpus de 100 PDFs (ES/EN)")
    if st.button("Generar 100 PDFs y corpus"):
        with st.spinner("Generando PDFs en economia, legal, estrategia, organizacional y datos..."):
            pdf_count, index_path, full_index_path, chunk_count = build_corpus.build()
            st.session_state.corpus_index = index_path
            st.session_state.corpus_dir = os.path.dirname(index_path)
            st.success(f"Generados {pdf_count} PDFs con {chunk_count} chunks. Indices: {index_path}, {full_index_path}")

    if st.button("Indexar en base de datos vectorial"):
        if "corpus_index" not in st.session_state:
            st.warning("Primero genera el corpus.")
        else:
            with st.spinner("Indexando en SQLite (embeddings) + FAISS (vectores) + Chroma (documentos)..."):
                store, n = multi_store.index_corpus(st.session_state.corpus_index)
                st.session_state.vector_store = store
                st.success(f"Indexados {n} chunks. Backends activos: {', '.join(store.status())}")

    st.header("Buscar en base de datos vectorial")
    query_db = st.text_input("Consulta en el corpus", value="")
    k = st.slider("Top k", 1, 20, 5)
    if query_db and "vector_store" in st.session_state:
        results = st.session_state.vector_store.search(query_db, k)
        for score, text, meta in results:
            st.markdown(f"**Score:** {score:.3f} | **Tema:** {meta['topic']} | **Subtema:** {meta['subtopic']} | **Idioma:** {meta['lang']}")
            st.write(text)
            st.caption(f"Fuente: {meta['source']}")

with tab_temas:
    vista = st.radio("Vista del catalogo", [
        "Distribucion", "Red interactiva", "Red persistente", "Indice de documentos",
        "Diccionario de datos", "Glosario", "Inventario de metadatos", "Tabla jerarquica", "Clusters"
    ], horizontal=True)

    if os.path.exists("corpus_index.jsonl"):
        with open("corpus_index.jsonl", "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
    else:
        entries = []
    df = pd.DataFrame(entries)

    if vista == "Distribucion":
        st.header("Distribucion del corpus")
        if not df.empty:
            st.dataframe(df[["topic", "subtopic", "lang", "chunk_index", "source"]].head(20))
            st.subheader("PDFs por tema e idioma")
            st.bar_chart(df.groupby(["topic", "lang"]).size().unstack(fill_value=0))
            st.subheader("Chunks por subtema")
            st.bar_chart(df["subtopic"].value_counts().head(20))

    elif vista == "Red interactiva":
        st.header("Red interactiva de chunks (GraphRAG)")
        topic_colors = {
            "economia": "#1f77b4",
            "legal": "#ff7f0e",
            "estrategia": "#2ca02c",
            "organizacional": "#d62728",
            "datos": "#9467bd",
        }
        try:
            from pyvis.network import Network
            HAS_PYVIS = True
        except Exception:
            HAS_PYVIS = False
        if HAS_PYVIS and entries:
            topics = sorted(df["topic"].unique()) if not df.empty else []
            selected = st.multiselect("Temas a visualizar", topics, default=topics[:2] if topics else [])
            max_nodes = st.slider("Maximo de nodos", 10, 300, 80)
            allowed = [(i, e) for i, e in enumerate(entries) if e["topic"] in selected][:max_nodes]
            allowed_set = {i for i, e in allowed}
            net = Network(height="520px", width="100%", bgcolor="#ffffff", font_color="black")
            for i, e in allowed:
                net.add_node(
                    i,
                    label=e["topic"],
                    color=topic_colors.get(e["topic"], "#999999"),
                    title=f"{e['subtopic']} ({e['lang']})\n{e['text'][:150]}...",
                )
            for i, e in allowed:
                for j in rag.adj.get(i, set()):
                    if j in allowed_set:
                        net.add_edge(i, j)
            net.write_html("_network.html")
            with open("_network.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=540)
        else:
            st.info("Instala pyvis para activar la red interactiva: pip install pyvis")

    elif vista == "Red persistente":
        st.header("Red persistente del corpus")
        if os.path.exists("data_catalog/corpus_graph.html"):
            with open("data_catalog/corpus_graph.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=720)
            st.caption("Grafo completo pre-generado y persistente: data_catalog/corpus_graph.html y .gml")
        else:
            st.warning("Ejecuta pre_build.py para generar el grafo persistente.")

    elif vista == "Indice de documentos":
        st.header("Indice de documentos")
        if os.path.exists("data_catalog/document_index.jsonl"):
            docs = []
            with open("data_catalog/document_index.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    docs.append(json.loads(line))
            st.dataframe(pd.DataFrame(docs))
        else:
            st.warning("Catalogo no generado.")

    elif vista == "Diccionario de datos":
        st.header("Diccionario de datos")
        if os.path.exists("data_catalog/data_dictionary.json"):
            with open("data_catalog/data_dictionary.json", "r", encoding="utf-8") as f:
                st.json(json.load(f))
        else:
            st.warning("Catalogo no generado.")

    elif vista == "Glosario":
        st.header("Glosario de dominio")
        if os.path.exists("data_catalog/glossary.json"):
            with open("data_catalog/glossary.json", "r", encoding="utf-8") as f:
                st.dataframe(pd.DataFrame(json.load(f)))
        else:
            st.warning("Catalogo no generado.")

    elif vista == "Inventario de metadatos":
        st.header("Inventario de metadatos")
        if os.path.exists("data_catalog/metadata_inventory.json"):
            with open("data_catalog/metadata_inventory.json", "r", encoding="utf-8") as f:
                st.dataframe(pd.DataFrame(json.load(f)))
        else:
            st.warning("Catalogo no generado.")

    elif vista == "Tabla jerarquica":
        st.header("Tabla jerarquica: tema -> subtema -> idioma -> documentos")
        if os.path.exists("data_catalog/hierarchy.json"):
            with open("data_catalog/hierarchy.json", "r", encoding="utf-8") as f:
                st.json(json.load(f))
        else:
            st.warning("Catalogo no generado.")

    elif vista == "Clusters":
        st.header("Clusters de topicos (K-Means sobre TF-IDF)")
        if os.path.exists("data_catalog/topic_clusters.json"):
            with open("data_catalog/topic_clusters.json", "r", encoding="utf-8") as f:
                st.json(json.load(f))
        else:
            st.warning("Catalogo no generado.")

with tab_archi:
    st.header("Arquitectura de la solucion")
    def _mermaid_html(code):
        return f"""
        <html>
        <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true}});</script>
        </head>
        <body>
        <pre class="mermaid">
{code}
        </pre>
        </body>
        </html>
        """
    option = st.selectbox("Diagrama", list(mermaid_diagrams.DIAGRAMS.keys()))
    st.subheader(option.replace("_", " ").title())
    components.html(_mermaid_html(mermaid_diagrams.DIAGRAMS[option]), height=500)

with tab_pdf:
    st.subheader("Generar documentos tecnicos")
    if st.button("Crear PDFs (ES / EN)"):
        es_path, en_path = generate_pdfs()
        with open(es_path, "rb") as f:
            st.download_button(
                "Descargar PDF en espanol",
                f.read(),
                file_name=os.path.basename(es_path),
            )
        with open(en_path, "rb") as f:
            st.download_button(
                "Descargar PDF en ingles",
                f.read(),
                file_name=os.path.basename(en_path),
            )
