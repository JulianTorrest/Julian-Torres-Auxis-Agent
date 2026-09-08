import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import time
import json
import pandas as pd

# Directorio de datos escribible (Streamlit Cloud requiere /tmp)
os.environ.setdefault("AUXIS_DATA_DIR", os.path.join(tempfile.gettempdir(), "auxis_agent"))

from llm_clients import get_llm
from orchestrator import create_workflow
from retriever import GraphRAG
from pii_guard import redact_pii
from prompt_guard import is_injection
from governance import check_policy
from pdf_generator import generate_pdfs
from observability_manager import ObservabilityManager
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
    "groq": "llama-3.1-8b-instant",
    "mistral": "mistral-tiny-latest",
    "ollama": "llama3.2",
    "deepseek": "deepseek-chat",
}

if "documents" not in st.session_state:
    st.session_state.documents = DEFAULT_DOCS

with st.sidebar:
    st.header("Perfil del agente")
    profiles = {
        "Fast (Ollama router)": {"selected_providers": ["ollama"], "mode": "router", "max_retries": 0},
        "Calidad (MoE OpenAI+DeepSeek)": {"selected_providers": ["openai", "deepseek"], "mode": "moe", "max_retries": 2},
        "Redundante (Parallel)": {"selected_providers": ["openai", "gemini", "groq"], "mode": "parallel", "max_retries": 0},
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
    st.session_state.setdefault("selected_providers", [])
    st.session_state.setdefault("mode", "router")
    st.session_state.setdefault("max_retries", 1)
    all_providers = ["openai", "gemini", "groq", "mistral", "ollama", "deepseek"]
    selected_providers = st.multiselect("Proveedores activos", all_providers, key="selected_providers")

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
    st.session_state["selected_provider"] = router_provider

    st.divider()

    st.divider()
    st.header("Gobernanza y Observabilidad")
    user = st.text_input("Usuario", value="anon")
    st.caption("Configura LANGSMITH_API_KEY y/o LANGFUSE_* como variables de entorno para trazas externas.")



with st.spinner("Cargando RAG pre-generado desde el corpus..."):
    rag = pre_build.load_rag()

tracer = ObservabilityManager()

st.title("Prueba Tecnica - Agente IA con LangGraph + Router/MoE")

tab_query, tab_rag, tab_gov, tab_trace, tab_vector, tab_temas, tab_archi, tab_pdf = st.tabs([
    "Consulta", "RAG y Grafo", "Gobernanza", "Observabilidad", "Base Vectorial", "Temas y Red", "Arquitectura", "PDFs"
])

with tab_query:
    st.subheader("Asistente de consulta")
    st.info(
        "Este asistente responde preguntas de negocio sobre economia, legal, estrategia, "
        "organizacional y datos, basandose en el corpus pre-generado. "
        "Selecciona proveedores en el sidebar y asegurate de tener las API keys en "
        "Settings > Secrets (no se muestran en la UI)."
    )
    with st.expander("Ejemplos de preguntas y modos de generacion"):
        st.markdown("""
        **Ejemplos de preguntas que puede responder (usuario de negocio):**

        *Economia:*
        - "¿Que es la inflacion y como afecta el poder adquisitivo?"
        - "Explica la relacion entre tasa de interes e inversion."
        - "Diferencia entre recesion y depresion economica."

        *Legal:*
        - "¿Cuales son los elementos esenciales de un contrato?"
        - "¿Que es una clausula de confidencialidad y para que sirve?"
        - "Ejemplos de responsabilidad civil y penal en el entorno empresarial."

        *Estrategia:*
        - "¿Que es un analisis FODA y como se construye?"
        - "Como se define una ventaja competitiva sostenible."
        - "Estrategias de crecimiento corporativo: penetracion, desarrollo de productos, mercados."

        *Organizacional:*
        - "¿Que es la cultura organizacional y por que importa?"
        - "Tipos de estructuras organizacionales y sus ventajas."
        - "Como gestionar el cambio dentro de una empresa."

        *Datos:*
        - "¿Que es un diccionario de datos y para que se usa?"
        - "Diferencia entre datos, metadatos e informacion."
        - "Para que sirven las bases de datos vectoriales en un sistema de RAG."

        **Perfiles del agente:**
        - *Fast (Ollama router)*: un proveedor, respuesta rapida.
        - *Calidad (MoE OpenAI+DeepSeek)*: varios modelos y elige la mejor respuesta.
        - *Redundante (Parallel)*: todos en paralelo, devuelve el primero.

        **Modos de generacion:**
        - **Router**: un solo proveedor.
        - **Parallel**: varios proveedores concurrentes, gana el primero.
        - **MoE**: varios proveedores, seleccion de la mejor respuesta.
        - **Retry**: reintenta hasta mejorar la calidad.
        """)

    # Configurar LLM y workflow segun la seleccion actual del sidebar
    # Agrega automaticamente proveedores con API key para fallback robusto
    selected = st.session_state.get("selected_providers", [])
    all_providers = ["openai", "gemini", "groq", "mistral", "ollama", "deepseek"]
    selected = [p for p in selected if p in all_providers]
    fallback_providers = [p for p in all_providers if p not in selected and p != "ollama" and os.getenv(f"{p.upper()}_API_KEY")]
    active_providers = [p for p in all_providers if p in selected or p in fallback_providers]
    if not active_providers:
        active_providers = ["fake"]
    configs = {}
    for p in active_providers:
        base = {"model": DEFAULT_MODELS.get(p, "")}
        if p != "ollama":
            base["api_key"] = os.getenv(f"{p.upper()}_API_KEY", "")
        configs[p] = base
    selected_provider = st.session_state.get("selected_provider") or (active_providers[0] if active_providers else "fake")
    if selected_provider not in active_providers:
        selected_provider = active_providers[0]
    llm_key = (tuple(sorted(active_providers)), st.session_state.get("max_retries", 1), selected_provider, user)
    if st.session_state.get("llm_key") != llm_key:
        st.session_state.llm = get_llm(active_providers, configs, st.session_state.get("max_retries", 1))
        st.session_state.llm_key = llm_key
    llm = st.session_state.llm
    workflow_key = (llm_key, user)
    if st.session_state.get("workflow_key") != workflow_key:
        st.session_state.workflow = create_workflow(rag, llm, tracer, session_id=user, token_budget=2000)
        st.session_state.workflow_key = workflow_key
    workflow = st.session_state.workflow

    missing_keys = [p for p in selected if p not in ("fake", "ollama") and not os.getenv(f"{p.upper()}_API_KEY")]
    if missing_keys:
        st.warning(
            f"Proveedores sin API key: {', '.join(missing_keys)}. "
            "Configura las variables en Streamlit Cloud Settings > Secrets para obtener respuestas reales."
        )

    query = st.text_area("Escribe tu consulta", height=80)
    if st.button("Ejecutar agente"):
        final = workflow.invoke({
            "query": query,
            "user": user,
            "start_time": time.time(),
            "mode": st.session_state.get("mode", "router"),
            "selected_provider": selected_provider,
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
        from guardian_kit import InputGuardian
        g = InputGuardian().check(user, test)
        st.json(g)
    st.subheader("Output Guardian (PII, credenciales, moderacion)")
    out = st.text_input("Texto de salida a validar", value="", key="guard_out")
    if out:
        from guardian_kit import output_guard
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
