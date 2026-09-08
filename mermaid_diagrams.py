DIAGRAMS = {
    "flujo_general": """
graph TB
    subgraph Usuario["Capa de usuario"]
        U[Usuario]
        S[Streamlit UI]
    end
    U -->|consulta| S
    S --> IG[Input Guardian]
    IG --> SG[Semantic Cache]
    SG -->|miss| G[GraphRAG / Vector DB]
    G --> CM[Context Manager]
    CM --> M[Multi-LLM Router / MoE / Retry]
    M --> OG[Output Guardian]
    OG --> MEM[Memoria persistente]
    OG --> OBS[Observabilidad]
    OG --> S
    S --> U
""",
    "etapas_langgraph": """
graph LR
    A[Sanitize] --> B[Guardian]
    B --> C{policy_ok?}
    C -->|Sí| D[Retrieve]
    C -->|No| E[Generate bloqueo]
    D --> F[Generate]
    E --> G[Trace]
    F --> G
    G --> H[END]
""",
    "servicios": """
graph TB
    UI[Streamlit UI]
    ORQ[LangGraph Orchestrator]
    G[Guardian]
    RAG[GraphRAG + FAISS + SQLite + Chroma]
    LLM[Multi-LLM: Ollama / OpenAI / Gemini / Groq / Mistral / DeepSeek]
    CACHE[Semantic Cache]
    CTX[Context Manager]
    MEM[Memory Manager]
    AUD[Auditoria SQLite + LangSmith + Langfuse]
    UI --> ORQ
    ORQ --> G
    ORQ --> RAG
    ORQ --> LLM
    ORQ --> CACHE
    ORQ --> CTX
    ORQ --> MEM
    ORQ --> AUD
""",
    "datos": """
graph LR
    CORP[100 PDFs ES/EN]
    IDX[corpus_index.jsonl + corpus_full.jsonl]
    PRE[prebuilt: graph_rag.pkl]
    CAT[data_catalog: índice, glosario, clusters, grafo]
    DB[(SQLite embeddings)]
    FAISS[(FAISS vectores)]
    CHROMA[(Chroma documentos)]
    TRAZ[(SQLite trazas)]
    CACHE[(SQLite cache semántico)]
    MEM[(SQLite memoria)]
    CORP --> IDX
    IDX --> PRE
    IDX --> DB
    IDX --> FAISS
    IDX --> CHROMA
    PRE --> CAT
    ORQ --> TRAZ
    ORQ --> CACHE
    ORQ --> MEM
""",
    "ciberseguridad": """
graph LR
    subgraph Entrada["Guardian de entrada"]
        E1[PII Redaction]
        E2[Prompt Injection]
        E3[Encoding sospechoso]
        E4[SQL peligroso]
        E5[Tópicos prohibidos]
        E6[Rate limit]
        E7[Content moderation: toxicidad, hate, acoso, crítica, autolesión, sexual, spam, whitelist]
    end
    subgraph Salida["Guardian de salida"]
        S1[PII out]
        S2[Credenciales]
        S3[Moderación out]
        S4[Whitelist]
    end
    Q[Consulta] --> E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7
    E7 --> OK{¿OK?}
    OK -->|Sí| LLM
    OK -->|No| BLOCK[Bloquear + Auditar]
    LLM --> S1 --> S2 --> S3 --> S4
    S4 --> OUT[Respuesta]
""",
    "rag_y_vectores": """
graph LR
    PDF[PDFs] --> TEXT[Extracción de texto]
    TEXT --> CHUNKS[Chunking por secciones]
    CHUNKS --> IDX[corpus_index.jsonl]
    CHUNKS --> TF[TF-IDF + Grafo de chunks]
    TF --> G[GraphRAG .pkl]
    IDX --> FAISS[FAISS index]
    IDX --> SQLITE[SQLite embeddings]
    IDX --> CHROMA[Chroma docs]
    G --> Q[Query]
    FAISS --> Q
    SQLITE --> Q
    CHROMA --> Q
""",
    "observabilidad": """
graph LR
    Q[Consulta] --> T[TraceStore SQLite]
    T --> R[Registro: user, query, answer, model, tokens, latency, rejected, reason, tool_used, moderation]
    R --> AR[get_audit_report]
    R --> MS[get_moderation_stats]
    T --> LS[LangSmith]
    T --> LF[LangFuse]
""",
    "soluciones": """
graph TB
    subgraph Usuario
        U[Usuario de negocio]
        S[Streamlit UI]
    end
    subgraph Agente
        A[Agente LangGraph]
        P[Perfiles: Fast, Calidad, Redundante]
        M[Modos: Router, Parallel, MoE, Retry]
    end
    subgraph Proveedores
        LLM[OpenAI, Gemini, Groq, Mistral, DeepSeek, Ollama, Fake]
    end
    U -->|consulta| S
    S -->|selecciona perfil y modo| A
    A --> P
    P --> M
    M -->|fallback automatico| LLM
    LLM -->|respuesta| A
    A -->|respuesta + metadatos| S
    S --> U
""",
    "integraciones": """
graph LR
    subgraph Frontend
        ST[Streamlit]
    end
    subgraph Core
        LG[LangGraph / Orchestrator]
        GC[Guardian Kit]
        CM[Context Manager]
        SC[Semantic Cache]
        MM[Memory Manager]
    end
    subgraph RAG
        GR[GraphRAG]
        TF[TF-IDF]
        FAISS[FAISS]
        CH[ChromaDB]
        SQ[SQLite embeddings]
    end
    subgraph Proveedores
        OP[OpenAI]
        GM[Gemini]
        GQ[Groq]
        MI[Mistral]
        DS[DeepSeek]
        OL[Ollama]
    end
    subgraph Observabilidad
        TR[SQLite trazas]
        LS[LangSmith]
        LF[LangFuse]
    end
    ST -->|HTTP| LG
    LG -->|sanitize| GC
    LG -->|retrieve| GR
    LG -->|generate| OP
    LG -->|generate| GM
    LG -->|generate| GQ
    LG -->|generate| MI
    LG -->|generate| DS
    LG -->|generate| OL
    LG -->|cache| SC
    LG -->|context| CM
    LG -->|memory| MM
    LG -->|trace| TR
    TR -->|opcional| LS
    TR -->|opcional| LF
    GR --> TF
    GR --> FAISS
    GR --> CH
    GR --> SQ
""",
    "datos_detalle": """
graph LR
    subgraph Entrada
        PDF[50 PDFs ES/EN]
    end
    subgraph Preproceso
        TXT[Texto extraido]
        CHK[Chunking por secciones]
        TFI[TF-IDF]
        GRA[Grafo de similaridad]
    end
    subgraph Almacenamiento
        CI[corpus_index.jsonl]
        CF[corpus_full.jsonl]
        GP[prebuilt/graph_rag.pkl]
        DC[data_catalog: indice, glosario, clusters, grafo]
        VS[vector_db: SQLite, FAISS, Chroma]
        TR[observability: SQLite trazas, cache, memoria]
    end
    subgraph Consumo
        Q[Consulta del usuario]
        R[Respuesta]
    end
    PDF --> TXT --> CHK
    CHK --> CI
    CHK --> CF
    CHK --> TFI --> GRA --> GP
    CHK --> VS
    CI --> DC
    GP --> Q
    VS --> Q
    DC --> Q
    Q --> TR
    Q --> R
""",
}
