from fpdf import FPDF
import os

def _safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")

def _render(pdf, title, sections):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _safe(title), ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    for header, body in sections:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, _safe(header), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, _safe(body))
        pdf.ln(3)

ES_SECTIONS = [
    ("Introduccion", "Este documento describe la prueba tecnica de un agente IA agentico construido con LangGraph, orquestacion modular, proteccion PII, defensa contra prompt injection, gobernanza y observabilidad. La interfaz esta desarrollada en Streamlit."),
    ("LLM", "El modelo de lenguaje es configurable entre un simulado y OpenAI (gpt-3.5-turbo, gpt-4, gpt-4o). Se inyecta en el nodo de generacion del grafo."),
    ("Orquestador y LangGraph", "El orquestador modela el flujo como un grafo de estados con nodos: sanitizar, guardrails, recuperar y generar. LangGraph controla las transiciones, incluyendo ramas condicionales cuando falla la gobernanza."),
    ("RAG de grafos", "Los documentos se fragmentan con chunking solapado. Se calcula TF-IDF sobre los chunks, formando vectores dispersos. Se construye un grafo de similitud y conectividad, y se expanden los nodos recuperados por vecinos."),
    ("Chunking y TF-IDF", "El chunking divide documentos en bloques de tamano configurable con solapamiento. TF-IDF pondera la relevancia de terminos para recuperar los chunks mas cercanos a la consulta."),
    ("PII", "Se detectan y redactan correos, telefonos y SSN antes de enviar el prompt al LLM, minimizando fugas de informacion personal."),
    ("Prompt Injection", "Se bloquean consultas que contengan patrones tipicos de inyeccion como ignore previous instructions, jailbreak o DAN."),
    ("Gobierno IA", "Se validan politicas basicas: PII expuesto, inyeccion y terminos sensibles. Si hay violacion el agente detiene la generacion y devuelve un mensaje de bloqueo."),
    ("Observabilidad", "Streamlit muestra el estado final de cada nodo (trazas), la matriz TF-IDF, la red de chunks y las validaciones de seguridad."),
    ("Conclusion", "El prototipo demuestra un flujo agentico seguro, observable y gobernado sobre una base de recuperacion con grafos, listo para evolucionar a un entorno de produccion."),
]

EN_SECTIONS = [
    ("Introduction", "This document describes the technical test of an agentic AI agent built with LangGraph, modular orchestration, PII protection, prompt injection defense, governance and observability. The UI is built with Streamlit."),
    ("LLM", "The language model is configurable between a fake simulator and OpenAI (gpt-3.5-turbo, gpt-4, gpt-4o). It is invoked in the graph generation node."),
    ("Orchestrator and LangGraph", "The orchestrator models the workflow as a state graph with nodes: sanitize, guardrails, retrieve and generate. LangGraph controls the transitions, including conditional branches when governance fails."),
    ("Graph RAG", "Documents are split with overlapping chunking. TF-IDF is computed on chunks, forming sparse vectors. A similarity and adjacency graph is built, and retrieved nodes are expanded by neighbors."),
    ("Chunking and TF-IDF", "Chunking splits documents into configurable blocks with overlap. TF-IDF weights term relevance to retrieve the chunks closest to the query."),
    ("PII", "Emails, phones and SSNs are detected and redacted before the prompt is sent to the LLM, minimizing personal data leakage."),
    ("Prompt Injection", "Queries containing typical injection patterns such as ignore previous instructions, jailbreak or DAN are blocked."),
    ("AI Governance", "Basic policies are validated: exposed PII, injection and sensitive terms. If a violation occurs the agent stops generation and returns a blocking message."),
    ("Observability", "Streamlit shows the final state of every node (traces), the TF-IDF matrix, the chunk graph and the security validations."),
    ("Conclusion", "The prototype demonstrates a safe, observable and governed agentic workflow on top of graph-based retrieval, ready to evolve to production."),
]

def generate_pdfs(output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    es_path = os.path.join(output_dir, "documento_tecnico_es.pdf")
    en_path = os.path.join(output_dir, "technical_document_en.pdf")

    pdf_es = FPDF()
    pdf_es.set_auto_page_break(auto=True, margin=15)
    _render(pdf_es, "Arquitectura de Agente IA", ES_SECTIONS)
    pdf_es.output(es_path)

    pdf_en = FPDF()
    pdf_en.set_auto_page_break(auto=True, margin=15)
    _render(pdf_en, "AI Agent Architecture", EN_SECTIONS)
    pdf_en.output(en_path)

    return es_path, en_path
