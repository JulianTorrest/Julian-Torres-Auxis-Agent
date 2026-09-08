import json
import os
import random
from fpdf import FPDF
from pdf_generator import _safe
from chunker import chunk_text

TOPICS = {
    "economia": [
        "macroeconomia", "microeconomia", "inflacion", "tipos_de_interes",
        "politica_monetaria", "comercio_internacional", "desarrollo_economico",
        "economia_laboral", "finanzas_publicas", "mercados_emergentes"
    ],
    "legal": [
        "derecho_civil", "derecho_penal", "contratos", "propiedad_intelectual",
        "cumplimiento_normativo", "proteccion_de_datos", "litigios",
        "fusiones_y_adquisiciones", "derecho_laboral", "arbitraje"
    ],
    "estrategia": [
        "planificacion_estrategica", "ventaja_competitiva", "innovacion",
        "analisis_FODA", "cadena_de_valor", "crecimiento_organico",
        "diversificacion", "alianzas_estrategicas", "transformacion_digital",
        "sostenibilidad"
    ],
    "organizacional": [
        "cultura_organizacional", "liderazgo", "gestion_del_cambio",
        "gestion_del_talento", "engagement", "comunicacion_interna",
        "estructura_organizacional", "toma_de_decisiones", "diversidad_e_inclusion",
        "gestion_del_desempeno"
    ],
    "datos": [
        "gobernanza_de_datos", "calidad_de_datos", "ciencia_de_datos",
        "analitica_descriptiva", "machine_learning", "privacidad_de_datos",
        "arquitectura_de_datos", "visualizacion_de_datos", "big_data", "dataops"
    ],
}

GENERIC_ES = [
    "Es fundamental establecer metricas claras que permitan medir el progreso.",
    "Las organizaciones deben adaptar sus procesos a los cambios del entorno.",
    "La tecnologia juega un papel transformador en la evolucion de esta area.",
    "Los equipos multidisciplinarios aportan perspectivas valiosas.",
    "La sostenibilidad a largo plazo depende de una vision integral.",
    "Es necesario gestionar riesgos de manera proactiva.",
    "La formacion continua fortalece las capacidades internas.",
    "La transparencia genera confianza entre los actores involucrados.",
    "La innovacion impulsa la diferenciacion en el mercado.",
    "Los datos de calidad son la base para decisiones estrategicas.",
]

GENERIC_EN = [
    "It is essential to establish clear metrics to measure progress.",
    "Organizations must adapt their processes to environmental changes.",
    "Technology plays a transformative role in the evolution of this area.",
    "Multidisciplinary teams bring valuable perspectives.",
    "Long-term sustainability depends on an integrated vision.",
    "Risks must be managed proactively.",
    "Continuous training strengthens internal capabilities.",
    "Transparency builds trust among stakeholders.",
    "Innovation drives differentiation in the market.",
    "High-quality data is the foundation for strategic decisions.",
]

def _label(s):
    return s.replace("_", " ")

def make_content(topic, subtopic, lang):
    seed = sum(ord(c) for c in subtopic)
    sub = _label(subtopic)
    if lang == "es":
        extras = random.Random(seed).sample(GENERIC_ES, 4)
        return (
            f"{sub.capitalize()} en {topic}\n\n"
            f"El {sub} es una dimension clave dentro del area de {topic}. "
            f"En este documento se explora su definicion, importancia y aplicaciones practicas. "
            f"{' '.join(extras)} "
            f"Se presentan casos de uso que ilustran como las organizaciones pueden beneficiarse de una gestion adecuada de {sub}. "
            f"Se discuten los riesgos mas comunes y las mejores practicas para mitigarlos. "
            f"Finalmente, se ofrecen recomendaciones para integrar el {sub} en la estrategia global de la organizacion."
        )
    extras = random.Random(seed).sample(GENERIC_EN, 4)
    return (
        f"{sub.capitalize()} in {topic}\n\n"
        f"{sub.capitalize()} is a key dimension within the {topic} field. "
        f"This document explores its definition, importance, and practical applications. "
        f"{' '.join(extras)} "
        f"Use cases illustrate how organizations can benefit from proper management of {sub}. "
        f"Common risks and best practices for mitigating them are discussed. "
        f"Finally, recommendations are provided for integrating {sub} into the organization's overall strategy."
    )

def write_pdf(path, topic, subtopic, lang):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    title = f"{_label(subtopic).capitalize()} - {topic} ({lang.upper()})"
    pdf.cell(0, 10, _safe(title), ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    content = make_content(topic, subtopic, lang)
    for paragraph in content.split("\n\n"):
        pdf.multi_cell(0, 6, _safe(paragraph))
        pdf.ln(3)
    pdf.output(path)
    return content

def build(output_dir="corpus", index_path="corpus_index.jsonl", full_index_path="corpus_full.jsonl"):
    os.makedirs(output_dir, exist_ok=True)
    index = []
    full_index = []
    pdf_count = 0
    for topic, subtopics in TOPICS.items():
        topic_dir = os.path.join(output_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        for subtopic in subtopics:
            for lang in ["es", "en"]:
                filename = f"{subtopic}_{lang}.pdf"
                path = os.path.join(topic_dir, filename)
                text = write_pdf(path, topic, subtopic, lang)
                full_index.append({
                    "source": os.path.abspath(path),
                    "topic": topic,
                    "subtopic": subtopic,
                    "lang": lang,
                    "text": text,
                })
                chunks = chunk_text(text, size=300, overlap=30)
                for i, chunk in enumerate(chunks):
                    index.append({
                        "id": f"{topic}_{subtopic}_{lang}_{i}",
                        "source": os.path.abspath(path),
                        "topic": topic,
                        "subtopic": subtopic,
                        "lang": lang,
                        "chunk_index": i,
                        "text": chunk,
                    })
                pdf_count += 1
    with open(index_path, "w", encoding="utf-8") as f:
        for entry in index:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(full_index_path, "w", encoding="utf-8") as f:
        for entry in full_index:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return pdf_count, os.path.abspath(index_path), os.path.abspath(full_index_path), len(index)
