from __future__ import annotations

import base64
import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
import textstat
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="OCR + GPT | Laboratorio LLM",
    page_icon="🖼️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.3rem; padding-bottom: 3rem; }
    .token {
        display: inline-block;
        padding: 5px 8px;
        margin: 3px 2px;
        border-radius: 7px;
        border: 1px solid rgba(0,0,0,.12);
        font-size: 0.95rem;
    }
    .small { font-size: .85rem; opacity: .75; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🖼️ OCR + GPT")
st.caption(
    "Imagen → OCR → texto editable → ampliación con OpenAI → métricas lingüísticas y semánticas."
)


# ============================================================
# MODELOS Y CONSTANTES
# ============================================================
OPENAI_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
]

EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

COLORS = [
    "#E8F1FF",
    "#FFF1E6",
    "#EAF7EA",
    "#F8E8FF",
    "#FFFBE5",
    "#EAF8F6",
]


@dataclass
class GenerationResult:
    text: str
    response: Any
    temperature: float
    top_p: float
    model: str
    style: str


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_image(
    image: Image.Image,
    grayscale: bool,
    contrast: float,
    sharpness: float,
    upscale: int,
    threshold: bool,
) -> Image.Image:
    img = image.convert("RGB")

    if grayscale:
        img = ImageOps.grayscale(img)

    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

    if threshold:
        if img.mode != "L":
            img = ImageOps.grayscale(img)
        img = img.point(lambda px: 255 if px > 165 else 0)

    if upscale > 1:
        img = img.resize(
            (img.width * upscale, img.height * upscale),
            Image.Resampling.LANCZOS,
        )

    img = img.filter(ImageFilter.SHARPEN)
    return img


def run_ocr(image: Image.Image, language: str, psm: int) -> tuple[str, pd.DataFrame]:
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(image, lang=language, config=config)

    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DATAFRAME,
    )

    if not data.empty:
        data = data.dropna(subset=["text"]).copy()
        data["text"] = data["text"].astype(str)
        data = data[data["text"].str.strip() != ""]
        if "conf" in data.columns:
            data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(0)

    return clean_text(text), data


def word_list(text: str) -> list[str]:
    return re.findall(
        r"\b[\wÁÉÍÓÚÜÑáéíóúüñ'-]+\b",
        text,
        flags=re.UNICODE,
    )


def sentence_list(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?¿¡])\s+", text)
        if s.strip()
    ]


def measures(text: str) -> dict[str, float]:
    words = word_list(text)
    unique = {w.lower() for w in words}
    sentences = sentence_list(text)

    return {
        "Caracteres": len(text),
        "Palabras": len(words),
        "Palabras únicas": len(unique),
        "Oraciones": len(sentences),
        "Promedio palabras/oración": len(words) / max(1, len(sentences)),
        "Promedio caracteres/palabra": (
            sum(len(w) for w in words) / max(1, len(words))
        ),
        "Diversidad léxica": len(unique) / max(1, len(words)),
        "Densidad alfanumérica": len(re.findall(r"\w", text, flags=re.UNICODE))
        / max(1, len(text)),
    }


def grammar_heuristic(text: str) -> float:
    """Indicador didáctico: busca patrones frecuentes de mala segmentación."""
    if not text.strip():
        return 0.0

    penalty = 0.0
    patterns = [
        r"\s+[,.!?;:]",             # espacio antes de puntuación
        r"[,.!?;:]{2,}",             # puntuación duplicada
        r"\b(\w+)\s+\1\b",         # palabra duplicada
        r"\s{2,}",                  # espacios excesivos
        r"(?i)\b(de de|la la|el el|que que|y y|en en)\b",
    ]
    for pattern in patterns:
        penalty += len(re.findall(pattern, text))

    words = word_list(text)
    score = 100 * math.exp(-penalty / max(8.0, len(words) * 0.12))
    return round(max(0.0, min(100.0, score)), 1)


def syntax_heuristic(text: str) -> float:
    """Indicador estructural aproximado; no reemplaza un parser lingüístico."""
    sentences = sentence_list(text)
    words = word_list(text)
    if not sentences or not words:
        return 0.0

    avg_len = len(words) / len(sentences)
    length_score = max(0.0, 100 - abs(avg_len - 20) * 3)
    connectors = len(
        re.findall(
            r"\b(y|pero|porque|aunque|mientras|que|si|cuando|donde|además|sin embargo)\b",
            text.lower(),
        )
    )
    connector_score = min(100.0, 55.0 + connectors * 4.0)
    return round(length_score * 0.6 + connector_score * 0.4, 1)


def semantic_similarity(text_a: str, text_b: str, model_name: str) -> float:
    model = load_embedding_model(model_name)
    vectors = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    score = float(np.dot(vectors[0], vectors[1]))
    return round(max(-1.0, min(1.0, score)), 4)


def coherence_score(text: str, model_name: str) -> float:
    sentences = sentence_list(text)
    if len(sentences) < 2:
        return 100.0

    model = load_embedding_model(model_name)
    vectors = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    adjacent_scores = [
        float(np.dot(vectors[i], vectors[i + 1]))
        for i in range(len(vectors) - 1)
    ]
    mean_score = float(np.mean(adjacent_scores))
    return round(max(0.0, min(100.0, (mean_score + 1.0) * 50.0)), 1)


def colored_partition(text: str, mode: str) -> list[tuple[str, str]]:
    if mode == "Palabras":
        parts = re.findall(
            r"[\wÁÉÍÓÚÜÑáéíóúüñ'-]+",
            text,
            flags=re.UNICODE,
        )
    elif mode == "Caracteres":
        parts = list(text)
    elif mode == "Líneas":
        parts = [x for x in text.splitlines() if x.strip()]
    else:
        parts = sentence_list(text)

    return [(part, COLORS[i % len(COLORS)]) for i, part in enumerate(parts)]


def render_partitions(text: str, mode: str, show_index: bool = True) -> None:
    chunks = colored_partition(text, mode)
    html: list[str] = []
    for idx, (part, bg) in enumerate(chunks, start=1):
        safe = (
            part.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        label = f"{idx}: {safe}" if show_index else safe
        html.append(
            f'<span class="token" style="background:{bg}">{label}</span>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def openai_generate(
    api_key: str,
    model: str,
    source_text: str,
    style: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    reasoning_effort: str,
    verbosity: str,
    extra_instruction: str,
) -> GenerationResult:
    client = OpenAI(api_key=api_key)

    styles = {
        "Formal": "Usa un registro formal, profesional y claro.",
        "Técnica": "Usa terminología técnica, precisa y estructurada.",
        "Académica": "Usa un registro académico, lógico y conceptual.",
    }

    prompt = f"""
Eres un editor experto en español. Recibes texto obtenido mediante OCR.

TEXTO OCR:
---
{source_text}
---

Objetivo:
1. Corrige errores evidentes de OCR sin inventar información.
2. Amplía el contenido manteniendo el significado y los hechos originales.
3. Mejora cohesión, claridad, sintaxis, gramática y estructura.
4. {styles[style]}
5. No inventes nombres, cifras, fechas, referencias o fuentes.
6. Devuelve únicamente el texto final, sin explicar tus cambios.

Instrucción adicional:
{extra_instruction or 'Ninguna.'}
""".strip()

    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "text": {"verbosity": verbosity},
    }

    # Para las familias de razonamiento GPT-5.x, temperature/top_p solo son
    # compatibles con reasoning.effort="none" según la guía de modelos.
    is_reasoning_family = model.startswith(("gpt-5", "gpt-6", "o3", "o4"))
    if is_reasoning_family:
        payload["reasoning"] = {"effort": reasoning_effort}
        if reasoning_effort == "none":
            payload["temperature"] = float(temperature)
            payload["top_p"] = float(top_p)
    else:
        payload["temperature"] = float(temperature)
        payload["top_p"] = float(top_p)

    response = client.responses.create(**payload)
    return GenerationResult(
        text=response.output_text.strip(),
        response=response,
        temperature=temperature,
        top_p=top_p,
        model=model,
        style=style,
    )


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    result: dict[str, int] = {}
    for source_name, target_name in [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ]:
        value = getattr(usage, source_name, None)
        if value is not None:
            try:
                result[target_name] = int(value)
            except (TypeError, ValueError):
                pass
    return result


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    secret_key = get_secret("OPENAI_API_KEY")
    typed_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Puedes introducirla durante la sesión o usar OPENAI_API_KEY en Streamlit Secrets.",
    )
    api_key = typed_key.strip() or secret_key

    if secret_key and not typed_key:
        st.success("Clave disponible desde Streamlit Secrets.")
    elif typed_key:
        st.success("Clave cargada en esta sesión.")
    else:
        st.warning("Falta la API key de OpenAI.")

    st.divider()
    st.subheader("LLM")

    model = st.selectbox("Modelo GPT", OPENAI_MODELS, index=0)

    reasoning_options = ["none", "low", "medium", "high", "xhigh"]
    reasoning_effort = st.select_slider(
        "Reasoning effort",
        options=reasoning_options,
        value="none",
        help="En modelos GPT-5.x, deja 'none' para usar temperature y top-p.",
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.4,
        0.05,
    )

    top_p = st.slider(
        "Top-p",
        0.1,
        1.0,
        1.0,
        0.05,
    )

    verbosity = st.select_slider(
        "Verbosity",
        options=["low", "medium", "high"],
        value="medium",
    )

    max_output_tokens = st.number_input(
        "Máximo de tokens de salida",
        min_value=64,
        max_value=16000,
        value=1200,
        step=64,
    )

    style = st.radio(
        "Tipo de respuesta",
        ["Formal", "Técnica", "Académica"],
        index=0,
    )

    extra_instruction = st.text_area(
        "Instrucción adicional",
        placeholder="Ej.: desarrolla los conceptos y conserva las cifras del documento.",
        height=90,
    )

    st.divider()
    st.subheader("OCR")

    ocr_language = st.selectbox(
        "Idioma OCR",
        ["spa", "eng", "spa+eng"],
        index=0,
    )
    psm = st.selectbox(
        "Modo PSM",
        [3, 4, 6, 11, 12],
        index=2,
        help="PSM 6 suele funcionar bien con bloques de texto. Prueba otros modos para diseños diferentes.",
    )
    grayscale = st.checkbox("Escala de grises", value=True)
    contrast = st.slider("Contraste", 0.8, 2.5, 1.2, 0.1)
    sharpness = st.slider("Nitidez", 0.8, 3.0, 1.0, 0.1)
    upscale = st.select_slider("Escala OCR", [1, 2, 3], value=2)
    threshold = st.checkbox("Umbral blanco/negro", value=False)

    st.divider()
    st.subheader("Métricas semánticas")
    embedding_model = st.selectbox(
        "Modelo de embeddings",
        EMBEDDING_MODELS,
        index=0,
    )


# ============================================================
# CARGA DE IMAGEN
# ============================================================
uploaded = st.file_uploader(
    "📤 Sube una imagen",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
    help="La imagen se mantiene en memoria de la sesión de Streamlit.",
)

if uploaded is None:
    st.info("Carga una imagen para comenzar el ejercicio.")
    st.stop()

try:
    original = Image.open(io.BytesIO(uploaded.getvalue()))
except Exception as exc:
    st.error(f"No se pudo abrir la imagen: {exc}")
    st.stop()

processed = preprocess_image(
    original,
    grayscale=grayscale,
    contrast=contrast,
    sharpness=sharpness,
    upscale=upscale,
    threshold=threshold,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagen original")
    st.image(original, use_container_width=True)
    st.caption(f"{uploaded.name} · {original.width} × {original.height}px")

with col2:
    st.subheader("Imagen preparada para OCR")
    st.image(processed, use_container_width=True)
    st.caption(f"Escala ×{upscale} · idioma {ocr_language} · PSM {psm}")

if st.button("🔎 Ejecutar OCR", type="primary", use_container_width=True):
    try:
        with st.spinner("Extrayendo texto con Tesseract OCR..."):
            ocr_text, ocr_data = run_ocr(processed, ocr_language, psm)
        st.session_state["ocr_text"] = ocr_text
        st.session_state["ocr_data"] = ocr_data
        st.session_state.pop("generation", None)
        st.success("OCR completado.")
    except Exception as exc:
        st.error(
            "No se pudo ejecutar Tesseract. Verifica la instalación de Tesseract y "
            f"el paquete de idioma seleccionado. Detalle: {exc}"
        )

ocr_text = st.session_state.get("ocr_text", "")
if not ocr_text:
    st.stop()

# ============================================================
# ÁREA PRINCIPAL
# ============================================================
tab_ocr, tab_part, tab_quality, tab_gpt, tab_metrics = st.tabs(
    [
        "📝 Texto OCR",
        "🎨 Particiones",
        "📊 Calidad OCR",
        "🤖 Ampliar con GPT",
        "📈 Métricas",
    ]
)

with tab_ocr:
    st.subheader("Texto extraído y editable")
    edited = st.text_area("Texto OCR", value=ocr_text, height=320)

    if st.button("💾 Guardar edición OCR", use_container_width=True):
        st.session_state["ocr_text"] = clean_text(edited)
        st.session_state.pop("generation", None)
        st.success("Texto OCR actualizado.")
        ocr_text = st.session_state["ocr_text"]

    m = measures(edited)
    cols = st.columns(5)
    cols[0].metric("Palabras", int(m["Palabras"]))
    cols[1].metric("Oraciones", int(m["Oraciones"]))
    cols[2].metric("Caracteres", int(m["Caracteres"]))
    cols[3].metric("Palabras únicas", int(m["Palabras únicas"]))
    cols[4].metric("Diversidad", f"{m['Diversidad léxica']:.2f}")

with tab_part:
    st.subheader("Texto dividido y coloreado")
    st.caption(
        "Estas son particiones lingüísticas/visuales para el laboratorio. "
        "No representan necesariamente los tokens internos del modelo GPT."
    )

    partition_mode = st.radio(
        "Método de partición",
        ["Palabras", "Oraciones", "Líneas", "Caracteres"],
        horizontal=True,
    )
    render_partitions(ocr_text, partition_mode)

    st.divider()
    parts = [p for p, _ in colored_partition(ocr_text, partition_mode)]
    part_df = pd.DataFrame(
        {
            "posición": range(1, len(parts) + 1),
            "partición": parts,
            "caracteres": [len(p) for p in parts],
        }
    )
    st.dataframe(part_df.head(1000), use_container_width=True, hide_index=True)

with tab_quality:
    st.subheader("Confianza del OCR")
    data = st.session_state.get("ocr_data")

    if isinstance(data, pd.DataFrame) and not data.empty:
        confidence = pd.to_numeric(data.get("conf"), errors="coerce").dropna()
        avg = float(confidence.mean()) if not confidence.empty else 0.0
        med = float(confidence.median()) if not confidence.empty else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Confianza media", f"{avg:.1f}%")
        c2.metric("Confianza mediana", f"{med:.1f}%")
        c3.metric("Elementos detectados", len(data))

        cols = [
            x
            for x in [
                "text", "conf", "left", "top", "width", "height",
                "block_num", "par_num", "line_num",
            ]
            if x in data.columns
        ]
        st.dataframe(data[cols].head(1000), use_container_width=True, hide_index=True)
    else:
        st.info("Tesseract no devolvió datos detallados de confianza.")

with tab_gpt:
    st.subheader("Ampliación del texto con OpenAI")

    if not api_key:
        st.warning("Ingresa la API key de OpenAI en la barra lateral.")

    if model.startswith("gpt-5") and reasoning_effort != "none":
        st.info(
            "Con reasoning distinto de 'none', esta app no envía temperature/top-p "
            "porque esos parámetros son incompatibles en los modelos GPT-5.x según la guía actual de modelos."
        )

    st.write(
        f"Modelo: **{model}** · Estilo: **{style}** · "
        f"Temperature: **{temperature:.2f}** · Top-p: **{top_p:.2f}** · "
        f"Reasoning: **{reasoning_effort}**"
    )

    if st.button(
        "✨ Ampliar texto con GPT",
        type="primary",
        use_container_width=True,
        disabled=not bool(api_key),
    ):
        try:
            with st.spinner("Generando texto ampliado..."):
                result = openai_generate(
                    api_key=api_key,
                    model=model,
                    source_text=ocr_text,
                    style=style,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=int(max_output_tokens),
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    extra_instruction=extra_instruction,
                )
            st.session_state["generation"] = result
            st.success("Respuesta generada.")
        except Exception as exc:
            st.error(f"Error al llamar a OpenAI: {exc}")

    generation = st.session_state.get("generation")
    if isinstance(generation, GenerationResult):
        st.markdown("### Resultado")
        st.markdown(generation.text)

        usage = response_usage(generation.response)
        if usage:
            st.markdown("### Uso de tokens")
            u1, u2, u3 = st.columns(3)
            u1.metric("Input tokens", usage.get("input_tokens", "—"))
            u2.metric("Output tokens", usage.get("output_tokens", "—"))
            u3.metric("Total tokens", usage.get("total_tokens", "—"))

        st.download_button(
            "⬇️ Descargar texto generado",
            data=generation.text,
            file_name="texto_generado.txt",
            mime="text/plain",
            use_container_width=True,
        )

with tab_metrics:
    generation = st.session_state.get("generation")
    if not isinstance(generation, GenerationResult):
        st.info("Genera primero el texto con GPT para calcular las métricas de salida.")
        st.stop()

    generated = generation.text
    source_measures = measures(ocr_text)
    gen_measures = measures(generated)

    st.subheader("Medidas del texto")
    comparison = pd.DataFrame(
        {"OCR": source_measures, "GPT": gen_measures}
    )
    st.dataframe(comparison.round(3), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Palabras generadas", int(gen_measures["Palabras"]))
    m2.metric(
        "Ratio de expansión",
        f"{gen_measures['Palabras'] / max(1, source_measures['Palabras']):.2f}×",
    )
    try:
        readability = float(textstat.flesch_reading_ease(generated))
    except Exception:
        readability = 0.0
    m3.metric("Flesch Reading Ease", f"{readability:.1f}")
    m4.metric("Diversidad léxica", f"{gen_measures['Diversidad léxica']:.2f}")

    st.subheader("Calidad lingüística / semántica")
    with st.spinner("Calculando métricas semánticas..."):
        semantic = semantic_similarity(ocr_text, generated, embedding_model)
        coherence = coherence_score(generated, embedding_model)

    syntax = syntax_heuristic(generated)
    grammar = grammar_heuristic(generated)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Coherencia", f"{coherence:.1f}/100")
    q2.metric("Semántica", f"{semantic:.3f}")
    q3.metric("Sintaxis*", f"{syntax:.1f}/100")
    q4.metric("Gramática*", f"{grammar:.1f}/100")

    st.caption(
        "La coherencia usa similitud entre oraciones consecutivas en embeddings; "
        "la semántica compara OCR y salida. Sintaxis y gramática son heurísticas didácticas, "
        "no un diagnóstico lingüístico profesional."
    )

    st.subheader("Comparación semántica")
    semantic_percent = max(0.0, min(1.0, (semantic + 1.0) / 2.0))
    st.progress(semantic_percent)
    st.write(f"Similitud coseno OCR ↔ GPT: **{semantic:.4f}**")

    st.subheader("Frecuencia de palabras")
    source_freq = Counter(w.lower() for w in word_list(ocr_text))
    gen_freq = Counter(w.lower() for w in word_list(generated))

    freq_df = pd.DataFrame(
        {
            "palabra": [w for w, _ in gen_freq.most_common(20)],
            "frecuencia GPT": [c for _, c in gen_freq.most_common(20)],
        }
    )
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Nota: la API key no debe guardarse en el repositorio. Para despliegues, usa Streamlit Secrets."
)
