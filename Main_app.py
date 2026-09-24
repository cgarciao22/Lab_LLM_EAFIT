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
    }
    .token-chip {
        display: inline-block;
        padding: 3px 8px;
        margin: 3px 3px 3px 0;
        border-radius: 7px;
        border: 1px solid rgba(0, 0, 0, .10);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: .88rem;
        white-space: pre-wrap;
        vertical-align: middle;
    }
    .token-label {
        font-size: .72rem;
        opacity: .72;
        margin-right: 3px;
    }
    .response-card {
        border: 1px solid rgba(128, 128, 128, .25);
        border-radius: 12px;
        padding: 14px;
        min-height: 220px;
        background: rgba(127, 127, 127, .04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 Groq LLM Lab")
st.caption(
    "Explora modelos, tokenización, particiones coloreadas, Bag of Words, "
    "similitud, embeddings y el efecto de la temperatura sobre la generación."
)


# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "minimaxai/minimax-m2.7",
]

NON_CHAT_MODEL_MARKERS = (
    "whisper",
    "orpheus",
    "prompt-guard",
    "safeguard",
)

PALETTE = [
    "#FFF1B8", "#C7F9CC", "#BDE0FE", "#E9D5FF", "#FFD6A5",
    "#FFC8DD", "#D9F99D", "#BAE6FD", "#DDD6FE", "#FDE68A",
    "#FBCFE8", "#A7F3D0", "#BFDBFE", "#E2E8F0", "#FED7AA",
]


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(api_key: str) -> List[Dict[str, Any]]:
    client = Groq(api_key=api_key)
    response = client.models.list()
    models: List[Dict[str, Any]] = []

    for item in getattr(response, "data", []):
        record = {
            "id": getattr(item, "id", None),
            "owned_by": getattr(item, "owned_by", None),
            "active": getattr(item, "active", None),
            "context_window": getattr(item, "context_window", None),
            "max_completion_tokens": getattr(item, "max_completion_tokens", None),
            "created": getattr(item, "created", None),
        }
        if record["id"]:
            models.append(record)

    return sorted(models, key=lambda x: x["id"])


def get_model_metadata(models: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    for model in models:
        if model.get("id") == model_id:
            return model
    return {"id": model_id}


def is_text_generation_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(marker in lowered for marker in NON_CHAT_MODEL_MARKERS)


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def choose_encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def token_analysis(text: str, encoding_name: str) -> Tuple[pd.DataFrame, int]:
    enc = choose_encoding(encoding_name)
    token_ids = enc.encode(text)
    rows = []
    for idx, token_id in enumerate(token_ids):
        try:
            token_text = enc.decode_single_token_bytes(token_id).decode(
                "utf-8", errors="replace"
            )
        except Exception:
            token_text = enc.decode([token_id])
        rows.append(
            {
                "posición": idx,
                "token_id": token_id,
                "token": token_text,
            }
        )
    return pd.DataFrame(rows), len(token_ids)


def regex_tokens(text: str) -> List[str]:
    # Words, numbers, spaces and punctuation as visible partitions.
    return re.findall(r"\w+|\s+|[^\w\s]", text, flags=re.UNICODE)


def whitespace_tokens(text: str) -> List[str]:
    # Keep spaces visible so the partition matches the original text.
    return re.findall(r"\S+|\s+", text, flags=re.UNICODE)


def character_tokens(text: str) -> List[str]:
    return list(text)


def method_tokens(text: str, method: str, encoding_name: str) -> Tuple[List[str], List[Optional[int]]]:
    if method == "BPE / tiktoken":
        df, _ = token_analysis(text, encoding_name)
        return df["token"].tolist(), [int(x) for x in df["token_id"].tolist()]
    if method == "Palabras + espacios":
        return whitespace_tokens(text), [None] * len(whitespace_tokens(text))
    if method == "Palabras + puntuación":
        toks = regex_tokens(text)
        return toks, [None] * len(toks)
    if method == "Caracteres":
        toks = character_tokens(text)
        return toks, [None] * len(toks)
    raise ValueError(f"Método desconocido: {method}")


def colored_partition_html(tokens: List[str], ids: List[Optional[int]], show_ids: bool) -> str:
    chips = []
    color_index = 0
    for token, token_id in zip(tokens, ids):
        if token == "":
            continue
        bg = PALETTE[color_index % len(PALETTE)]
        safe_token = html.escape(token).replace("\n", "↵\n").replace("\t", "→\t")
        prefix = f'<span class="token-label">{token_id}</span>' if show_ids and token_id is not None else ""
        chips.append(
            f'<span class="token-chip" style="background:{bg}">{prefix}{safe_token}</span>'
        )
        color_index += 1
    return '<div class="token-box">' + "".join(chips) + "</div>"


def bow_analysis(texts: List[str], use_tfidf: bool = False):
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    vectorizer = Vectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    table = pd.DataFrame(
        matrix.toarray(),
        index=[f"Texto {i + 1}" for i in range(len(texts))],
        columns=terms,
    )
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = (
        pd.DataFrame({"término": terms, "frecuencia": frequencies})
        .sort_values(["frecuencia", "término"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return table, vocab, matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def completion_usage(completion: Any) -> Dict[str, Optional[int]]:
    usage = getattr(completion, "usage", None)
    return {
        "prompt_tokens": safe_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": safe_int(getattr(usage, "completion_tokens", None)),
        "total_tokens": safe_int(getattr(usage, "total_tokens", None)),
    }


def generate_one(
    client: Groq,
    model_id: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_completion_tokens": int(max_tokens),
    }
    if seed is not None:
        request["seed"] = int(seed)

    completion = client.chat.completions.create(**request)
    return {
        "temperature": temperature,
        "text": completion.choices[0].message.content or "",
        "usage": completion_usage(completion),
    }


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = st.text_input(
        "GROQ API Key",
        value="",
        type="password",
        placeholder="gsk_...",
        key="groq_api_key",
        help="Se usa en memoria para la sesión de Streamlit.",
    ).strip()

    if not api_key:
        try:
            configured = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            configured = ""
        if configured:
            st.success("API key cargada desde Streamlit Secrets.")
        else:
            st.info("Ingresa tu API key para usar la generación y consultar modelos.")

    model_source = st.radio(
        "Catálogo de modelos",
        ["Groq (dinámico)", "Lista de respaldo"],
        index=0,
    )

    models: List[Dict[str, Any]] = []
    if model_source == "Groq (dinámico)" and api_key:
        try:
            models = fetch_models(api_key)
            st.success(f"{len(models)} modelos disponibles.")
        except Exception as exc:
            st.warning(f"No se pudo consultar el catálogo: {exc}")

    if not models:
        models = [{"id": m} for m in FALLBACK_MODELS]

    generation_models = [m for m in models if is_text_generation_model(m["id"])]
    if not generation_models:
        generation_models = models

    model_ids = [m["id"] for m in generation_models]
    default_model = "openai/gpt-oss-120b" if "openai/gpt-oss-120b" in model_ids else model_ids[0]
    model_id = st.selectbox(
        "Modelo para generación",
        model_ids,
        index=model_ids.index(default_model),
        help="Se excluyen del selector de generación modelos de audio/moderación para evitar llamadas incompatibles.",
    )

    st.divider()
    st.subheader("Parámetros de generación")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.70, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)

    selected_metadata = get_model_metadata(models, model_id)
    model_max_completion = safe_int(selected_metadata.get("max_completion_tokens"))
    max_token_limit = min(model_max_completion or 65536, 65536)
    default_max_tokens = min(1024, max_token_limit)
    max_tokens = st.number_input(
        "Max completion tokens",
        min_value=64,
        max_value=max_token_limit,
        value=default_max_tokens,
        step=64,
        help=f"Límite ajustado al modelo seleccionado (UI hasta {max_token_limit:,}).",
    )

    use_seed = st.checkbox(
        "Fijar seed para comparar de forma más controlada",
        value=True,
        help="La reproducibilidad con seed es de mejor esfuerzo; cambiar la temperatura modifica el muestreo.",
    )
    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=1234, step=1, disabled=not use_seed)

    system_prompt = st.text_area(
        "System prompt",
        value="Eres un asistente útil, preciso y claro.",
        height=90,
    )

    st.divider()
    st.subheader("Tokenización local")
    encoding_name = st.selectbox(
        "Encoding BPE",
        ["cl100k_base", "o200k_base"],
        index=0,
        help="Permite inspeccionar una tokenización BPE local. Puede diferir del tokenizador del modelo de Groq.",
    )

    st.caption("Para despliegue, usa Streamlit Secrets en lugar de guardar la API key en el código.")


# -----------------------------------------------------------------------------
# Header metrics
# -----------------------------------------------------------------------------
metadata = get_model_metadata(models, model_id)
info_cols = st.columns(4)
info_cols[0].metric("Modelo", model_id)
info_cols[1].metric("Proveedor", metadata.get("owned_by") or "—")
info_cols[2].metric(
    "Context window",
    f'{metadata.get("context_window"):,}' if metadata.get("context_window") else "—",
)
info_cols[3].metric(
    "Max completion",
    f'{metadata.get("max_completion_tokens"):,}' if metadata.get("max_completion_tokens") else "—",
)


tabs = st.tabs(
    [
        "✍️ Generación",
        "🌡️ Comparador de temperatura",
        "🔢 Tokens y particiones",
        "🧮 Bag of Words",
        "📐 Similitud",
        "🧠 Embeddings",
        "📚 Modelos Groq",
    ]
)


# -----------------------------------------------------------------------------
# Tab 1: Generation
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Generación de texto")
    prompt = st.text_area(
        "Prompt del usuario",
        height=170,
        placeholder="Ejemplo: Explica qué es un embedding con un ejemplo sencillo.",
        key="generation_prompt",
    )

    a, b = st.columns([1, 1])
    with a:
        generate = st.button(
            "🚀 Generar respuesta",
            type="primary",
            use_container_width=True,
            disabled=not bool(api_key) or not bool(prompt.strip()),
        )
    with b:
        st.info(f"Temperature **{temperature:.2f}** · Top-p **{top_p:.2f}** · Max tokens **{max_tokens:,}**")

    if generate:
        try:
            client = Groq(api_key=api_key)
            with st.spinner("Generando con Groq..."):
                result = generate_one(client, model_id, system_prompt, prompt, temperature, top_p, max_tokens)

            st.markdown("### Respuesta")
            st.write(result["text"])

            usage = result["usage"]
            cols = st.columns(4)
            cols[0].metric("Prompt tokens", usage["prompt_tokens"] if usage["prompt_tokens"] is not None else "—")
            cols[1].metric("Completion tokens", usage["completion_tokens"] if usage["completion_tokens"] is not None else "—")
            cols[2].metric("Total tokens", usage["total_tokens"] if usage["total_tokens"] is not None else "—")
            cols[3].metric("Temperature", f"{temperature:.2f}")
        except Exception as exc:
            st.error(f"Error en la generación: {exc}")


# -----------------------------------------------------------------------------
# Tab 2: Temperature comparison
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🌡️ Comparar respuestas con diferentes temperaturas")
    st.write(
        "Usa exactamente el mismo prompt, modelo, system prompt, top-p y límite de tokens; "
        "solo cambia la temperatura. Esto facilita observar el efecto del muestreo."
    )

    temp_prompt = st.text_area(
        "Prompt para el experimento",
        value=(prompt if prompt else "Escribe una historia de 5 líneas sobre una ciudad futurista."),
        height=150,
        key="temperature_prompt",
    )

    temp_options = [round(x / 20, 2) for x in range(0, 41)]  # 0.00 ... 2.00
    default_temps = [0.0, 0.4, 0.8, 1.2, 1.6]
    selected_temperatures = st.multiselect(
        "Temperaturas a comparar",
        temp_options,
        default=default_temps,
        format_func=lambda x: f"{x:.2f}",
        max_selections=6,
        key="temperature_values",
    )

    run_compare = st.button(
        "🔬 Ejecutar comparación",
        type="primary",
        use_container_width=True,
        disabled=not bool(api_key) or not bool(temp_prompt.strip()) or len(selected_temperatures) < 2,
        key="run_temperature_comparison",
    )

    if run_compare:
        client = Groq(api_key=api_key)
        comparison_results: List[Dict[str, Any]] = []
        progress = st.progress(0.0, text="Preparando experimento...")
        errors: List[str] = []

        for i, temp_value in enumerate(selected_temperatures, start=1):
            try:
                result = generate_one(
                    client,
                    model_id,
                    system_prompt,
                    temp_prompt,
                    float(temp_value),
                    float(top_p),
                    int(max_tokens),
                    seed=int(seed) if use_seed else None,
                )
                comparison_results.append(result)
            except Exception as exc:
                errors.append(f"Temperature {temp_value:.2f}: {exc}")
            progress.progress(
                i / len(selected_temperatures),
                text=f"Generando {i}/{len(selected_temperatures)}...",
            )

        progress.empty()
        st.session_state["temperature_results"] = comparison_results
        if errors:
            for error in errors:
                st.warning(error)

    results = st.session_state.get("temperature_results", [])

    if results:
        st.markdown("### Respuestas")
        grid = st.columns(len(results))
        for col, result in zip(grid, results):
            with col:
                t = result["temperature"]
                st.markdown(f"**Temperature {t:.2f}**")
                safe_response = html.escape(result["text"]).replace("\n", "<br>")
                st.markdown(
                    f'<div class="response-card">{safe_response}</div>',
                    unsafe_allow_html=True,
                )

                usage = result["usage"]
                st.caption(
                    f"Prompt: {usage['prompt_tokens'] or '—'} · "
                    f"Completion: {usage['completion_tokens'] or '—'} · "
                    f"Total: {usage['total_tokens'] or '—'}"
                )

        st.markdown("### Métricas comparables")
        metric_rows = []
        for result in results:
            text = result["text"]
            token_df, token_count = token_analysis(text, encoding_name)
            words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
            unique_words = len(set(w.lower() for w in words))
            metric_rows.append(
                {
                    "temperature": result["temperature"],
                    "caracteres": len(text),
                    "palabras": len(words),
                    "palabras únicas": unique_words,
                    "tokens BPE": token_count,
                    "tokens/palabra": round(token_count / max(len(words), 1), 2),
                }
            )
        metric_df = pd.DataFrame(metric_rows).sort_values("temperature")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.markdown("### Similitud entre respuestas")
        response_texts = [r["text"] for r in results]
        if len(response_texts) >= 2:
            try:
                _, _, matrix = bow_analysis(response_texts, use_tfidf=True)
                response_cosine = cosine_similarity(matrix)
                labels = [f"T={r['temperature']:.2f}" for r in results]
                sim_df = pd.DataFrame(response_cosine, index=labels, columns=labels)
                st.dataframe(
                    sim_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.3f}"),
                    use_container_width=True,
                )
                st.caption(
                    "Esta matriz usa TF-IDF + similitud coseno como medida léxica. "
                    "Una similitud menor no implica por sí sola una respuesta peor: indica más diferencia textual."
                )
            except ValueError:
                st.info("No hay vocabulario suficiente para calcular similitud.")


# -----------------------------------------------------------------------------
# Tab 3: Tokenization and colored partitions
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🔢 Tokens, IDs y particiones coloreadas")
    token_text = st.text_area(
        "Texto para analizar",
        value=(prompt if prompt else "Los modelos de lenguaje procesan el texto en pequeñas unidades."),
        height=140,
        key="token_text",
    )

    methods = st.multiselect(
        "Métodos de segmentación",
        ["BPE / tiktoken", "Palabras + espacios", "Palabras + puntuación", "Caracteres"],
        default=["BPE / tiktoken", "Palabras + puntuación"],
        key="token_methods",
    )

    show_token_ids = st.checkbox("Mostrar token IDs cuando estén disponibles", value=True)

    if token_text.strip() and methods:
        for method in methods:
            try:
                toks, ids = method_tokens(token_text, method, encoding_name)
                st.markdown(f"#### {method}")
                st.markdown(colored_partition_html(toks, ids, show_token_ids), unsafe_allow_html=True)

                count_col, length_col = st.columns(2)
                count_col.metric("Particiones", len(toks))
                length_col.metric("Longitud media", f"{len(token_text) / max(len(toks), 1):.2f} caracteres")

                table = pd.DataFrame(
                    {
                        "posición": range(len(toks)),
                        "partición": toks,
                        "token_id": ids,
                    }
                )
                with st.expander(f"Ver tabla — {method}"):
                    st.dataframe(table.head(500), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"No se pudo aplicar {method}: {exc}")

        st.info(
            "Lectura pedagógica: BPE divide el texto en subunidades; los otros métodos muestran "
            "particiones alternativas. Los IDs son específicos del vocabulario/tokenizador usado."
        )


# -----------------------------------------------------------------------------
# Tab 4: BOW
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🧮 Bag of Words")
    bow_input = st.text_area(
        "Un documento por línea",
        value=(
            "Los modelos de lenguaje generan texto a partir de probabilidades.\n"
            "Los embeddings representan texto en vectores.\n"
            "La similitud compara representaciones de texto."
        ),
        height=160,
        key="bow_input",
    )
    use_tfidf = st.checkbox("Usar TF-IDF en lugar de conteos", value=False)
    documents = [line.strip() for line in bow_input.splitlines() if line.strip()]

    if documents:
        try:
            bow_table, vocab_table, matrix = bow_analysis(documents, use_tfidf=use_tfidf)
            c1, c2, c3 = st.columns(3)
            c1.metric("Documentos", len(documents))
            c2.metric("Vocabulario", len(vocab_table))
            c3.metric("Características", matrix.shape[1])
            st.dataframe(bow_table, use_container_width=True)
            st.markdown("#### Frecuencia de términos")
            st.dataframe(vocab_table.head(30), use_container_width=True, hide_index=True)
            if len(vocab_table) > 0:
                st.bar_chart(vocab_table.head(20).set_index("término")["frecuencia"])
        except ValueError as exc:
            st.warning(f"Se necesitan palabras válidas: {exc}")


# -----------------------------------------------------------------------------
# Tab 5: Similarity
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("📐 Métricas de similitud")
    similarity_input = st.text_area(
        "Un texto por línea",
        value=(
            "El gato duerme en el sofá.\n"
            "Un felino está descansando sobre el sofá.\n"
            "La economía global cambia con la tecnología."
        ),
        height=160,
        key="similarity_input",
    )
    sim_documents = [line.strip() for line in similarity_input.splitlines() if line.strip()]

    if len(sim_documents) >= 2:
        try:
            _, _, bow_matrix = bow_analysis(sim_documents, use_tfidf=True)
            cosine = cosine_similarity(bow_matrix)
            euclidean = euclidean_distances(bow_matrix)

            token_sets = [set(re.findall(r"\b\w+\b", doc.lower(), flags=re.UNICODE)) for doc in sim_documents]
            jaccard = np.eye(len(sim_documents))
            for i in range(len(sim_documents)):
                for j in range(i + 1, len(sim_documents)):
                    union = token_sets[i] | token_sets[j]
                    inter = token_sets[i] & token_sets[j]
                    score = len(inter) / len(union) if union else 0.0
                    jaccard[i, j] = score
                    jaccard[j, i] = score

            labels = [f"Texto {i + 1}" for i in range(len(sim_documents))]
            st.markdown("#### Coseno (TF-IDF)")
            st.dataframe(pd.DataFrame(cosine, index=labels, columns=labels).round(4), use_container_width=True)
            st.markdown("#### Distancia euclídea (TF-IDF)")
            st.dataframe(pd.DataFrame(euclidean, index=labels, columns=labels).round(4), use_container_width=True)
            st.markdown("#### Jaccard (palabras)")
            st.dataframe(pd.DataFrame(jaccard, index=labels, columns=labels).round(4), use_container_width=True)
        except ValueError as exc:
            st.error(f"No fue posible calcular las métricas: {exc}")
    else:
        st.info("Escribe al menos dos textos.")


# -----------------------------------------------------------------------------
# Tab 6: Embeddings
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("🧠 Embeddings semánticos")
    embedding_model_name = st.selectbox(
        "Modelo de embeddings local",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        index=1,
    )
    embedding_input = st.text_area(
        "Textos a embebder — uno por línea",
        value=(
            "Los gatos son animales domésticos.\n"
            "Un felino puede vivir con personas.\n"
            "La física estudia la materia y la energía."
        ),
        height=150,
        key="embedding_input",
    )
    embedding_documents = [line.strip() for line in embedding_input.splitlines() if line.strip()]

    if len(embedding_documents) >= 2:
        try:
            with st.spinner("Cargando modelo de embeddings..."):
                embedder = load_embedding_model(embedding_model_name)
                vectors = np.asarray(
                    embedder.encode(embedding_documents, normalize_embeddings=False, show_progress_bar=False)
                )
            normalized = normalize_rows(vectors)
            cosine_embeddings = normalized @ normalized.T

            c1, c2, c3 = st.columns(3)
            c1.metric("Textos", len(embedding_documents))
            c2.metric("Dimensión", vectors.shape[1])
            c3.metric("Norma media", f"{np.linalg.norm(vectors, axis=1).mean():.3f}")

            component_count = min(12, vectors.shape[1])
            partial = pd.DataFrame(
                vectors[:, :component_count],
                index=[f"Texto {i + 1}" for i in range(len(embedding_documents))],
                columns=[f"dim_{i}" for i in range(component_count)],
            )
            st.markdown("#### Vista parcial")
            st.dataframe(partial.round(5), use_container_width=True)

            st.markdown("#### Similaridad coseno entre embeddings")
            labels = [f"Texto {i + 1}" for i in range(len(embedding_documents))]
            cos_df = pd.DataFrame(cosine_embeddings, index=labels, columns=labels)
            st.dataframe(
                cos_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
                use_container_width=True,
            )
            with st.expander("Ver vectores completos"):
                st.dataframe(
                    pd.DataFrame(vectors, index=labels).round(6),
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"No se pudieron generar los embeddings: {exc}")
    else:
        st.info("Escribe al menos dos textos.")


# -----------------------------------------------------------------------------
# Tab 7: Models
# -----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("📚 Modelos disponibles en Groq")
    if models and model_source == "Groq (dinámico)" and api_key:
        model_table = pd.DataFrame(models)
        if "created" in model_table.columns:
            model_table["created"] = pd.to_datetime(model_table["created"], unit="s", errors="coerce")
        st.dataframe(model_table, use_container_width=True, hide_index=True)
    elif not api_key:
        st.warning("Ingresa la API key para consultar el catálogo de Groq.")
    else:
        st.info("Se está usando la lista de respaldo de modelos.")

st.divider()
st.caption(
    "Groq LLM Lab · La API key no se escribe en el código. Los análisis de tokenización y embeddings "
    "son locales, mientras que la generación se ejecuta mediante Groq."
)
