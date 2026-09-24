import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Groq LLM Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
        .metric-card {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 12px;
            padding: 12px 14px;
            background: rgba(127,127,127,.04);
        }
        code { white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 Groq LLM Lab")
st.caption(
    "Explora modelos, generación de texto, tokens, IDs, Bag of Words, "
    "métricas de similitud y embeddings."
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "groq/compound-mini",
]


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_api_key() -> str:
    """Read key from session input, then optionally from Streamlit secrets/env."""
    key = st.session_state.get("groq_api_key", "").strip()
    if key:
        return key

    # Optional deployment/local fallback.
    try:
        secret_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        secret_key = ""

    return secret_key


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(api_key: str) -> List[Dict[str, Any]]:
    client = Groq(api_key=api_key)
    response = client.models.list()
    models = []

    for item in getattr(response, "data", []):
        model = {
            "id": getattr(item, "id", None),
            "owned_by": getattr(item, "owned_by", None),
            "active": getattr(item, "active", None),
            "context_window": getattr(item, "context_window", None),
            "max_completion_tokens": getattr(item, "max_completion_tokens", None),
            "created": getattr(item, "created", None),
        }
        if model["id"]:
            models.append(model)

    models.sort(key=lambda x: x["id"])
    return models


def get_model_metadata(models: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    for model in models:
        if model["id"] == model_id:
            return model
    return {"id": model_id}


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
                "token": token_text.replace("\n", "\\n"),
            }
        )

    return pd.DataFrame(rows), len(token_ids)


def bow_analysis(texts: List[str], use_tfidf: bool = False):
    if use_tfidf:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )
    else:
        vectorizer = CountVectorizer(
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


# -----------------------------------------------------------------------------
# Sidebar: configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = st.text_input(
        "GROQ API Key",
        value="",
        type="password",
        placeholder="gsk_...",
        help="La clave se usa en memoria para las llamadas a Groq y no se muestra.",
        key="groq_api_key",
    )

    if not api_key:
        try:
            configured = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            configured = ""
        if configured:
            st.success("API key cargada desde Streamlit Secrets.")
        else:
            st.info("Ingresa tu API key para consultar Groq.")

    model_source = st.radio(
        "Catálogo de modelos",
        ["Groq (dinámico)", "Lista de respaldo"],
        index=0,
        help="El catálogo dinámico consulta los modelos activos disponibles para tu API key.",
    )

    models: List[Dict[str, Any]] = []

    if model_source == "Groq (dinámico)" and api_key:
        try:
            models = fetch_models(api_key)
            st.success(f"{len(models)} modelos devueltos por Groq.")
        except Exception as exc:
            st.warning(f"No se pudo consultar el catálogo dinámico: {exc}")

    if not models:
        models = [{"id": m} for m in FALLBACK_MODELS]

    model_ids = [m["id"] for m in models]
    default_model = (
        "llama-3.3-70b-versatile"
        if "llama-3.3-70b-versatile" in model_ids
        else model_ids[0]
    )

    model_id = st.selectbox(
        "Modelo",
        model_ids,
        index=model_ids.index(default_model),
    )

    st.divider()
    st.subheader("Parámetros de generación")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        help="Valores bajos tienden a producir respuestas más deterministas.",
    )

    top_p = st.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Controla el núcleo de probabilidad usado durante el muestreo.",
    )

    max_tokens = st.number_input(
        "Max completion tokens",
        min_value=64,
        max_value=65536,
        value=1024,
        step=64,
    )

    system_prompt = st.text_area(
        "System prompt",
        value="Eres un asistente útil, preciso y claro.",
        height=100,
    )

    st.divider()
    st.subheader("Tokenización local")

    encoding_name = st.selectbox(
        "Encoding tiktoken",
        ["cl100k_base", "o200k_base"],
        index=0,
        help=(
            "Sirve para análisis didáctico de tokens/IDs. "
            "Los IDs no deben interpretarse como los IDs internos exactos de cada LLM de Groq."
        ),
    )

    st.divider()
    st.caption(
        "Consejo: para despliegues, usa Streamlit Secrets o variables de entorno en lugar "
        "de guardar claves en el código."
    )

# -----------------------------------------------------------------------------
# Model info strip
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
    f'{metadata.get("max_completion_tokens"):,}'
    if metadata.get("max_completion_tokens")
    else "—",
)

tabs = st.tabs(
    [
        "✍️ Generación",
        "🔢 Tokens e IDs",
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
        height=180,
        placeholder="Ejemplo: Explica qué es un embedding con un ejemplo sencillo.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        generate = st.button(
            "🚀 Generar respuesta",
            type="primary",
            use_container_width=True,
            disabled=not bool(api_key.strip()) or not bool(prompt.strip()),
        )

    with col2:
        st.caption(
            f"Temperature: **{temperature:.2f}** · Top-p: **{top_p:.2f}** · "
            f"Máx. tokens: **{max_tokens:,}**"
        )

    if generate:
        try:
            client = Groq(api_key=api_key.strip())

            with st.spinner("Generando con Groq..."):
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_completion_tokens=int(max_tokens),
                )

            answer = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)

            st.markdown("### Respuesta")
            st.markdown(answer)

            st.divider()
            st.markdown("### Consumo reportado por la API")

            usage_cols = st.columns(4)
            prompt_tokens = safe_int(getattr(usage, "prompt_tokens", None))
            completion_tokens = safe_int(getattr(usage, "completion_tokens", None))
            total_tokens = safe_int(getattr(usage, "total_tokens", None))

            usage_cols[0].metric("Prompt tokens", prompt_tokens if prompt_tokens is not None else "—")
            usage_cols[1].metric(
                "Completion tokens",
                completion_tokens if completion_tokens is not None else "—",
            )
            usage_cols[2].metric("Total tokens", total_tokens if total_tokens is not None else "—")
            usage_cols[3].metric("Temperatura", f"{temperature:.2f}")

            if usage is not None:
                with st.expander("Detalle de usage"):
                    try:
                        st.json(usage.model_dump())
                    except Exception:
                        st.write(usage)

        except Exception as exc:
            st.error(f"Error en la generación: {exc}")

# -----------------------------------------------------------------------------
# Tab 2: Tokens
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Tokens e IDs")

    token_text = st.text_area(
        "Texto a tokenizar",
        value=prompt if prompt else "Los embeddings representan significado en un espacio vectorial.",
        height=140,
    )

    if token_text.strip():
        try:
            token_df, token_count = token_analysis(token_text, encoding_name)
            col1, col2, col3 = st.columns(3)
            col1.metric("Número de tokens", token_count)
            col2.metric("Caracteres", len(token_text))
            col3.metric(
                "Caracteres/token",
                f"{len(token_text) / max(token_count, 1):.2f}",
            )

            st.markdown("#### Tabla de tokenización")
            st.dataframe(token_df.head(500), use_container_width=True, hide_index=True)

            with st.expander("Ver IDs como lista"):
                st.code(str(token_df["token_id"].tolist()[:500]), language="text")

            st.info(
                "Nota: esta tabla usa tiktoken de forma local para inspección pedagógica. "
                "El tokenizador de un modelo de Groq puede producir tokens/IDs diferentes."
            )
        except Exception as exc:
            st.error(f"No se pudo tokenizar el texto: {exc}")

# -----------------------------------------------------------------------------
# Tab 3: BOW
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Bag of Words")

    bow_input = st.text_area(
        "Un documento por línea",
        value=(
            "Los modelos de lenguaje generan texto a partir de probabilidades.\n"
            "Los embeddings representan texto en vectores.\n"
            "La similitud compara representaciones de texto."
        ),
        height=170,
    )

    use_tfidf = st.checkbox(
        "Usar TF-IDF en lugar de conteos crudos",
        value=False,
    )

    documents = [line.strip() for line in bow_input.splitlines() if line.strip()]

    if len(documents) >= 1:
        try:
            bow_table, vocab_table, matrix = bow_analysis(documents, use_tfidf=use_tfidf)

            c1, c2, c3 = st.columns(3)
            c1.metric("Documentos", len(documents))
            c2.metric("Vocabulario", len(vocab_table))
            c3.metric("Características", matrix.shape[1])

            st.markdown("#### Matriz")
            st.dataframe(bow_table, use_container_width=True)

            st.markdown("#### Frecuencia de términos")
            st.dataframe(vocab_table.head(30), use_container_width=True, hide_index=True)

            if len(vocab_table) > 0:
                chart_df = vocab_table.head(20).set_index("término")
                st.bar_chart(chart_df["frecuencia"])

        except ValueError as exc:
            st.warning(f"Se necesitan palabras válidas para construir el vocabulario: {exc}")

# -----------------------------------------------------------------------------
# Tab 4: Similarity
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Métricas de similitud")

    similarity_input = st.text_area(
        "Un texto por línea",
        value=(
            "El gato duerme en el sofá.\n"
            "Un felino está descansando sobre el sofá.\n"
            "La economía global cambia con la tecnología."
        ),
        height=170,
    )

    sim_documents = [line.strip() for line in similarity_input.splitlines() if line.strip()]

    if len(sim_documents) >= 2:
        try:
            _, _, bow_matrix = bow_analysis(sim_documents, use_tfidf=True)

            cosine = cosine_similarity(bow_matrix)
            euclidean = euclidean_distances(bow_matrix)

            # Jaccard over word sets.
            token_sets = [
                set(doc.lower().split())
                for doc in sim_documents
            ]
            jaccard = np.eye(len(sim_documents))
            for i in range(len(sim_documents)):
                for j in range(i + 1, len(sim_documents)):
                    union = token_sets[i] | token_sets[j]
                    inter = token_sets[i] & token_sets[j]
                    score = len(inter) / len(union) if union else 0.0
                    jaccard[i, j] = score
                    jaccard[j, i] = score

            st.markdown("#### Coseno (TF-IDF)")
            cosine_df = pd.DataFrame(
                cosine,
                index=[f"Texto {i+1}" for i in range(len(sim_documents))],
                columns=[f"Texto {i+1}" for i in range(len(sim_documents))],
            )
            st.dataframe(cosine_df.round(4), use_container_width=True)

            st.markdown("#### Distancia euclídea (TF-IDF)")
            euclidean_df = pd.DataFrame(
                euclidean,
                index=[f"Texto {i+1}" for i in range(len(sim_documents))],
                columns=[f"Texto {i+1}" for i in range(len(sim_documents))],
            )
            st.dataframe(euclidean_df.round(4), use_container_width=True)

            st.markdown("#### Jaccard (conjuntos de palabras)")
            jaccard_df = pd.DataFrame(
                jaccard,
                index=[f"Texto {i+1}" for i in range(len(sim_documents))],
                columns=[f"Texto {i+1}" for i in range(len(sim_documents))],
            )
            st.dataframe(jaccard_df.round(4), use_container_width=True)

            st.info(
                "Coseno/Jaccard altos suelen indicar mayor similitud; en distancia euclídea, "
                "valores menores indican mayor proximidad. Son métricas sobre representaciones "
                "distintas y no deben compararse como si midieran exactamente lo mismo."
            )
        except ValueError as exc:
            st.error(f"No fue posible calcular las métricas: {exc}")
    else:
        st.info("Escribe al menos dos textos, uno por línea.")

# -----------------------------------------------------------------------------
# Tab 5: Embeddings
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Embeddings semánticos")

    embedding_model_name = st.selectbox(
        "Modelo de embeddings local",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        index=1,
        help="Los embeddings se generan localmente; esto no consume tokens de Groq.",
    )

    embedding_input = st.text_area(
        "Textos a embebder — uno por línea",
        value=(
            "Los gatos son animales domésticos.\n"
            "Un felino puede vivir con personas.\n"
            "La física estudia la materia y la energía."
        ),
        height=160,
    )

    embedding_documents = [
        line.strip() for line in embedding_input.splitlines() if line.strip()
    ]

    if len(embedding_documents) >= 2:
        try:
            with st.spinner("Cargando modelo de embeddings..."):
                embedder = load_embedding_model(embedding_model_name)
                vectors = embedder.encode(
                    embedding_documents,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )

            vectors = np.asarray(vectors)
            normalized = normalize_rows(vectors)
            cosine_embeddings = normalized @ normalized.T

            c1, c2, c3 = st.columns(3)
            c1.metric("Textos", len(embedding_documents))
            c2.metric("Dimensión", vectors.shape[1])
            c3.metric("Norma media", f"{np.linalg.norm(vectors, axis=1).mean():.3f}")

            st.markdown("#### Vista parcial de los vectores")
            component_count = min(12, vectors.shape[1])
            embedding_df = pd.DataFrame(
                vectors[:, :component_count],
                index=[f"Texto {i+1}" for i in range(len(embedding_documents))],
                columns=[f"dim_{i}" for i in range(component_count)],
            )
            st.dataframe(embedding_df.round(5), use_container_width=True)

            st.markdown("#### Similaridad coseno entre embeddings")
            cos_df = pd.DataFrame(
                cosine_embeddings,
                index=[f"Texto {i+1}" for i in range(len(embedding_documents))],
                columns=[f"Texto {i+1}" for i in range(len(embedding_documents))],
            )
            st.dataframe(cos_df.round(4), use_container_width=True)

            # Heatmap-like display using native dataframe styling.
            st.markdown("#### Mapa de similitud")
            st.dataframe(
                cos_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
                use_container_width=True,
            )

            with st.expander("Ver los vectores completos"):
                st.dataframe(
                    pd.DataFrame(
                        vectors,
                        index=[f"Texto {i+1}" for i in range(len(embedding_documents))],
                    ).round(6),
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(
                "No se pudieron generar los embeddings. "
                "Puede deberse a descarga/carga del modelo o a dependencias: "
                f"{exc}"
            )
    else:
        st.info("Escribe al menos dos textos, uno por línea.")

# -----------------------------------------------------------------------------
# Tab 6: Models
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Modelos disponibles en Groq")

    if api_key:
        if models and model_source == "Groq (dinámico)":
            model_table = pd.DataFrame(models)
            if "created" in model_table.columns:
                model_table["created"] = pd.to_datetime(
                    model_table["created"],
                    unit="s",
                    errors="coerce",
                )
            st.dataframe(
                model_table,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Se está usando la lista de respaldo. Activa el catálogo dinámico "
                "para consultar directamente a Groq."
            )
    else:
        st.warning("Ingresa la API key para consultar la lista de modelos de Groq.")

st.divider()
st.caption(
    "Groq LLM Lab · La API key se mantiene en la sesión de Streamlit. "
    "No se guarda en el código."
)

