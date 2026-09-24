import html
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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
    .main .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
    .token-box {
        border: 1px solid rgba(128, 128, 128, .25);
        border-radius: 12px;
        padding: 16px;
        line-height: 2.4;
        background: rgba(127, 127, 127, .04);
        margin-bottom: 10px;
        overflow-x: auto;
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
