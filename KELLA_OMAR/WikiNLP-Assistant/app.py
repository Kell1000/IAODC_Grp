"""
Etape 6 du TP - Interface web du chatbot (Streamlit).

Lancement :
    streamlit run app.py
    streamlit run app.py --server.port 8502      # si le port 8501 est occupe

Pourquoi Streamlit et non FastAPI + HTML/CSS/JS ?
  - Un seul langage (Python), aucun fichier front a maintenir.
  - Le rechargement a chaud accelere beaucoup les TP.
  - L'essentiel du TP porte sur le NLP et le RAG, pas sur le CSS.
Une version Gradio (app_gradio.py) existe pour le deploiement sur
Hugging Face Spaces, ou Streamlit n'est plus un SDK propose.

Points d'attention specifiques a Streamlit
  - Le script est RE-EXECUTE en entier a chaque interaction. Tout etat doit
    vivre dans st.session_state, sinon il est perdu a chaque clic.
  - @st.cache_resource garde le pipeline (et donc le modele d'embeddings) en
    memoire entre les interactions : sans lui, on rechargerait 470 Mo de
    modele a chaque question.
"""

from __future__ import annotations

import time

import streamlit as st

from src.config import CFG
from src.rag import RagPipeline, load_chunks

# --------------------------------------------------------------------------
#  Configuration de la page : doit etre le PREMIER appel Streamlit
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="WikiNLP Assistant",
    page_icon="W",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
#  Styles. Les variables de couleur s'adaptent au theme clair/sombre choisi
#  par l'utilisateur dans Streamlit, au lieu de forcer un fond blanc.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main > div { max-width: 820px; }

      .wiki-header {
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1f3a5f 0%, #2d6a9f 100%);
        color: #ffffff;
        margin-bottom: 1.2rem;
      }
      .wiki-header h1 { margin: 0; font-size: 1.6rem; color: #ffffff; }
      .wiki-header p  { margin: .35rem 0 0; opacity: .9; font-size: .92rem; }

      .msg-user, .msg-bot {
        padding: .8rem 1rem;
        border-radius: 10px;
        margin-bottom: .6rem;
        border-left: 4px solid;
        background: rgba(128, 128, 128, 0.10);
      }
      .msg-user { border-left-color: #2d6a9f; }
      .msg-bot  { border-left-color: #2e8b57; }
      .msg-bot.no-answer { border-left-color: #d08700; }
      .msg-label { font-size: .74rem; text-transform: uppercase;
                   letter-spacing: .06em; opacity: .65; margin-bottom: .3rem; }

      .source-card {
        padding: .55rem .8rem;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: .4rem;
        font-size: .88rem;
      }
      .metrics-line { font-size: .78rem; opacity: .65; margin-top: .3rem; }

      /* Lisible sur telephone : la grille des metriques passe en colonne */
      @media (max-width: 640px) {
        .wiki-header h1 { font-size: 1.25rem; }
        .main > div { padding-left: .6rem; padding-right: .6rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
#  Chargement du pipeline (mis en cache entre les interactions)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline(retriever_kind: str):
    """Construit le pipeline RAG une seule fois par type de retriever.

    retriever_kind est un argument (et pas une lecture de CFG) pour que
    Streamlit invalide le cache automatiquement quand l'utilisateur change
    de moteur dans la barre laterale.
    """
    chunks = load_chunks()
    if not chunks:
        return None, 0
    from src.rag import get_retriever

    retriever = get_retriever(retriever_kind, chunks=chunks)
    return RagPipeline(retriever=retriever, chunks=chunks), len(chunks)


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending" not in st.session_state:
        st.session_state.pending = None


init_state()

# --------------------------------------------------------------------------
#  Barre laterale : configuration et diagnostic
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Configuration")

    retriever_kind = st.radio(
        "Moteur de recherche",
        options=["embeddings", "tfidf"],
        index=0 if CFG.retriever == "embeddings" else 1,
        format_func=lambda value: (
            "Semantique (embeddings)" if value == "embeddings" else "Lexical (TF-IDF)"
        ),
        help=(
            "Semantique : comprend les synonymes, telecharge un modele.\n"
            "Lexical : instantane, aucun telechargement, aveugle aux synonymes."
        ),
    )

    provider_options = ["extractive", "groq", "gemini", "ollama"]
    provider = st.selectbox(
        "Fournisseur LLM",
        options=provider_options,
        index=provider_options.index(CFG.llm_provider)
        if CFG.llm_provider in provider_options else 0,
        help="extractive = mode demonstration, sans cle API et sans reseau.",
    )

    top_k = st.slider("Passages a recuperer (top_k)", 1, 10, CFG.top_k)
    # Le seuil par defaut suit le moteur choisi : les scores TF-IDF et
    # semantiques ne sont pas sur la meme echelle (voir src/config.py).
    min_score = st.slider(
        "Seuil de pertinence (min_score)", 0.0, 1.0,
        CFG.min_score_for(retriever_kind), 0.01,
        help=(
            "En dessous de ce score, l'assistant refuse de repondre plutot "
            "que d'inventer. Montez-le s'il repond a des questions hors sujet ; "
            "baissez-le s'il refuse des questions legitimes."
        ),
    )

    st.divider()
    st.subheader("Etat du systeme")

    ready, message = CFG.llm_is_ready()
    if provider == "extractive":
        st.info("Mode demonstration : aucune cle API requise.")
    elif (provider == "groq" and not CFG.groq_api_key) or (
        provider == "gemini" and not CFG.gemini_api_key
    ):
        st.warning(
            f"Aucune cle {provider.upper()} dans .env.\n\n"
            "L'assistant basculera automatiquement en mode extractif si "
            "l'appel echoue."
        )
    else:
        st.success(message)

    st.caption(f"Langue du corpus : {CFG.wiki_lang}")
    st.caption(f"Modele d'embeddings : {CFG.embedding_model.split('/')[-1]}")
    st.caption(f"Base vectorielle : {CFG.vector_backend}")

    st.divider()
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    with st.expander("Comment enrichir le corpus ?"):
        st.code(
            "python -m src.collect_wikipedia --max 40\n"
            "python -m src.preprocess\n"
            "python -m src.rag --build",
            language="powershell",
        )
        st.caption("Puis rechargez cette page.")

# --------------------------------------------------------------------------
#  En-tete
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="wiki-header">
      <h1>WikiNLP Assistant</h1>
      <p>Posez une question : l'assistant cherche la reponse dans un corpus
         d'articles Wikipedia et cite ses sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
#  Chargement du corpus et de l'index
# --------------------------------------------------------------------------
with st.spinner("Chargement de l'index (premier lancement : peut prendre 1 a 3 min)..."):
    try:
        pipeline, n_chunks = get_pipeline(retriever_kind)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"**Impossible de construire l'index.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Verifiez que le pipeline de donnees a bien ete execute."
        )
        st.stop()

if pipeline is None:
    st.error("**Aucun corpus trouve.**")
    st.markdown(
        "L'assistant n'a aucun document a interroger. Executez le pipeline "
        "de donnees, puis rechargez cette page :"
    )
    st.code(
        "python -m src.collect_wikipedia\n"
        "python -m src.preprocess\n"
        "python -m src.rag --build",
        language="powershell",
    )
    st.stop()

st.caption(
    f"Corpus charge : {n_chunks} passages | moteur : "
    f"{'semantique' if retriever_kind == 'embeddings' else 'lexical'} | "
    f"LLM : {provider}"
)

# --------------------------------------------------------------------------
#  Questions suggerees
# --------------------------------------------------------------------------
if not st.session_state.history:
    st.markdown("**Exemples de questions :**")
    suggestions = [
        "Qu'est-ce que le traitement automatique des langues ?",
        "A quoi sert l'apprentissage profond ?",
        "Qu'est-ce que l'intelligence artificielle ?",
    ]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions):
        with column:
            if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion}"):
                st.session_state.pending = suggestion
                st.rerun()

# --------------------------------------------------------------------------
#  Formulaire de saisie
# --------------------------------------------------------------------------
with st.form("question_form", clear_on_submit=True):
    columns = st.columns([5, 1])
    with columns[0]:
        question_input = st.text_input(
            "Votre question",
            placeholder="Ex : Qui a invente l'architecture Transformer ?",
            label_visibility="collapsed",
        )
    with columns[1]:
        submitted = st.form_submit_button("Envoyer", use_container_width=True,
                                          type="primary")

question = None
if submitted and question_input.strip():
    question = question_input.strip()
elif st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

# --------------------------------------------------------------------------
#  Traitement de la question
# --------------------------------------------------------------------------
if question:
    with st.spinner("Recherche dans le corpus et redaction de la reponse..."):
        start = time.perf_counter()
        try:
            answer = pipeline.answer(
                question, top_k=top_k, min_score=min_score, provider=provider
            )
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": answer.answer,
                    "sources": answer.sources,
                    "has_context": answer.has_context,
                    "best_score": answer.best_score,
                    "provider": answer.provider,
                    "model": answer.model,
                    "retrieval_time_s": answer.retrieval_time_s,
                    "generation_time_s": answer.generation_time_s,
                    "total_time_s": answer.total_time_s,
                    "llm_ok": answer.llm_ok,
                    "llm_error": answer.llm_error,
                    "passages": [
                        {
                            "rank": passage.rank,
                            "score": passage.score,
                            "title": passage.title,
                            "text": passage.text,
                            "url": passage.url,
                        }
                        for passage in answer.passages
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001 - l'UI ne doit jamais crasher
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": None,
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                    "total_time_s": time.perf_counter() - start,
                }
            )
    st.rerun()

# --------------------------------------------------------------------------
#  Affichage de la conversation (la plus recente en haut)
# --------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("Conversation")

    for index, turn in enumerate(reversed(st.session_state.history)):
        st.markdown(
            f'<div class="msg-user"><div class="msg-label">Vous</div>'
            f"{turn['question']}</div>",
            unsafe_allow_html=True,
        )

        # --- Erreur fatale ---
        if turn.get("fatal_error"):
            st.error(
                f"**Une erreur technique s'est produite.**\n\n"
                f"`{turn['fatal_error']}`\n\n"
                "Consultez le terminal pour la trace complete."
            )
            continue

        # --- Aucun passage pertinent ---
        if not turn["has_context"]:
            st.markdown(
                f'<div class="msg-bot no-answer">'
                f'<div class="msg-label">Assistant - aucun passage pertinent</div>'
                f"{turn['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Meilleur score obtenu : {turn['best_score']:.3f} "
                f"(seuil requis : {min_score}). Baissez le seuil dans la barre "
                "laterale, reformulez, ou collectez des articles sur ce theme."
            )
            continue

        # --- Reponse normale ---
        st.markdown(
            f'<div class="msg-bot"><div class="msg-label">Assistant</div>'
            f"{turn['answer']}</div>",
            unsafe_allow_html=True,
        )

        # Le LLM distant a echoue : on l'annonce sans masquer la reponse de repli
        if not turn.get("llm_ok", True):
            with st.expander("Le fournisseur LLM a echoue (repli extractif)"):
                st.warning(turn.get("llm_error", "Erreur inconnue"))
                st.caption(
                    "La reponse ci-dessus vient du mode extractif. Causes "
                    "frequentes : quota depasse, cle invalide, reseau coupe."
                )

        # --- Sources ---
        if turn["sources"]:
            st.markdown("**Sources Wikipedia utilisees**")
            for source in turn["sources"]:
                st.markdown(
                    f'<div class="source-card">'
                    f"[{source['n']}] <b>{source['title']}</b> "
                    f"(similarite {source['score']})<br>"
                    f"<a href=\"{source['url']}\" target=\"_blank\">{source['url']}</a>"
                    f" &middot; licence {source['license']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # --- Passages bruts : la transparence du RAG ---
        with st.expander(f"Voir les {len(turn['passages'])} passages transmis au LLM"):
            st.caption(
                "C'est exactement le contexte recu par le modele. Si la reponse "
                "contient une information absente d'ici, c'est une hallucination."
            )
            for passage in turn["passages"]:
                st.markdown(
                    f"**[{passage['rank']}] {passage['title']}** "
                    f"- similarite {passage['score']:.4f}"
                )
                st.text(passage["text"][:600] + ("..." if len(passage["text"]) > 600 else ""))
                st.divider()

        st.markdown(
            f'<div class="metrics-line">'
            f"{turn['provider']} / {turn['model']} &middot; "
            f"recherche {turn['retrieval_time_s']:.2f}s &middot; "
            f"generation {turn['generation_time_s']:.2f}s &middot; "
            f"total {turn['total_time_s']:.2f}s &middot; "
            f"meilleur score {turn['best_score']:.3f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        if index < len(st.session_state.history) - 1:
            st.divider()

# --------------------------------------------------------------------------
#  Pied de page
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "WikiNLP Assistant - TP NLP/RAG. Contenus issus de Wikipedia, "
    "sous licence CC BY-SA 4.0. Les reponses sont generees automatiquement "
    "a partir des extraits cites : verifiez toujours les sources."
)
