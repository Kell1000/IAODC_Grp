"""
Etape 11 du TP - Variante Gradio, destinee a Hugging Face Spaces.

POURQUOI UNE SECONDE INTERFACE ?
--------------------------------
Verifie sur la documentation officielle de Hugging Face (docs/hub,
"Spaces Overview", consulte en octobre 2026) :

    "Static Spaces are free for everyone. Gradio and Docker Spaces run on
     compute and require a paid plan to create: PRO for personal accounts,
     Team or Enterprise for organizations. Free personal accounts in good
     standing can still host up to 2 Gradio Spaces running on ZeroGPU."

Trois consequences concretes pour un TP a 0 euro :
  1. Streamlit n'est plus un SDK propose par Spaces (seuls Gradio, Docker et
     HTML statique le sont). Une application Streamlit ne peut donc pas etre
     deployee telle quelle.
  2. Un Space Docker demande un compte payant : la voie Docker est fermee.
  3. La seule voie gratuite sur Spaces est un Space GRADIO (2 maximum par
     compte personnel). D'ou ce fichier.

ATTENTION : ces regles sont celles constatees a la date ci-dessus. Hugging
Face peut les changer. VERIFIEZ la page officielle avant de deployer, et
reportez-vous a docs/DEPLOYMENT.md pour la solution de secours.

Lancement local :
    python app_gradio.py
    -> http://localhost:7860

Sur Spaces, le fichier doit s'appeler app.py (voir docs/DEPLOYMENT.md).
"""

from __future__ import annotations

import os

import gradio as gr

from src.config import CFG
from src.rag import RagPipeline, load_chunks

# --------------------------------------------------------------------------
#  Chargement unique au demarrage
# --------------------------------------------------------------------------
# Sur l'hebergement gratuit, la RAM est tres limitee (de l'ordre du Go).
# On force donc la recherche lexicale : elle ne charge ni torch ni modele
# d'embeddings. La qualite baisse un peu, mais l'application tient dans
# l'enveloppe memoire et demarre en quelques secondes.
FORCE_TFIDF = os.getenv("FORCE_TFIDF", "1") == "1"
RETRIEVER_KIND = "tfidf" if FORCE_TFIDF else CFG.retriever

PIPELINE: RagPipeline | None = None
N_CHUNKS = 0
LOAD_ERROR = ""

try:
    _chunks = load_chunks()
    if _chunks:
        from src.rag import get_retriever

        PIPELINE = RagPipeline(
            retriever=get_retriever(RETRIEVER_KIND, chunks=_chunks), chunks=_chunks
        )
        N_CHUNKS = len(_chunks)
    else:
        LOAD_ERROR = (
            "Aucun corpus trouve. Le depot doit contenir "
            "data/processed/chunks.jsonl ou data/processed/demo_chunks.jsonl."
        )
except Exception as exc:  # noqa: BLE001 - l'app doit demarrer pour afficher l'erreur
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def format_sources(sources: list[dict]) -> str:
    """Met en forme les sources en Markdown."""
    if not sources:
        return "_Aucune source : aucun passage pertinent n'a ete trouve._"
    lines = ["### Sources Wikipedia", ""]
    for source in sources:
        lines.append(
            f"**[{source['n']}] {source['title']}** "
            f"(similarite {source['score']})  \n"
            f"[{source['url']}]({source['url']}) - licence {source['license']}"
        )
        lines.append("")
    return "\n".join(lines)


def format_passages(passages) -> str:
    """Affiche le contexte exact transmis au LLM (transparence du RAG)."""
    if not passages:
        return "_Aucun passage._"
    lines = [
        "Contexte exact recu par le modele. Toute information de la reponse "
        "absente d'ici est une hallucination.",
        "",
    ]
    for passage in passages:
        text = passage.text[:700] + ("..." if len(passage.text) > 700 else "")
        lines += [
            f"**[{passage.rank}] {passage.title}** - similarite {passage.score:.4f}",
            "",
            f"> {text}",
            "",
        ]
    return "\n".join(lines)


def respond(question: str, history, provider: str, top_k: int, min_score: float):
    """Fonction appelee a chaque envoi. Renvoie l'etat complet de l'interface."""
    history = history or []

    if PIPELINE is None:
        error = f"**L'application n'a pas pu demarrer.**\n\n`{LOAD_ERROR}`"
        history = history + [
            {"role": "user", "content": question or ""},
            {"role": "assistant", "content": error},
        ]
        return history, "", "", ""

    question = (question or "").strip()
    if not question:
        return history, "", "", ""

    try:
        answer = PIPELINE.answer(
            question, top_k=int(top_k), min_score=float(min_score), provider=provider
        )
    except Exception as exc:  # noqa: BLE001
        history = history + [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"**Erreur technique.**\n\n`{type(exc).__name__}: {exc}`",
            },
        ]
        return history, "", "", ""

    reply = answer.answer
    if not answer.has_context:
        reply = (
            f"{answer.answer}\n\n"
            f"_(meilleur score : {answer.best_score:.3f}, seuil requis : "
            f"{min_score}. Reformulez ou baissez le seuil.)_"
        )
    if not answer.llm_ok:
        reply += (
            f"\n\n---\n_Le fournisseur `{provider}` a echoue ; reponse produite "
            f"en mode extractif. Detail : {answer.llm_error.splitlines()[0]}_"
        )

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]

    metrics = (
        f"{answer.provider} / {answer.model} | "
        f"recherche {answer.retrieval_time_s:.2f}s | "
        f"generation {answer.generation_time_s:.2f}s | "
        f"total {answer.total_time_s:.2f}s | "
        f"meilleur score {answer.best_score:.3f}"
    )
    return history, format_sources(answer.sources), format_passages(answer.passages), metrics


# --------------------------------------------------------------------------
#  Interface
# --------------------------------------------------------------------------
with gr.Blocks(title="WikiNLP Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# WikiNLP Assistant\n"
        "Posez une question : l'assistant cherche la reponse dans un corpus "
        "d'articles Wikipedia, puis cite ses sources."
    )

    if LOAD_ERROR:
        gr.Markdown(f"> **Probleme au demarrage :** `{LOAD_ERROR}`")
    else:
        gr.Markdown(
            f"> Corpus : **{N_CHUNKS} passages** | moteur : "
            f"**{'lexical TF-IDF' if RETRIEVER_KIND == 'tfidf' else 'semantique'}**"
            + ("  \n> _Mode leger impose par la RAM de l'hebergement gratuit._"
               if FORCE_TFIDF else "")
        )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation", height=420, type="messages",
                show_copy_button=True,
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ex : Qu'est-ce que le traitement automatique des langues ?",
                    show_label=False, scale=5, container=False,
                )
                send_button = gr.Button("Envoyer", variant="primary", scale=1)
            clear_button = gr.Button("Effacer la conversation", size="sm")
            metrics_box = gr.Markdown("")

        with gr.Column(scale=2):
            provider_dropdown = gr.Dropdown(
                choices=["extractive", "groq", "gemini", "ollama"],
                value=CFG.llm_provider,
                label="Fournisseur LLM",
                info="extractive = sans cle API ni reseau",
            )
            top_k_slider = gr.Slider(1, 10, value=CFG.top_k, step=1,
                                     label="Passages recuperes (top_k)")
            min_score_slider = gr.Slider(
                0.0, 1.0, value=CFG.min_score_for(RETRIEVER_KIND), step=0.01,
                label="Seuil de pertinence",
                info="En dessous, l'assistant refuse de repondre.",
            )
            sources_box = gr.Markdown("")
            with gr.Accordion("Passages transmis au LLM", open=False):
                passages_box = gr.Markdown("")

    gr.Examples(
        examples=[
            "Qu'est-ce que le traitement automatique des langues ?",
            "A quoi sert l'apprentissage profond ?",
            "Qu'est-ce que l'intelligence artificielle ?",
            "Quelle est la recette de la tarte tatin ?",
        ],
        inputs=question_box,
        label="Exemples (le dernier est hors corpus : l'assistant doit refuser)",
    )

    gr.Markdown(
        "---\n"
        "Contenus issus de Wikipedia, licence CC BY-SA 4.0. Les reponses sont "
        "generees automatiquement a partir des extraits cites : verifiez "
        "toujours les sources."
    )

    inputs = [question_box, chatbot, provider_dropdown, top_k_slider, min_score_slider]
    outputs = [chatbot, sources_box, passages_box, metrics_box]

    send_button.click(respond, inputs=inputs, outputs=outputs).then(
        lambda: "", outputs=question_box
    )
    question_box.submit(respond, inputs=inputs, outputs=outputs).then(
        lambda: "", outputs=question_box
    )
    clear_button.click(lambda: ([], "", "", ""), outputs=outputs)


if __name__ == "__main__":
    # server_name="0.0.0.0" est indispensable dans un conteneur : sinon le
    # serveur n'ecoute que sur la boucle locale et reste injoignable.
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
