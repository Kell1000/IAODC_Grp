"""
Fixtures partagees par tous les tests.

PRINCIPE : les tests ne doivent JAMAIS dependre du reseau ni d'un modele
telecharge. Sinon ils echouent dans la CI de GitHub, ou dans le train. On
utilise donc un mini-corpus ecrit en dur ici, et on force :

    RETRIEVER=tfidf          -> aucun telechargement de modele
    LLM_PROVIDER=extractive  -> aucun appel reseau

Ces variables sont posees AVANT tout import de src.config, car la
configuration est lue une seule fois a l'import du module.
"""

from __future__ import annotations

import os

# /!\ Doit venir avant les imports de src.* (voir docstring ci-dessus)
os.environ.setdefault("RETRIEVER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "extractive")
os.environ.setdefault("WIKI_LANG", "fr")

import pytest  # noqa: E402

# --------------------------------------------------------------------------
#  Mini-corpus de test : 3 documents, sujets nettement distincts pour que
#  les assertions de recherche soient sans ambiguite.
# --------------------------------------------------------------------------
RAW_ARTICLES = [
    {
        "pageid": 101,
        "title": "Traitement automatique des langues",
        "url": "https://fr.wikipedia.org/wiki/TAL",
        "lang": "fr",
        "license": "CC BY-SA 4.0",
        "text": (
            "== Definition ==\n"
            "Le traitement automatique des langues est un domaine a la croisee "
            "de la linguistique et de l'informatique [1]. Il vise a construire "
            "des programmes capables de traiter le langage humain ecrit ou "
            "parle. Les applications sont nombreuses : traduction automatique, "
            "resume de texte, analyse de sentiment et agents conversationnels.\n"
            "\n\n\n"
            "== Methodes ==\n"
            "Les approches statistiques ont longtemps domine le domaine. "
            "Depuis 2017, les architectures neuronales de type transformeur "
            "donnent les meilleurs resultats sur la plupart des taches.\n"
            "== Voir aussi ==\n"
            "Liste de logiciels de TAL, Bibliographie complete."
        ),
    },
    {
        "pageid": 102,
        "title": "Apprentissage profond",
        "url": "https://fr.wikipedia.org/wiki/Apprentissage_profond",
        "lang": "fr",
        "license": "CC BY-SA 4.0",
        "text": (
            "L'apprentissage profond est un ensemble de methodes "
            "d'apprentissage automatique fondees sur des reseaux de neurones "
            "artificiels comportant de nombreuses couches. Ces reseaux "
            "apprennent des representations hierarchiques des donnees. "
            "L'entrainement necessite de grandes quantites de donnees et une "
            "puissance de calcul importante, souvent fournie par des cartes "
            "graphiques. Les domaines d'application incluent la vision par "
            "ordinateur et la reconnaissance de la parole."
        ),
    },
    {
        "pageid": 103,
        "title": "Cuisine italienne",
        "url": "https://fr.wikipedia.org/wiki/Cuisine_italienne",
        "lang": "fr",
        "license": "CC BY-SA 4.0",
        "text": (
            "La cuisine italienne est reputee pour ses pates, ses pizzas et "
            "son huile d'olive. Chaque region possede ses specialites "
            "culinaires propres. Le risotto vient du nord du pays, tandis que "
            "la pizza napolitaine est originaire de Naples. Les tomates, le "
            "basilic et le parmesan comptent parmi les ingredients les plus "
            "utilises dans les recettes traditionnelles."
        ),
    },
]


@pytest.fixture
def raw_articles() -> list[dict]:
    """Copie du mini-corpus brut (copie : un test ne doit pas polluer l'autre)."""
    return [dict(article) for article in RAW_ARTICLES]


@pytest.fixture
def clean_articles(raw_articles) -> list[dict]:
    """Mini-corpus passe par le vrai pretraitement."""
    from src.preprocess import process_record

    return [process_record(article, "fr") for article in raw_articles]


@pytest.fixture
def chunks(clean_articles) -> list[dict]:
    """Chunks construits a partir du mini-corpus, sans toucher au disque."""
    from src.preprocess import normalize_tokens
    from src.rag import chunk_text

    result: list[dict] = []
    for article in clean_articles:
        pieces = chunk_text(article["text_clean"], size=400, overlap=80, lang="fr")
        for index, piece in enumerate(pieces):
            result.append(
                {
                    "id": f"{article['pageid']}_{index}",
                    "text": piece,
                    "norm_text": " ".join(normalize_tokens(piece, "fr")),
                    "title": article["title"],
                    "url": article["url"],
                    "lang": "fr",
                    "license": "CC BY-SA 4.0",
                    "chunk_index": index,
                    "n_chars": len(piece),
                }
            )
    return result


@pytest.fixture
def pipeline(chunks):
    """Pipeline RAG complet en mode hors ligne (TF-IDF + extractif)."""
    from src.rag import RagPipeline, TfidfRetriever

    return RagPipeline(retriever=TfidfRetriever(chunks), chunks=chunks)
