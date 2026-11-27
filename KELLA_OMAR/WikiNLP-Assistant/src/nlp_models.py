"""
Etape 4 du TP - Experiences NLP comparables.

Trois experiences, a lancer dans cet ordre :

  EXPERIENCE A - NLP classique (scikit-learn, aucun modele a telecharger)
      frequences de mots, TF-IDF, similarite cosinus, recherche lexicale.

  EXPERIENCE B - Modeles preentraines (sentence-transformers)
      embeddings de phrases, recherche semantique, et COMPARAISON directe
      avec la recherche TF-IDF de l'experience A.

  EXPERIENCE B2 - Classification de texte (optionnelle, telechargement)
      analyse de sentiment avec un petit modele preentraine.
      Optionnelle expres : elle telecharge ~270 Mo de plus. On ne la lance
      que si on en a besoin, pour ne pas saturer le disque.

Utilisation
-----------
    python -m src.nlp_models --exp-a
    python -m src.nlp_models --exp-b
    python -m src.nlp_models --sentiment      # telecharge un modele en plus
    python -m src.nlp_models --all
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import numpy as np

from src.collect_wikipedia import read_jsonl
from src.config import CFG, CLEAN_ARTICLES
from src.preprocess import normalize_tokens, tokenize

# Questions de test reutilisees dans toutes les experiences : comparer n'a de
# sens que si l'on pose exactement les memes questions a chaque methode.
TEST_QUERIES = [
    "Qu'est-ce que le traitement automatique des langues ?",
    "Comment fonctionne un reseau de neurones ?",
    "Qui a invente l'architecture Transformer ?",
    "A quoi sert l'apprentissage profond ?",
]


def load_corpus() -> list[dict]:
    """Charge le corpus pretraite, avec un message clair s'il est absent."""
    records = read_jsonl(CLEAN_ARTICLES)
    if not records:
        raise SystemExit(
            f"[X] Corpus pretraite absent : {CLEAN_ARTICLES}\n"
            "    Lancez d'abord :\n"
            "      python -m src.collect_wikipedia\n"
            "      python -m src.preprocess"
        )
    return records


# ==========================================================================
#  EXPERIENCE A : NLP classique
# ==========================================================================
def experiment_a(top_n: int = 15) -> dict:
    """Tokenisation, frequences, TF-IDF, similarite cosinus, recherche."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print("=" * 74)
    print("EXPERIENCE A - NLP CLASSIQUE")
    print("=" * 74)

    corpus = load_corpus()
    titles = [record["title"] for record in corpus]
    print(f"\n  Corpus : {len(corpus)} documents\n")

    # ---------------------------------------------------------------
    # A.1 Frequences de mots : brutes contre normalisees
    # ---------------------------------------------------------------
    print("-" * 74)
    print("A.1 FREQUENCES DE MOTS")
    print("-" * 74)

    all_text = " ".join(record["text_clean"] for record in corpus)
    raw_tokens = [token.lower() for token in tokenize(all_text)]
    norm_tokens = [token for record in corpus for token in record["tokens_norm"]]

    raw_top = Counter(raw_tokens).most_common(top_n)
    norm_top = Counter(norm_tokens).most_common(top_n)

    print(f"\n  Tokens bruts : {len(raw_tokens):,}".replace(",", " ")
          + f" | vocabulaire : {len(set(raw_tokens)):,}".replace(",", " "))
    print(f"  Tokens normalises : {len(norm_tokens):,}".replace(",", " ")
          + f" | vocabulaire : {len(set(norm_tokens)):,}".replace(",", " "))

    print(f"\n  {'SANS pretraitement':<32} | {'AVEC pretraitement':<32}")
    print(f"  {'-' * 32} | {'-' * 32}")
    for index in range(top_n):
        left = f"{raw_top[index][0]:<20} {raw_top[index][1]:>6}" if index < len(raw_top) else ""
        right = f"{norm_top[index][0]:<20} {norm_top[index][1]:>6}" if index < len(norm_top) else ""
        print(f"  {left:<32} | {right:<32}")

    print("\n  LECON : sans pretraitement, le classement est domine par les mots")
    print("  outils (de, la, le, et...). Ils sont les plus frequents de la langue")
    print("  et ne disent RIEN du sujet. C'est pour cela que TF-IDF les penalise")
    print("  et que l'on supprime les stop words pour la recherche lexicale.")

    # ---------------------------------------------------------------
    # A.2 TF-IDF : les termes qui caracterisent chaque document
    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("A.2 TF-IDF : LES TERMES CARACTERISTIQUES")
    print("-" * 74)
    print("\n  TF-IDF = frequence du terme dans le document (TF) ponderee par sa")
    print("  rarete dans le corpus (IDF). Un mot present partout a un IDF faible :")
    print("  il ne distingue rien. Un mot frequent ICI mais rare AILLEURS obtient")
    print("  un score eleve : c'est un bon descripteur du document.\n")

    documents_norm = [record["norm_text"] for record in corpus]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents_norm)
    vocabulary = np.array(vectorizer.get_feature_names_out())

    print(f"  Matrice TF-IDF : {matrix.shape[0]} documents x "
          f"{matrix.shape[1]} termes")
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
    print(f"  Remplissage : {density:.3f} % (une matrice TF-IDF est tres creuse,")
    print(f"  d'ou le stockage 'sparse' de scipy : {matrix.data.nbytes / 1024:.0f} Ko")
    print(f"  au lieu de {matrix.shape[0] * matrix.shape[1] * 8 / 1024 / 1024:.1f} Mo en dense)\n")

    for index in range(min(3, len(corpus))):
        row = matrix[index].toarray().ravel()
        best = row.argsort()[-6:][::-1]
        terms = ", ".join(f"{vocabulary[j]} ({row[j]:.3f})" for j in best if row[j] > 0)
        print(f"  {titles[index][:40]:<42} -> {terms}")

    # ---------------------------------------------------------------
    # A.3 Similarite cosinus entre documents
    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("A.3 SIMILARITE COSINUS ENTRE DOCUMENTS")
    print("-" * 74)
    print("\n  Le cosinus mesure l'angle entre deux vecteurs, pas leur longueur :")
    print("  un article long et un court sur le meme sujet restent similaires.\n")

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0)  # un document est trivialement similaire a lui-meme

    pairs = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            pairs.append((similarity[i, j], titles[i], titles[j]))
    pairs.sort(reverse=True)

    print("  Les 5 paires de documents les plus proches :")
    for score, title_a, title_b in pairs[:5]:
        print(f"    {score:.4f}  {title_a[:30]:<32} <-> {title_b[:30]}")

    # ---------------------------------------------------------------
    # A.4 Recherche lexicale
    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("A.4 RECHERCHE DES DOCUMENTS PERTINENTS")
    print("-" * 74)

    timings = []
    for query in TEST_QUERIES:
        normalized = " ".join(normalize_tokens(query, CFG.wiki_lang))
        start = time.perf_counter()
        scores = cosine_similarity(vectorizer.transform([normalized]), matrix).ravel()
        timings.append(time.perf_counter() - start)
        best = scores.argsort()[-3:][::-1]

        print(f"\n  Question : {query}")
        print(f"  Requete normalisee : '{normalized}'")
        for rank, index in enumerate(best, start=1):
            flag = "" if scores[index] > 0.05 else "   <- score tres faible"
            print(f"    {rank}. [{scores[index]:.4f}] {titles[index]}{flag}")

    print(f"\n  Temps moyen par recherche : {np.mean(timings) * 1000:.2f} ms")
    print("\n  AVANTAGES de TF-IDF : instantane, explicable, aucun modele a")
    print("  telecharger, excellent sur les termes techniques exacts.")
    print("  LIMITES : aveugle aux synonymes. Si la question dit 'cree' et le")
    print("  document 'invente', le score peut tomber a zero.")

    return {
        "n_docs": len(corpus),
        "vocab_raw": len(set(raw_tokens)),
        "vocab_norm": len(set(norm_tokens)),
        "tfidf_shape": matrix.shape,
        "mean_search_ms": float(np.mean(timings) * 1000),
    }


# ==========================================================================
#  EXPERIENCE B : modeles preentraines
# ==========================================================================
def experiment_b(top_k: int = 3) -> dict:
    """Embeddings de phrases, puis comparaison lexical / semantique."""
    print("\n" + "=" * 74)
    print("EXPERIENCE B - MODELES PREENTRAINES (EMBEDDINGS)")
    print("=" * 74)

    try:
        from src.embeddings import (
            EmbeddingBackendUnavailable,
            SentenceEmbedder,
            cosine_similarity_matrix,
        )
    except ImportError as exc:
        print(f"  [X] Import impossible : {exc}")
        return {}

    # ---------------------------------------------------------------
    # B.1 Les embeddings captent le sens, pas les mots
    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("B.1 LE SENS CONTRE LES MOTS")
    print("-" * 74)

    pairs = [
        ("Le chat dort sur le canape.", "Un felin se repose sur le sofa."),
        ("L'apprentissage profond utilise des reseaux de neurones.",
         "Le deep learning repose sur des reseaux neuronaux."),
        ("Paris est la capitale de la France.", "La tarte aux pommes est delicieuse."),
    ]

    try:
        embedder = SentenceEmbedder()
        start = time.perf_counter()
        print(f"\n  Modele : {embedder.model_name}")
        print(f"  Dimensions : {embedder.dim}")
        load_time = time.perf_counter() - start
    except EmbeddingBackendUnavailable as exc:
        print(f"\n  [X] {exc}")
        return {}

    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"\n  {'Paire de phrases':<46} {'TF-IDF':>8} {'Embedding':>11}")
    print(f"  {'-' * 46} {'-' * 8} {'-' * 11}")
    for sentence_a, sentence_b in pairs:
        # TF-IDF sur deux phrases seules : mesure le recouvrement de mots.
        local_tfidf = TfidfVectorizer().fit_transform(
            [" ".join(normalize_tokens(sentence_a)), " ".join(normalize_tokens(sentence_b))]
        )
        lexical = float((local_tfidf[0] @ local_tfidf[1].T).toarray()[0][0])

        vectors = embedder.encode([sentence_a, sentence_b])
        semantic = float(cosine_similarity_matrix(vectors[0], vectors[1])[0][0])

        print(f"  {sentence_a[:44]:<46} {lexical:>8.3f} {semantic:>11.3f}")
        print(f"  {sentence_b[:44]:<46}")
        print()

    print("  LECON : sur les deux premieres paires (meme sens, mots differents),")
    print("  TF-IDF donne un score bas et l'embedding un score eleve. Sur la")
    print("  troisieme (sens different), les deux sont bas. L'embedding mesure")
    print("  le SENS ; TF-IDF mesure le PARTAGE DE MOTS.")

    # ---------------------------------------------------------------
    # B.2 Comparaison des deux moteurs sur le vrai corpus
    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("B.2 TF-IDF CONTRE RECHERCHE SEMANTIQUE SUR LE CORPUS")
    print("-" * 74)

    from src.rag import EmbeddingRetriever, TfidfRetriever, load_chunks

    chunks = load_chunks()
    if not chunks:
        print("\n  [X] Aucun chunk. Lancez : python -m src.rag --build")
        return {"model": embedder.model_name}

    print(f"\n  Corpus de recherche : {len(chunks)} chunks")

    start = time.perf_counter()
    lexical_retriever = TfidfRetriever(chunks)
    tfidf_build = time.perf_counter() - start

    start = time.perf_counter()
    semantic_retriever = EmbeddingRetriever(chunks)
    semantic_build = time.perf_counter() - start

    print(f"  Construction index TF-IDF    : {tfidf_build:.2f}s")
    print(f"  Construction index semantique : {semantic_build:.2f}s")

    overlaps, lexical_times, semantic_times = [], [], []

    for query in TEST_QUERIES:
        start = time.perf_counter()
        lexical_hits = lexical_retriever.search(query, top_k=top_k)
        lexical_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        semantic_hits = semantic_retriever.search(query, top_k=top_k)
        semantic_times.append(time.perf_counter() - start)

        common = {hit.chunk_id for hit in lexical_hits} & {
            hit.chunk_id for hit in semantic_hits
        }
        overlaps.append(len(common))

        print(f"\n  Question : {query}")
        print(f"    {'TF-IDF':<36} | {'SEMANTIQUE':<36}")
        for rank in range(top_k):
            left = (
                f"[{lexical_hits[rank].score:.3f}] {lexical_hits[rank].title[:26]}"
                if rank < len(lexical_hits) else ""
            )
            right = (
                f"[{semantic_hits[rank].score:.3f}] {semantic_hits[rank].title[:26]}"
                if rank < len(semantic_hits) else ""
            )
            print(f"    {left:<36} | {right:<36}")
        print(f"    Passages identiques : {len(common)}/{top_k}")

    print("\n" + "-" * 74)
    print("  SYNTHESE")
    print("-" * 74)
    print(f"  Recouvrement moyen des resultats : "
          f"{np.mean(overlaps):.2f}/{top_k} passages")
    print(f"  Temps moyen TF-IDF     : {np.mean(lexical_times) * 1000:.2f} ms")
    print(f"  Temps moyen semantique : {np.mean(semantic_times) * 1000:.2f} ms")
    print("\n  Note : le temps semantique inclut l'encodage de la question par")
    print("  le modele. TF-IDF reste plus rapide ; la question est de savoir si")
    print("  la qualite supplementaire justifie le cout. A VOUS de juger sur VOS")
    print("  questions : relevez ces chiffres, ils dependent de votre machine.")

    return {
        "model": embedder.model_name,
        "dim": embedder.dim,
        "n_chunks": len(chunks),
        "mean_overlap": float(np.mean(overlaps)),
        "tfidf_ms": float(np.mean(lexical_times) * 1000),
        "semantic_ms": float(np.mean(semantic_times) * 1000),
    }


# ==========================================================================
#  EXPERIENCE B2 : classification de texte (optionnelle)
# ==========================================================================
# Modele par defaut : petit, tres connu, mais ANGLAIS uniquement (~270 Mo).
# Pour du francais : "cmarkea/distilcamembert-base-sentiment" (~270 Mo)
# ou "nlptown/bert-base-multilingual-uncased-sentiment" (~670 Mo).
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def experiment_sentiment(model_name: str = SENTIMENT_MODEL) -> dict:
    """Analyse de sentiment avec un petit modele preentraine."""
    print("\n" + "=" * 74)
    print("EXPERIENCE B2 - CLASSIFICATION DE TEXTE (SENTIMENT)")
    print("=" * 74)
    print(f"\n  [!] Cette experience telecharge un modele (~270 Mo) : {model_name}")

    try:
        from transformers import pipeline
    except ImportError:
        print("\n  [X] 'transformers' n'est pas installe.")
        print("      python -m pip install transformers")
        return {}

    try:
        classifier = pipeline(
            "sentiment-analysis", model=model_name, device=-1  # -1 = CPU
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [X] Chargement impossible : {type(exc).__name__}: {exc}")
        return {}

    samples = [
        "This library is wonderful and very well documented.",
        "The installation failed and the error messages are useless.",
        "Natural language processing is a field of computer science.",
    ]

    print(f"\n  {'Texte':<52} {'Label':<10} {'Score':>7}")
    print(f"  {'-' * 52} {'-' * 10} {'-' * 7}")
    results = []
    for text in samples:
        start = time.perf_counter()
        prediction = classifier(text[:512])[0]
        elapsed = time.perf_counter() - start
        results.append({"text": text, **prediction, "time_s": elapsed})
        print(f"  {text[:50]:<52} {prediction['label']:<10} "
              f"{prediction['score']:>7.3f}")

    print("\n  LECON IMPORTANTE : la troisieme phrase est NEUTRE et factuelle,")
    print("  mais le modele doit choisir entre POSITIVE et NEGATIVE : il n'a pas")
    print("  de classe 'neutre'. Un modele ne peut predire que les classes sur")
    print("  lesquelles il a ete entraine. Lisez toujours la fiche du modele")
    print("  avant de lui faire confiance.")
    print("\n  Note : ce modele est ANGLOPHONE. Sur du francais, ses scores ne")
    print("  sont pas fiables. Pour du francais, voir les modeles suggeres dans")
    print("  le code (distilcamembert, bert multilingual).")

    return {"model": model_name, "results": results}


# ==========================================================================
#  CLI
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiences NLP du TP.")
    parser.add_argument("--exp-a", action="store_true", help="NLP classique.")
    parser.add_argument("--exp-b", action="store_true", help="Embeddings.")
    parser.add_argument("--sentiment", action="store_true",
                        help="Classification (telecharge ~270 Mo).")
    parser.add_argument("--all", action="store_true",
                        help="Experiences A et B (sans la classification).")
    args = parser.parse_args(argv)

    if not any([args.exp_a, args.exp_b, args.sentiment, args.all]):
        parser.print_help()
        return 0

    if args.exp_a or args.all:
        experiment_a()
    if args.exp_b or args.all:
        experiment_b()
    if args.sentiment:
        experiment_sentiment()

    print("\n[OK] Experiences terminees. Relevez les chiffres affiches : ils")
    print("     dependent de VOTRE corpus et de VOTRE machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
