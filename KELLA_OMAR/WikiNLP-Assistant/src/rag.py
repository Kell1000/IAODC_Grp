"""
Etape 5 du TP - Systeme RAG (Retrieval-Augmented Generation).

LES TROIS CHOSES A NE PAS CONFONDRE
-----------------------------------
1. LLM SEUL
   On envoie la question au modele, sans aucun document. Le modele repond de
   memoire. Il peut etre brillant... ou inventer une reponse fausse avec
   assurance (hallucination), sans aucune source verifiable.

2. RECHERCHE DOCUMENTAIRE (TF-IDF ou embeddings)
   On renvoie les passages les plus proches de la question. Les sources sont
   exactes et verifiables, mais rien n'est redige : c'est a l'humain de lire
   les extraits et de faire la synthese.

3. RAG = 2 puis 1
   On cherche d'abord les passages (etape 2), puis on demande au LLM de
   rediger une reponse EN S'APPUYANT uniquement sur ces passages (etape 1).
   On obtient une reponse redigee ET des sources citables. Le LLM sert de
   moteur de reformulation, pas de base de connaissances.

Le pipeline implemente ici :
   documents -> chunks -> vecteurs -> index -> recherche -> prompt -> LLM -> reponse + sources

Utilisation
-----------
    python -m src.rag --build                     # construit l'index
    python -m src.rag --ask "Qu'est-ce que le NLP ?"
    python -m src.rag --compare "Qu'est-ce qu'un transformeur ?"
    python -m src.rag --info
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.collect_wikipedia import read_jsonl
from src.config import CFG, CHUNKS_FILE, CLEAN_ARTICLES, DEMO_CHUNKS, RAW_ARTICLES
from src.llm import LLMResult, generate
from src.preprocess import normalize_tokens, split_sentences

# Phrase renvoyee quand aucun passage n'atteint le score minimal. Elle est
# definie une seule fois ici pour que les tests puissent la verifier.
NO_ANSWER = (
    "Je ne trouve pas cette information dans les documents fournis. "
    "Le corpus Wikipedia collecte ne semble pas traiter ce sujet. "
    "Essayez de reformuler, ou collectez des articles sur ce theme."
)


# ==========================================================================
#  1. Decoupage en chunks
# ==========================================================================
def chunk_text(text: str, size: int, overlap: int, lang: str = "fr") -> list[str]:
    """Decoupe un texte en morceaux d'environ `size` caracteres.

    POURQUOI DECOUPER ?
      - Un article Wikipedia fait 20 000 caracteres : trop gros pour le
        contexte d'un petit LLM, et trop imprecis comme unite de recherche.
        Un embedding d'article entier "moyenne" tous ses sujets et ne
        ressemble plus a rien de precis.
      - Un chunk de ~900 caracteres parle d'UNE idee : son embedding est net.

    POURQUOI UN CHEVAUCHEMENT ?
      Sans chevauchement, une phrase cle peut etre coupee en deux et perdre
      son sens dans les deux morceaux. Les `overlap` derniers caracteres d'un
      chunk sont donc repris au debut du suivant.

    On coupe aux FRONTIERES DE PHRASES et non au milieu d'un mot : un chunk
    qui commence par "...ment du langage" est inexploitable.
    """
    # Un chunk est de la prose sur une seule ligne : on ecrase tous les blancs
    # (sauts de ligne compris). Sans cela, les sauts de ligne herites de la
    # mise en page de Wikipedia se retrouvent dans le passage affiche et dans
    # le prompt du LLM, sous la forme de longues suites d'espaces.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        # Meme un texte court passe le filtre de qualite : la regle doit etre
        # la meme partout, sinon un fragment de formule entre dans l'index
        # par ce raccourci.
        return [text] if is_informative(text) else []

    sentences = split_sentences(text, lang) or [text]

    chunks: list[str] = []
    current: list[str] = []   # on accumule des PHRASES, pas des caracteres
    current_len = 0

    def flush() -> list[str]:
        """Ferme le chunk courant et renvoie les phrases a reporter.

        Le chevauchement est pris PAR PHRASES ENTIERES, en partant de la fin.
        Un decoupage par caracteres (text[-overlap:]) semble equivalent mais
        produit des chunks commencant au milieu d'un mot, du genre
        "tegorisation de documents : ...". C'est illisible pour l'humain, et
        surtout l'embedding d'un fragment tronque represente mal son sens.
        On l'a observe en executant le TP : d'ou cette version.
        """
        if current:
            chunks.append(" ".join(current).strip())
        tail: list[str] = []
        tail_len = 0
        for sentence in reversed(current):
            if tail_len + len(sentence) + 1 > overlap:
                break
            tail.insert(0, sentence)
            tail_len += len(sentence) + 1
        return tail

    for sentence in sentences:
        # Phrase anormalement longue (tableau, liste sans ponctuation) :
        # on la coupe brutalement, sinon elle ferait exploser le chunk.
        if len(sentence) > size:
            flush()
            current, current_len = [], 0
            for start in range(0, len(sentence), size):
                piece = sentence[start : start + size].strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_len + len(sentence) + 1 <= size:
            current.append(sentence)
            current_len += len(sentence) + 1
        else:
            current = flush() if overlap else []
            current.append(sentence)
            current_len = sum(len(part) + 1 for part in current)

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if is_informative(chunk)]


# Un chunk doit contenir de la PROSE. Les articles scientifiques de Wikipedia
# comportent des sections de formules qui, une fois le balisage retire, ne
# laissent que des bribes : "a. x. f(x). i=1. n."
#
# Observe reellement dans l'interface pendant ce TP : le passage le mieux
# classe pour "A quoi sert l'apprentissage profond ?" etait
#     "[1] Titre : Reseau de neurones artificiels a. x."
# Ces fragments sont courts, donc TF-IDF les favorise (la normalisation par
# la longueur gonfle le poids de leurs rares mots), et ils n'apprennent rien
# au lecteur ni au LLM.
MIN_REAL_WORDS = 12        # nombre minimal de mots de 3 lettres ou plus
MIN_LETTER_RATIO = 0.55    # part minimale de lettres et d'espaces


def is_informative(chunk: str) -> bool:
    """Le chunk contient-il assez de prose pour etre utile ?"""
    if len(chunk) <= 50:
        return False

    words = [word for word in re.findall(r"[^\W\d_]+", chunk) if len(word) >= 3]
    if len(words) < MIN_REAL_WORDS:
        return False

    # Un pave de symboles mathematiques a une faible proportion de lettres.
    letters = sum(1 for char in chunk if char.isalpha() or char.isspace())
    return letters / len(chunk) >= MIN_LETTER_RATIO


def build_chunks(
    in_path: Path = CLEAN_ARTICLES,
    out_path: Path = CHUNKS_FILE,
    size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Transforme les articles pretraites en chunks prets pour l'index."""
    size = size or CFG.chunk_size
    overlap = overlap or CFG.chunk_overlap

    articles = read_jsonl(in_path)
    if not articles:
        raise SystemExit(
            f"[X] Aucun article pretraite dans {in_path}.\n"
            "    Lancez d'abord :\n"
            "      python -m src.collect_wikipedia\n"
            "      python -m src.preprocess"
        )

    chunks: list[dict] = []
    for article in articles:
        pieces = chunk_text(
            article.get("text_clean", ""), size, overlap, article.get("lang", "fr")
        )
        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{article.get('pageid', 'x')}_{index}",
                    "text": piece,
                    # Texte normalise, calcule UNE fois ici : le retriever
                    # TF-IDF le reutilise sans refaire le travail.
                    "norm_text": " ".join(
                        normalize_tokens(piece, article.get("lang", "fr"))
                    ),
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "lang": article.get("lang", "fr"),
                    "license": article.get("license", "CC BY-SA 4.0"),
                    "chunk_index": index,
                    "n_chars": len(piece),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [chunk["n_chars"] for chunk in chunks]
    print(f"  {len(articles)} article(s) -> {len(chunks)} chunk(s)")
    if lengths:
        print(f"  Taille des chunks : min {min(lengths)} / "
              f"moyenne {sum(lengths) // len(lengths)} / max {max(lengths)} car.")
    print(f"  Sortie : {out_path}")
    return chunks


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    """Charge les chunks, avec repli sur le mini-corpus de demonstration.

    Le repli est ce qui permet a l'application deployee de fonctionner meme
    si l'utilisateur n'a jamais lance la collecte (mode demonstration).
    """
    chunks = read_jsonl(path)
    if chunks:
        return chunks
    if DEMO_CHUNKS.exists():
        demo = read_jsonl(DEMO_CHUNKS)
        if demo:
            print(f"  [i] Corpus principal absent : utilisation du mini-corpus "
                  f"de demonstration ({len(demo)} chunks).")
            return demo
    return []


# ==========================================================================
#  2. Les deux moteurs de recherche
# ==========================================================================
@dataclass
class Retrieved:
    """Un passage retrouve, avec son score et ses metadonnees de source."""

    rank: int
    score: float
    text: str
    title: str
    url: str
    chunk_id: str


class TfidfRetriever:
    """Recherche LEXICALE : compte les mots, ignore le sens.

    Avantages  : instantane, aucun modele a telecharger, explicable
                 (on peut lister les mots qui ont provoque le score).
    Limites    : ne comprend pas les synonymes. "Qui a cree les reseaux de
                 neurones ?" ne trouvera pas un passage qui dit
                 "invente par McCulloch et Pitts" si le mot "cree" est absent.
    """

    kind = "tfidf"

    def __init__(self, chunks: list[dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.lang = chunks[0].get("lang", "fr") if chunks else "fr"

        # On indexe le texte DEJA normalise (minuscules, sans stop words,
        # racinise). La requete subira exactement le meme traitement :
        # c'est la condition pour que les mots se correspondent.
        corpus = [
            chunk.get("norm_text") or " ".join(normalize_tokens(chunk["text"], self.lang))
            for chunk in chunks
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # mots seuls + paires : capte "reseau neurone"
            min_df=1,
            sublinear_tf=True,    # log(tf) : un mot repete 10x ne vaut pas 10x
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, question: str, top_k: int = 4) -> list[Retrieved]:
        from sklearn.metrics.pairwise import linear_kernel

        query = " ".join(normalize_tokens(question, self.lang))
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        # Les lignes TF-IDF de scikit-learn sont normalisees L2 : le produit
        # scalaire (linear_kernel) est donc deja la similarite cosinus.
        scores = linear_kernel(query_vector, self.matrix).ravel()

        top_k = min(top_k, scores.shape[0])
        best = np.argpartition(-scores, top_k - 1)[:top_k]
        best = best[np.argsort(-scores[best])]

        return [
            Retrieved(
                rank=rank,
                score=float(scores[int(index)]),
                text=self.chunks[int(index)]["text"],
                title=self.chunks[int(index)]["title"],
                url=self.chunks[int(index)]["url"],
                chunk_id=self.chunks[int(index)]["id"],
            )
            for rank, index in enumerate(best, start=1)
        ]


class EmbeddingRetriever:
    """Recherche SEMANTIQUE : compare le sens via des vecteurs denses.

    Avantages  : trouve les synonymes et les reformulations.
    Limites    : telecharge un modele (~470 Mo), quelques ms par requete,
                 et peut rater une correspondance exacte de terme technique
                 rare que TF-IDF aurait trouvee.
    """

    kind = "embeddings"

    def __init__(self, chunks: list[dict], rebuild: bool = False,
                 warm: bool = True) -> None:
        from src.embeddings import SentenceEmbedder, embed_corpus
        from src.vector_store import get_vector_store

        self.chunks = chunks
        self.embedder = SentenceEmbedder()
        self.store = get_vector_store()

        # Reutilise l'index sur disque si sa taille correspond au corpus.
        if not rebuild and self.store.load() and self.store.count() == len(chunks):
            print(f"  Index vectoriel charge depuis le disque "
                  f"({self.store.count()} vecteurs).")
            # Le modele n'a pas ete touche : on le charge maintenant, pendant
            # l'ecran de chargement, et non a la premiere question (voir
            # warmup()). Le cache d'embeddings peut aussi avoir evite tout
            # appel au modele lors de la construction : d'ou le warmup final.
            if warm:
                elapsed = self.warmup()
                print(f"  Modele d'embeddings prechauffe en {elapsed:.1f}s.")
            return

        print("  Construction de l'index vectoriel...")
        texts = [chunk["text"] for chunk in chunks]
        vectors = embed_corpus(texts)

        if hasattr(self.store, "reset"):
            self.store.reset()
            self.store = get_vector_store()

        self.store.add(
            ids=[chunk["id"] for chunk in chunks],
            vectors=vectors,
            documents=texts,
            metadatas=[
                {
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in chunks
            ],
        )
        self.store.save()
        print(f"  Index vectoriel enregistre ({self.store.count()} vecteurs).")

        # embed_corpus() a pu servir le corpus depuis son cache disque sans
        # jamais charger le modele : on prechauffe donc ici aussi.
        if warm:
            elapsed = self.warmup()
            print(f"  Modele d'embeddings prechauffe en {elapsed:.1f}s.")

    def warmup(self) -> float:
        """Charge le modele MAINTENANT et renvoie le temps que cela a pris.

        POURQUOI CETTE METHODE EXISTE
        -----------------------------
        `SentenceEmbedder` charge son modele paresseusement, au premier
        `encode()`. Quand l'index vectoriel est relu depuis le disque, on ne
        touche donc au modele qu'a la PREMIERE QUESTION.

        Mesure reelle sur Windows (annexe C) : cette premiere question a pris
        **94,6 s**, contre **0,19 s** pour les suivantes, soit un facteur 500.
        Dans l'interface web, l'utilisateur voyait l'index se charger en
        quelques secondes, puis sa premiere question restait bloquee une
        minute et demie sans explication.

        En appelant warmup() pendant l'ecran de chargement, on paie ce cout
        la ou l'utilisateur l'attend, et toutes les questions deviennent
        rapides. Cela ne reduit pas le temps total : cela le deplace au bon
        endroit.
        """
        start = time.perf_counter()
        self.embedder.encode_one("initialisation du modele")
        return time.perf_counter() - start

    def search(self, question: str, top_k: int = 4) -> list[Retrieved]:
        query_vector = self.embedder.encode_one(question)
        hits = self.store.search(query_vector, top_k=top_k)
        return [
            Retrieved(
                rank=hit["rank"],
                score=hit["score"],
                text=hit["document"],
                title=hit["metadata"].get("title", ""),
                url=hit["metadata"].get("url", ""),
                chunk_id=hit["id"],
            )
            for hit in hits
        ]


def get_retriever(kind: str | None = None, chunks: list[dict] | None = None,
                  rebuild: bool = False):
    """Fabrique le moteur de recherche demande, avec repli automatique.

    Si RETRIEVER=embeddings mais que sentence-transformers est absent, on
    retombe sur TF-IDF plutot que de faire planter l'application : mieux vaut
    une recherche lexicale qui fonctionne qu'un ecran d'erreur.
    """
    kind = (kind or CFG.retriever).lower()
    chunks = chunks if chunks is not None else load_chunks()
    if not chunks:
        raise SystemExit(
            "[X] Aucun chunk disponible.\n"
            "    Lancez le pipeline complet :\n"
            "      python -m src.collect_wikipedia\n"
            "      python -m src.preprocess\n"
            "      python -m src.rag --build"
        )

    if kind == "tfidf":
        return TfidfRetriever(chunks)
    if kind == "embeddings":
        try:
            return EmbeddingRetriever(chunks, rebuild=rebuild)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [!] Recherche semantique indisponible ({type(exc).__name__}: {exc})\n"
                "      Repli automatique sur la recherche lexicale TF-IDF.",
                file=sys.stderr,
            )
            return TfidfRetriever(chunks)
    raise ValueError(f"RETRIEVER inconnu : {kind} (attendu : tfidf | embeddings)")


# ==========================================================================
#  3. Le pipeline RAG
# ==========================================================================
@dataclass
class RagAnswer:
    """Reponse complete : texte, sources, mesures. Tout ce qu'affiche l'UI."""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    passages: list[Retrieved] = field(default_factory=list)
    has_context: bool = True
    best_score: float = 0.0
    retriever: str = ""
    provider: str = ""
    model: str = ""
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0
    total_time_s: float = 0.0
    llm_ok: bool = True
    llm_error: str = ""

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["passages"] = [passage.__dict__ for passage in self.passages]
        return data


class RagPipeline:
    """Assemble recherche + generation, avec garde-fou anti-hallucination."""

    def __init__(self, retriever=None, chunks: list[dict] | None = None,
                 rebuild: bool = False) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        self.retriever = retriever or get_retriever(chunks=self.chunks, rebuild=rebuild)

    # ------------------------------------------------------------------
    def build_context(self, passages: list[Retrieved]) -> str:
        """Met en forme les passages pour le prompt.

        La numerotation [1], [2]... est essentielle : elle permet au LLM de
        citer ses sources, et a l'utilisateur de verifier chaque affirmation.
        """
        blocks = []
        for passage in passages:
            blocks.append(
                f"[{passage.rank}] Titre : {passage.title}\n{passage.text}"
            )
        return "\n\n".join(blocks)

    def sources_from(self, passages: list[Retrieved]) -> list[dict]:
        """Liste des sources uniques (un article peut fournir 2 passages)."""
        seen: dict[str, dict] = {}
        for passage in passages:
            if passage.url not in seen:
                seen[passage.url] = {
                    "n": len(seen) + 1,
                    "title": passage.title,
                    "url": passage.url,
                    "score": round(passage.score, 4),
                    "license": "CC BY-SA 4.0",
                }
        return list(seen.values())

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        provider: str | None = None,
    ) -> RagAnswer:
        """Repond a une question : recherche, garde-fou, puis generation."""
        top_k = top_k or CFG.top_k
        # Le seuil depend du moteur REELLEMENT utilise (l'interface permet
        # d'en changer a chaud), pas de celui ecrit dans .env.
        if min_score is None:
            min_score = CFG.min_score_for(self.retriever.kind)
        started = time.perf_counter()

        question = (question or "").strip()
        if not question:
            return RagAnswer(
                question="", answer="Posez une question.", has_context=False,
                retriever=self.retriever.kind,
            )

        # --- Etape 1 : recherche ---
        retrieval_start = time.perf_counter()
        passages = self.retriever.search(question, top_k=top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        best_score = passages[0].score if passages else 0.0

        # --- Etape 2 : GARDE-FOU ---
        # Si le meilleur passage est trop faible, la question est hors corpus.
        # On refuse AVANT d'appeler le LLM. Double benefice : on ne consomme
        # pas de quota, et on supprime la principale source d'hallucination
        # (un LLM a qui on donne un contexte hors sujet brode quand meme).
        if not passages or best_score < min_score:
            return RagAnswer(
                question=question,
                answer=NO_ANSWER,
                sources=[],
                passages=passages,
                has_context=False,
                best_score=best_score,
                retriever=self.retriever.kind,
                provider="aucun (garde-fou declenche)",
                retrieval_time_s=retrieval_time,
                total_time_s=time.perf_counter() - started,
            )

        # --- Etape 3 : generation ---
        context = self.build_context(passages)
        result: LLMResult = generate(question, context, provider=provider)

        return RagAnswer(
            question=question,
            answer=result.text,
            sources=self.sources_from(passages),
            passages=passages,
            has_context=True,
            best_score=best_score,
            retriever=self.retriever.kind,
            provider=result.provider,
            model=result.model,
            retrieval_time_s=retrieval_time,
            generation_time_s=result.latency_s,
            total_time_s=time.perf_counter() - started,
            llm_ok=result.ok,
            llm_error=result.error,
        )


# ==========================================================================
#  4. Comparaison des trois approches (demande de l'etape 5 du TP)
# ==========================================================================
def compare_approaches(question: str, top_k: int = 3) -> None:
    """Montre cote a cote : TF-IDF, recherche semantique, puis RAG complet."""
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("[X] Aucun chunk. Lancez : python -m src.rag --build")

    print("=" * 74)
    print(f"QUESTION : {question}")
    print("=" * 74)

    # --- 1. TF-IDF ---
    print("\n--- 1. RECHERCHE LEXICALE (TF-IDF) ---")
    print("    Ce que ca fait : compte les mots communs. Aucune redaction.")
    start = time.perf_counter()
    tfidf = TfidfRetriever(chunks)
    tfidf_hits = tfidf.search(question, top_k=top_k)
    tfidf_time = time.perf_counter() - start
    for hit in tfidf_hits:
        print(f"    #{hit.rank} [{hit.score:.4f}] {hit.title}")
        print(f"        {hit.text[:150].replace(chr(10), ' ')}...")
    print(f"    Temps (index + recherche) : {tfidf_time:.3f}s")

    # --- 2. Semantique ---
    print("\n--- 2. RECHERCHE SEMANTIQUE (embeddings) ---")
    print("    Ce que ca fait : compare le sens. Toujours aucune redaction.")
    try:
        start = time.perf_counter()
        semantic = EmbeddingRetriever(chunks)
        semantic_hits = semantic.search(question, top_k=top_k)
        semantic_time = time.perf_counter() - start
        for hit in semantic_hits:
            print(f"    #{hit.rank} [{hit.score:.4f}] {hit.title}")
            print(f"        {hit.text[:150].replace(chr(10), ' ')}...")
        print(f"    Temps (index + recherche) : {semantic_time:.3f}s")

        common = {hit.chunk_id for hit in tfidf_hits} & {h.chunk_id for h in semantic_hits}
        print(f"\n    Passages communs aux deux methodes : {len(common)}/{top_k}")
        if len(common) < top_k:
            print("    -> Les deux methodes ne trouvent PAS la meme chose : "
                  "c'est exactement l'interet de les comparer.")
    except Exception as exc:  # noqa: BLE001
        print(f"    [!] Indisponible : {exc}")

    # --- 3. RAG ---
    print("\n--- 3. RAG COMPLET (recherche + generation) ---")
    print("    Ce que ca fait : redige une reponse a partir des passages,")
    print("    et cite ses sources.")
    pipeline = RagPipeline(chunks=chunks)
    answer = pipeline.answer(question, top_k=top_k)
    print(f"\n    Reponse ({answer.provider}) :\n")
    for line in answer.answer.split("\n"):
        print(f"      {line}")
    print(f"\n    Sources citees :")
    for source in answer.sources:
        print(f"      [{source['n']}] {source['title']} - {source['url']}")
    if not answer.sources:
        print("      (aucune : garde-fou declenche)")
    print(f"\n    Meilleur score      : {answer.best_score:.4f} "
          f"(seuil : {CFG.min_score_for(pipeline.retriever.kind)})")
    print(f"    Temps recherche     : {answer.retrieval_time_s:.3f}s")
    print(f"    Temps generation    : {answer.generation_time_s:.3f}s")
    print(f"    Temps total         : {answer.total_time_s:.3f}s")

    print("\n" + "=" * 74)
    print("A RETENIR : 1 et 2 donnent des EXTRAITS. 3 donne une REPONSE")
    print("REDIGEE + des SOURCES. Le contenu vient toujours de la recherche :")
    print("le LLM ne fait que reformuler ce qu'on lui a donne.")
    print("=" * 74)


def index_info() -> None:
    """Affiche l'etat du pipeline : que manque-t-il pour que ca marche ?"""
    print("=== Etat du pipeline RAG ===\n")
    steps = [
        ("1. Articles bruts", RAW_ARTICLES),
        ("2. Articles pretraites", CLEAN_ARTICLES),
        ("3. Chunks", CHUNKS_FILE),
    ]
    for label, path in steps:
        path = Path(path)
        if path.exists():
            count = len(read_jsonl(path))
            size = path.stat().st_size / 1024
            print(f"  [OK] {label:24s} {count:5d} ligne(s)  {size:8.1f} Ko")
        else:
            print(f"  [--] {label:24s} absent ({path.name})")

    print(f"\n  Retriever configure : {CFG.retriever}")
    print(f"  Backend vectoriel   : {CFG.vector_backend}")
    print(f"  Modele d'embeddings : {CFG.embedding_model}")
    print(f"  Fournisseur LLM     : {CFG.llm_provider}")
    ready, message = CFG.llm_is_ready()
    print(f"  LLM pret            : {ready} ({message})")
    print(f"  top_k               : {CFG.top_k}")
    print(f"  min_score tfidf     : {CFG.min_score_for('tfidf')}")
    print(f"  min_score embeddings: {CFG.min_score_for('embeddings')}")


# ==========================================================================
#  CLI
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Systeme RAG WikiNLP.")
    parser.add_argument("--build", action="store_true",
                        help="Construit les chunks puis l'index vectoriel.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force la reconstruction de l'index vectoriel.")
    parser.add_argument("--ask", metavar="QUESTION", help="Pose une question.")
    parser.add_argument("--compare", metavar="QUESTION",
                        help="Compare TF-IDF, semantique et RAG.")
    parser.add_argument("--info", action="store_true",
                        help="Affiche l'etat du pipeline.")
    parser.add_argument("--top-k", type=int, default=CFG.top_k)
    args = parser.parse_args(argv)

    if args.info:
        index_info()
        return 0

    if args.build or args.rebuild:
        print("=== Construction de l'index RAG ===")
        chunks = build_chunks()
        if CFG.retriever == "embeddings":
            try:
                EmbeddingRetriever(chunks, rebuild=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [!] Index vectoriel non construit : {exc}", file=sys.stderr)
                print("      La recherche TF-IDF reste disponible.")
        else:
            print("  RETRIEVER=tfidf : aucun index vectoriel a construire "
                  "(TF-IDF s'indexe au demarrage, en moins d'une seconde).")
        print("\n  [OK] Index pret. Testez :  python -m src.rag --ask \"votre question\"")
        return 0

    if args.compare:
        compare_approaches(args.compare, top_k=args.top_k)
        return 0

    if args.ask:
        pipeline = RagPipeline()
        answer = pipeline.answer(args.ask, top_k=args.top_k)
        print(f"\nQuestion : {answer.question}")
        print(f"\nReponse ({answer.provider} / {answer.model}) :\n{answer.answer}\n")
        if answer.sources:
            print("Sources :")
            for source in answer.sources:
                print(f"  [{source['n']}] {source['title']}")
                print(f"      {source['url']}  (licence {source['license']})")
        print(f"\nMeilleur score : {answer.best_score:.4f} | "
              f"recherche {answer.retrieval_time_s:.3f}s | "
              f"generation {answer.generation_time_s:.3f}s | "
              f"total {answer.total_time_s:.3f}s")
        if not answer.llm_ok:
            print(f"\n[!] Le LLM a echoue, repli extractif :\n{answer.llm_error}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
