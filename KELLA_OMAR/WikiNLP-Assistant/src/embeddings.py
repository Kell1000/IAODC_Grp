"""
Etape 4/5 du TP - Embeddings de phrases (recherche semantique).

Un embedding est un vecteur de nombres qui represente le SENS d'un texte.
Deux textes de sens proche ont des vecteurs proches, meme s'ils ne partagent
aucun mot : c'est toute la difference avec TF-IDF.

    TF-IDF  : "voiture" et "automobile" -> aucune correspondance (mots differents)
    Embedding : "voiture" et "automobile" -> vecteurs tres proches

Choix du modele (important avec 16 Go de RAM et 0 euro de budget)
-----------------------------------------------------------------
  paraphrase-multilingual-MiniLM-L12-v2 : ~470 Mo, 384 dimensions, MULTILINGUE.
      C'est le defaut du projet car notre corpus est en francais.
  all-MiniLM-L6-v2 : ~90 Mo, 384 dimensions, ANGLAIS uniquement.
      Plus leger et plus rapide, mais mediocre sur du francais.
      A utiliser si WIKI_LANG=en, ou si votre disque est tres contraint.

Les deux tournent sur CPU sans GPU. On ne charge qu'UN SEUL modele a la fois.

Le cache des modeles est redirige vers ./hf_cache par la variable HF_HOME
(voir .env.example) : utile quand le disque C: est presque plein.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.config import CFG, INDEX_DIR, ROOT_DIR

# Redirige le cache Hugging Face AVANT tout import de sentence_transformers,
# sinon la bibliotheque a deja lu la variable et ecrit dans C:\Users\...
_hf_home = os.getenv("HF_HOME", "").strip()
if _hf_home:
    resolved = Path(_hf_home)
    if not resolved.is_absolute():
        resolved = (ROOT_DIR / resolved).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(resolved)

# Taille de lot volontairement modeste : encoder 256 chunks d'un coup ferait
# grimper la RAM inutilement sur une machine a 16 Go partagee avec l'IDE.
BATCH_SIZE = 16

_MODEL_CACHE: dict[str, object] = {}


class EmbeddingBackendUnavailable(RuntimeError):
    """sentence-transformers n'est pas installe ou le modele est inaccessible."""


# ==========================================================================
#  Chargement du modele
# ==========================================================================
def load_model(model_name: str | None = None):
    """Charge (une seule fois) le modele d'embeddings et le garde en memoire.

    Le premier appel telecharge le modele (~470 Mo) : comptez 1 a 3 minutes
    selon votre connexion.

    Attention a ce que signifie « une seule fois ». Dans un MEME processus,
    les appels suivants relisent `_MODEL_CACHE` et sont instantanes. Mais
    chaque NOUVEAU processus doit recharger le modele depuis le disque, et ce
    n'est pas gratuit : 94,6 s mesurees dans un conteneur avec le cache deja
    rempli. C'est pourquoi le modele est charge paresseusement (a la premiere
    question, pas a l'import) et pourquoi `mean_retrieval_time_s` de
    `src/evaluate.py` est domine par la premiere question.
    """
    model_name = model_name or CFG.embedding_model
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingBackendUnavailable(
            "La bibliotheque 'sentence-transformers' n'est pas installee.\n"
            "  Option 1 (recherche semantique) :\n"
            "      python -m pip install sentence-transformers\n"
            "  Option 2 (sans installation lourde) : basculez en lexical,\n"
            "      mettez RETRIEVER=tfidf dans votre fichier .env"
        ) from exc

    print(f"  Chargement du modele d'embeddings : {model_name}")
    print(f"  (premier lancement = telechargement, cache : {os.getenv('HF_HOME', 'defaut')})")
    try:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # noqa: BLE001 - reseau, disque plein, nom errone
        raise EmbeddingBackendUnavailable(
            f"Impossible de charger '{model_name}' : {type(exc).__name__}: {exc}\n"
            "Causes frequentes : pas de reseau au premier telechargement, "
            "disque plein, ou nom de modele mal orthographie."
        ) from exc

    _MODEL_CACHE[model_name] = model
    return model


class SentenceEmbedder:
    """Enveloppe fine autour de SentenceTransformer.

    Deux garanties utiles pour la suite :
      - les vecteurs renvoyes sont NORMALISES (norme 1), donc la similarite
        cosinus se calcule par un simple produit scalaire ;
      - le type est float32, ce qui divise par deux la RAM de l'index par
        rapport au float64 par defaut de NumPy.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or CFG.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode une liste de textes en matrice (n_textes, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosinus == produit scalaire
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode un seul texte (une question) en vecteur 1D."""
        return self.encode([text])[0]


# ==========================================================================
#  Cache disque des embeddings du corpus
# ==========================================================================
def _cache_key(texts: list[str], model_name: str) -> str:
    """Empreinte du couple (corpus, modele).

    Si un seul chunk change, ou si on change de modele, la cle change et le
    cache est automatiquement invalide : impossible de reutiliser par erreur
    des vecteurs calcules pour un autre corpus.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()[:16]


def embed_corpus(
    texts: list[str],
    model_name: str | None = None,
    use_cache: bool = True,
    cache_dir: Path = INDEX_DIR,
) -> np.ndarray:
    """Encode tout le corpus, avec mise en cache sur disque.

    Mesure reelle de ce TP : 1 345 chunks encodes en 3 min 02 s sur un CPU
    sans GPU (85 lots de 16), soit environ 7,4 chunks par seconde. Comptez
    donc plutot 4 a 5 minutes pour 2 000 chunks. On evite de le
    refaire a chaque lancement de l'application : c'est la difference entre
    un chatbot qui demarre en 3 secondes et un qui demarre en 3 minutes.
    """
    model_name = model_name or CFG.embedding_model
    embedder = SentenceEmbedder(model_name)

    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)

    key = _cache_key(texts, model_name)
    vectors_path = cache_dir / f"emb_{key}.npy"
    meta_path = cache_dir / f"emb_{key}.json"

    if use_cache and vectors_path.exists():
        try:
            vectors = np.load(vectors_path)
            if vectors.shape[0] == len(texts):
                print(f"  Embeddings lus depuis le cache : {vectors_path.name} "
                      f"{vectors.shape}")
                return vectors.astype(np.float32)
            print("  [!] Cache incoherent (taille differente) : recalcul.")
        except Exception as exc:  # noqa: BLE001 - fichier .npy corrompu
            print(f"  [!] Cache illisible ({type(exc).__name__}) : recalcul.")

    print(f"  Encodage de {len(texts)} passage(s) sur CPU...")
    vectors = embedder.encode(texts, show_progress=True)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, vectors)
        meta_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "n_texts": len(texts),
                    "dim": int(vectors.shape[1]),
                    "dtype": str(vectors.dtype),
                    "size_mb": round(vectors.nbytes / 1024 / 1024, 2),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  Embeddings mis en cache : {vectors_path.name} "
              f"({vectors.nbytes / 1024 / 1024:.1f} Mo)")
    return vectors


# ==========================================================================
#  Similarite
# ==========================================================================
def cosine_similarity_matrix(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """Similarite cosinus entre des requetes et des documents.

    Si les vecteurs sont deja normalises (c'est le cas ici), le cosinus est
    un simple produit matriciel. On renormalise quand meme par securite :
    cette fonction peut recevoir des vecteurs venus d'ailleurs.
    """
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    if doc_vectors.size == 0:
        return np.zeros((query_vectors.shape[0], 0), dtype=np.float32)

    query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    doc_norm = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1.0   # evite une division par zero
    doc_norm[doc_norm == 0] = 1.0

    return (query_vectors / query_norm) @ (doc_vectors / doc_norm).T


if __name__ == "__main__":
    # Demonstration :  python -m src.embeddings
    print("=== Test du modele d'embeddings ===")
    print(f"  Modele configure : {CFG.embedding_model}\n")

    phrases = [
        "Le chat dort sur le canape.",
        "Un felin se repose sur le sofa.",
        "La voiture roule tres vite sur l'autoroute.",
    ]
    try:
        embedder = SentenceEmbedder()
        vectors = embedder.encode(phrases)
    except EmbeddingBackendUnavailable as exc:
        print(f"[X] {exc}")
        raise SystemExit(1)

    print(f"\n  Dimensions : {vectors.shape} (3 phrases x {embedder.dim} dim.)")
    similarity = cosine_similarity_matrix(vectors, vectors)

    print("\n  Matrice de similarite cosinus :")
    for i, phrase in enumerate(phrases):
        scores = "  ".join(f"{value:.3f}" for value in similarity[i])
        print(f"    [{scores}]  {phrase[:42]}")

    print(
        f"\n  Phrases 1 et 2 (meme sens, mots differents) : {similarity[0][1]:.3f}"
        f"\n  Phrases 1 et 3 (sens different)              : {similarity[0][2]:.3f}"
    )
    if similarity[0][1] > similarity[0][2]:
        print("\n  [OK] Le modele capte bien le sens, pas seulement les mots.")
    else:
        print("\n  [!] Resultat inattendu : modele mal adapte a cette langue ?",
              file=sys.stderr)
