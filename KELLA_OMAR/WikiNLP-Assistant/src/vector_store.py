"""
Etape 5 du TP - Base vectorielle (stockage et recherche des embeddings).

Deux implementations interchangeables, choisies par VECTOR_BACKEND dans .env :

  numpy  (defaut) : index "plat" exact, une simple matrice NumPy sur disque.
      + aucune dependance supplementaire, aucun probleme de compilation
      + recherche EXACTE (pas d'approximation)
      + sur 2000 chunks x 384 dimensions, une recherche prend ~1 ms
      - ne passerait pas l'echelle a 10 millions de vecteurs

  chroma : ChromaDB persistant (pip install chromadb)
      + vraie base vectorielle, metadonnees, requetes filtrees
      + c'est l'outil que vous rencontrerez en entreprise
      - dependances plus lourdes, parfois delicates a installer sous Windows

POURQUOI un index NumPy par defaut ?
Pour 20 a 50 articles Wikipedia, on obtient quelques milliers de chunks. A
cette echelle, la recherche exhaustive est plus rapide ET plus precise qu'un
index approximatif. Choisir une grosse base vectorielle ici serait de la
sur-ingenierie. Le TP montre quand meme ChromaDB, car savoir brancher une
vraie base vectorielle fait partie des competences attendues.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import CFG, INDEX_DIR


class SearchHit(dict):
    """Un resultat de recherche : dictionnaire avec id, score, texte, metadonnees."""


class VectorStore(Protocol):
    """Contrat commun aux deux implementations."""

    def add(self, ids: list[str], vectors: np.ndarray, documents: list[str],
            metadatas: list[dict]) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]: ...

    def save(self) -> None: ...

    def load(self) -> bool: ...

    def count(self) -> int: ...


# ==========================================================================
#  Backend 1 : index plat NumPy
# ==========================================================================
class NumpyVectorStore:
    """Index vectoriel exact, persiste en .npy + .jsonl."""

    name = "numpy"

    def __init__(self, path: Path = INDEX_DIR, collection: str = "wikinlp") -> None:
        self.path = Path(path)
        self.collection = collection
        self.vectors_file = self.path / f"{collection}_vectors.npy"
        self.records_file = self.path / f"{collection}_records.jsonl"
        self.meta_file = self.path / f"{collection}_meta.json"

        self._vectors: np.ndarray | None = None
        self._records: list[dict] = []

    # -- ecriture ------------------------------------------------------
    def add(self, ids, vectors, documents, metadatas) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"{len(ids)} identifiants mais {vectors.shape[0]} vecteurs."
            )

        records = [
            {"id": doc_id, "document": document, "metadata": metadata}
            for doc_id, document, metadata in zip(ids, documents, metadatas)
        ]

        if self._vectors is None or self._vectors.size == 0:
            self._vectors = vectors
            self._records = records
        else:
            self._vectors = np.vstack([self._vectors, vectors])
            self._records.extend(records)

    # -- lecture -------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchHit]:
        if self._vectors is None or self._vectors.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Les vecteurs du corpus sont deja normalises : le produit scalaire
        # EST la similarite cosinus. Une seule multiplication matricielle.
        scores = self._vectors @ query

        top_k = min(top_k, scores.shape[0])
        # argpartition trouve les k meilleurs sans trier tout le tableau,
        # puis on ne trie que ces k valeurs : moins couteux qu'un tri complet.
        best = np.argpartition(-scores, top_k - 1)[:top_k]
        best = best[np.argsort(-scores[best])]

        hits: list[SearchHit] = []
        for rank, index in enumerate(best, start=1):
            record = self._records[int(index)]
            hits.append(
                SearchHit(
                    rank=rank,
                    id=record["id"],
                    score=float(scores[int(index)]),
                    document=record["document"],
                    metadata=record["metadata"],
                )
            )
        return hits

    # -- persistance ---------------------------------------------------
    def save(self) -> None:
        if self._vectors is None:
            raise RuntimeError("Rien a sauvegarder : appelez add() d'abord.")
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_file, self._vectors)
        with self.records_file.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.meta_file.write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "collection": self.collection,
                    "n_vectors": int(self._vectors.shape[0]),
                    "dim": int(self._vectors.shape[1]),
                    "embedding_model": CFG.embedding_model,
                    "size_mb": round(self._vectors.nbytes / 1024 / 1024, 2),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Charge l'index depuis le disque. Renvoie False s'il n'existe pas."""
        if not (self.vectors_file.exists() and self.records_file.exists()):
            return False
        try:
            self._vectors = np.load(self.vectors_file).astype(np.float32)
            self._records = [
                json.loads(line)
                for line in self.records_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception as exc:  # noqa: BLE001 - index corrompu
            print(f"  [!] Index illisible ({type(exc).__name__}) : reconstruction requise.")
            return False

        if self._vectors.shape[0] != len(self._records):
            print("  [!] Index incoherent (vecteurs != enregistrements).")
            return False
        return True

    def count(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def reset(self) -> None:
        for file in (self.vectors_file, self.records_file, self.meta_file):
            file.unlink(missing_ok=True)
        self._vectors, self._records = None, []


# ==========================================================================
#  Backend 2 : ChromaDB
# ==========================================================================
class ChromaVectorStore:
    """Index vectoriel persistant base sur ChromaDB."""

    name = "chroma"

    def __init__(self, path: Path = INDEX_DIR, collection: str = "wikinlp") -> None:
        self.path = Path(path) / "chroma"
        self.collection_name = collection
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB n'est pas installe.\n"
                    "  Installez-le :  python -m pip install chromadb\n"
                    "  Ou revenez a l'index NumPy : VECTOR_BACKEND=numpy dans .env"
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            # Client persistant : les donnees survivent a l'arret du programme.
            self._client = chromadb.PersistentClient(path=str(self.path))
        return self._client

    def _get_collection(self, create: bool = True):
        if self._collection is None:
            client = self._get_client()
            if create:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    # cosine : coherent avec nos vecteurs normalises.
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self._collection = client.get_collection(name=self.collection_name)
        return self._collection

    def add(self, ids, vectors, documents, metadatas) -> None:
        collection = self._get_collection()
        vectors = np.asarray(vectors, dtype=np.float32)
        # Chroma limite la taille d'un lot : on decoupe pour ne pas depasser.
        batch = 500
        for start in range(0, len(ids), batch):
            end = start + batch
            collection.add(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchHit]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        result = collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[SearchHit] = []
        ids = result["ids"][0]
        for rank, doc_id in enumerate(ids, start=1):
            # Chroma renvoie une DISTANCE cosinus ; on repasse en SIMILARITE
            # pour que les scores soient comparables au backend NumPy.
            distance = float(result["distances"][0][rank - 1])
            hits.append(
                SearchHit(
                    rank=rank,
                    id=doc_id,
                    score=1.0 - distance,
                    document=result["documents"][0][rank - 1],
                    metadata=result["metadatas"][0][rank - 1] or {},
                )
            )
        return hits

    def save(self) -> None:
        # PersistentClient ecrit deja a chaque add() : rien a faire ici.
        pass

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            return self._get_collection(create=False).count() > 0
        except Exception:  # noqa: BLE001 - collection absente
            return False

    def count(self) -> int:
        try:
            return int(self._get_collection().count())
        except Exception:  # noqa: BLE001
            return 0

    def reset(self) -> None:
        try:
            self._get_client().delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - deja absente
            pass
        self._collection = None
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


# ==========================================================================
#  Fabrique
# ==========================================================================
def get_vector_store(backend: str | None = None, collection: str = "wikinlp"):
    """Renvoie l'implementation demandee par la configuration."""
    backend = (backend or CFG.vector_backend).lower()
    if backend == "numpy":
        return NumpyVectorStore(collection=collection)
    if backend == "chroma":
        return ChromaVectorStore(collection=collection)
    raise ValueError(f"VECTOR_BACKEND inconnu : {backend} (attendu : numpy | chroma)")


if __name__ == "__main__":
    # Demonstration :  python -m src.vector_store
    print("=== Test de la base vectorielle ===")
    print(f"  Backend configure : {CFG.vector_backend}\n")

    rng = np.random.default_rng(seed=42)
    fake_vectors = rng.standard_normal((5, 8)).astype(np.float32)
    fake_vectors /= np.linalg.norm(fake_vectors, axis=1, keepdims=True)

    store = get_vector_store(collection="demo_test")
    store.add(
        ids=[f"doc{i}" for i in range(5)],
        vectors=fake_vectors,
        documents=[f"Document numero {i}" for i in range(5)],
        metadatas=[{"title": f"Article {i}", "url": f"https://x/{i}"} for i in range(5)],
    )
    store.save()
    print(f"  {store.count()} vecteurs indexes dans le backend '{store.name}'")

    # On interroge avec le vecteur du document 2 : il doit ressortir premier
    # avec un score de 1.0 (un vecteur est parfaitement similaire a lui-meme).
    results = store.search(fake_vectors[2], top_k=3)
    print("\n  Recherche avec le vecteur du document 2 :")
    for hit in results:
        print(f"    #{hit['rank']} score={hit['score']:.4f}  {hit['document']}")

    assert results[0]["id"] == "doc2", "Le document 2 devrait arriver premier."
    print("\n  [OK] La base vectorielle fonctionne.")
    store.reset()
    print("  Index de test supprime.")
