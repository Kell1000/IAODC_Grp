"""
Configuration centrale du projet WikiNLP Assistant.

Tout parametre reglable est lu ICI, et nulle part ailleurs. Les autres
modules importent CFG : on peut donc changer de modele, de langue ou de
fournisseur LLM sans toucher une seule ligne de code metier.

Ordre de priorite (du plus fort au plus faible) :
    1. variable d'environnement reelle du systeme (secret Docker / HF Spaces)
    2. fichier .env a la racine du projet
    3. valeur par defaut ecrite dans ce fichier

Aucune cle API n'est jamais ecrite en dur dans le code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Chemins du projet, calcules a partir de l'emplacement de CE fichier : le
# projet fonctionne donc quel que soit le repertoire courant (utile en Docker,
# ou le cwd vaut /app, et sous Streamlit qui change parfois de cwd).
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = ROOT_DIR / "eval"

# Fichiers de travail du pipeline : un nom de fichier = une etape du TP
RAW_ARTICLES = RAW_DIR / "articles.jsonl"                 # etape 2 : collecte
CLEAN_ARTICLES = PROCESSED_DIR / "articles_clean.jsonl"   # etape 3 : nettoyage
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"              # etape 5 : chunks RAG
DEMO_CHUNKS = PROCESSED_DIR / "demo_chunks.jsonl"         # mini-corpus versionne

# Charge .env s'il existe. override=False => un vrai secret systeme gagne
# toujours sur le fichier .env (comportement attendu en production).
load_dotenv(ROOT_DIR / ".env", override=False)


def _env(key: str, default: str) -> str:
    """Lit une variable d'environnement en tolerant la chaine vide.

    os.getenv(key, default) renvoie "" si la variable existe mais est vide,
    ce qui casserait les conversions int() / float(). On traite donc une
    valeur vide comme une valeur absente.
    """
    value = os.getenv(key, "")
    return value.strip() if value.strip() else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    """Tous les reglages du projet, valides a la construction."""

    # ---- Collecte Wikipedia --------------------------------------------
    wiki_lang: str = field(default_factory=lambda: _env("WIKI_LANG", "fr"))
    wiki_user_agent: str = field(
        default_factory=lambda: _env(
            "WIKI_USER_AGENT", "WikiNLP-Assistant/1.0 (etudiant@example.com)"
        )
    )
    max_articles: int = field(default_factory=lambda: _env_int("MAX_ARTICLES", 30))
    wiki_sleep: float = field(default_factory=lambda: _env_float("WIKI_SLEEP", 0.5))

    # ---- Decoupage et recherche ----------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))

    # --- Seuil du garde-fou : UN SEUIL PAR MOTEUR -----------------------
    # Les scores de TF-IDF et des embeddings ne sont PAS sur la meme echelle,
    # et c'est une erreur classique de leur appliquer le meme seuil.
    #
    # Mesure reelle faite pendant ce TP (corpus de 1 345 chunks, cf. annexe C) :
    #   TF-IDF      : questions DANS le corpus   0,151 a 0,205
    #                 questions HORS corpus      0,081 a 0,121
    #   -> un seuil unique de 0,25 refusait les 5 questions legitimes
    #      (false_refusals_in_corpus = 5) tout en refusant correctement le
    #      hors-corpus : le seuil etait au-dessus des DEUX populations.
    #   EMBEDDINGS  : questions DANS le corpus   0,821 a 0,897
    #                 questions HORS corpus      0,230 a 0,365
    #   -> l'estimation initiale de 0,30 tombait DANS la population hors
    #      corpus : mesure a 0,30, refusal_rate_out_of_corpus valait 0,667
    #      (une question hors sujet repondue au score 0,3652).
    #
    # Sonde complementaire de 20 questions (hors jeu d'evaluation officiel,
    # pour ne pas en fausser les metriques) :
    #   TF-IDF      : DANS le corpus 0,105 a 0,234 | HORS 0,076 a 0,139
    #   -> les deux populations SE RECOUVRENT. La question legitime "Comment
    #      un ordinateur apprend-il a partir d'exemples ?" donne 0,1054, soit
    #      MOINS que la question hors sujet "Comment soigner une angine chez
    #      l'enfant ?" a 0,1390. Aucun seuil ne les separe parfaitement :
    #      0,13 est le MEILLEUR COMPROMIS DISPONIBLE, juste sur le jeu
    #      officiel, et non un separateur parfait.
    #   EMBEDDINGS  : DANS le corpus 0,495 a 0,897 | HORS 0,186 a 0,365
    #   -> l'ecart subsiste. 0,43 est le MILIEU EXACT de l'ecart utile
    #      0,365-0,495 (0,43 - 0,365 = 0,495 - 0,43 = 0,065). Ne pas monter
    #      vers 0,55 : la question legitime "Qu'est-ce que le TAL ?" a pour
    #      MEILLEUR score 0,4954 et serait donc refusee en bloc.
    #
    # Precision importante sur cette borne de 0,4954 : c'est le score du
    # passage de RANG 1, qui se trouve etre hors sujet ; le bon passage (la
    # definition) arrive 3e, plus bas encore. C'est bien la valeur qui compte
    # ici, car le garde-fou compare le seuil au MEILLEUR score (best_score) :
    # au-dessus de 0,495 la question entiere est rejetee et le bon passage
    # n'atteint jamais le LLM. En RAG, ce qui importe est d'etre dans le
    # top-k, pas d'etre premier.
    #
    # D'ou deux valeurs par defaut distinctes, placees entre les deux
    # populations observees pour chaque moteur.
    min_score_tfidf: float = field(
        default_factory=lambda: _env_float("MIN_SCORE_TFIDF", 0.13)
    )
    min_score_embeddings: float = field(
        default_factory=lambda: _env_float("MIN_SCORE_EMBEDDINGS", 0.43)
    )
    # MIN_SCORE force la meme valeur pour les deux moteurs. Vide = automatique.
    min_score_override: str = field(default_factory=lambda: _env("MIN_SCORE", ""))

    # ---- Moteur de recherche -------------------------------------------
    retriever: str = field(default_factory=lambda: _env("RETRIEVER", "embeddings"))
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )
    vector_backend: str = field(
        default_factory=lambda: _env("VECTOR_BACKEND", "numpy")
    )

    # ---- Modele de langue ----------------------------------------------
    llm_provider: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", "extractive")
    )
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.1-8b-instant")
    )
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash")
    )
    ollama_host: str = field(
        default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1)
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", 512)
    )
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 60))

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Garde-fous : un reglage invalide doit echouer tot, avec un message
        # clair, plutot que produire un comportement silencieusement faux.
        valid_retrievers = {"tfidf", "embeddings"}
        if self.retriever not in valid_retrievers:
            raise ValueError(
                f"RETRIEVER='{self.retriever}' invalide. "
                f"Valeurs possibles : {sorted(valid_retrievers)}"
            )

        valid_providers = {"extractive", "groq", "gemini", "ollama"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"LLM_PROVIDER='{self.llm_provider}' invalide. "
                f"Valeurs possibles : {sorted(valid_providers)}"
            )

        valid_backends = {"numpy", "chroma"}
        if self.vector_backend not in valid_backends:
            raise ValueError(
                f"VECTOR_BACKEND='{self.vector_backend}' invalide. "
                f"Valeurs possibles : {sorted(valid_backends)}"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) doit etre strictement "
                f"inferieur a CHUNK_SIZE ({self.chunk_size}), sinon le "
                "decoupage boucle indefiniment."
            )

        if self.top_k < 1:
            raise ValueError("TOP_K doit valoir au moins 1.")

        if self.min_score_override:
            try:
                float(self.min_score_override)
            except ValueError:
                raise ValueError(
                    f"MIN_SCORE='{self.min_score_override}' n'est pas un nombre. "
                    "Laissez-le vide pour un seuil automatique par moteur."
                ) from None

        # Cree les dossiers de travail : evite les FileNotFoundError plus tard.
        for directory in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, EVAL_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def min_score_for(self, retriever: str | None = None) -> float:
        """Seuil du garde-fou adapte au moteur de recherche utilise.

        Si MIN_SCORE est renseigne dans .env, il s'applique a tous les
        moteurs (comportement explicite demande par l'utilisateur). Sinon,
        on prend le seuil calibre pour le moteur actif.
        """
        if self.min_score_override:
            return float(self.min_score_override)
        retriever = (retriever or self.retriever).lower()
        if retriever == "tfidf":
            return self.min_score_tfidf
        return self.min_score_embeddings

    @property
    def min_score(self) -> float:
        """Seuil applicable au moteur configure dans .env."""
        return self.min_score_for(self.retriever)

    # ------------------------------------------------------------------
    def llm_is_ready(self) -> tuple[bool, str]:
        """Le fournisseur LLM choisi est-il utilisable ?

        Renvoie (True, message) ou (False, raison lisible par un humain).
        L'interface web appelle cette methode pour afficher un message clair
        au lieu de laisser remonter une erreur HTTP 401 incomprehensible.
        """
        if self.llm_provider == "extractive":
            return True, "Mode extractif : aucune cle API necessaire."
        if self.llm_provider == "groq":
            if not self.groq_api_key:
                return False, "GROQ_API_KEY est vide dans .env."
            return True, f"Groq pret (modele {self.groq_model})."
        if self.llm_provider == "gemini":
            if not self.gemini_api_key:
                return False, "GEMINI_API_KEY est vide dans .env."
            return True, f"Gemini pret (modele {self.gemini_model})."
        if self.llm_provider == "ollama":
            return True, f"Ollama attendu sur {self.ollama_host} ({self.ollama_model})."
        return False, f"Fournisseur inconnu : {self.llm_provider}"

    def summary(self) -> dict[str, object]:
        """Vue non sensible de la configuration : les cles API sont masquees."""
        data = asdict(self)
        for secret_key in ("groq_api_key", "gemini_api_key"):
            value = str(data.get(secret_key) or "")
            data[secret_key] = f"***{value[-4:]}" if value else "(vide)"
        # min_score est une propriete calculee : asdict() ne la voit pas.
        data["min_score_effectif"] = self.min_score
        data["min_score_override"] = self.min_score_override or "(auto)"
        return data


# Instance unique importee partout :  from src.config import CFG
CFG = Config()


if __name__ == "__main__":
    # Diagnostic rapide :  python -m src.config
    print("=== Configuration WikiNLP Assistant ===")
    for key, value in CFG.summary().items():
        print(f"  {key:18s} = {value}")
    ready, message = CFG.llm_is_ready()
    print(f"\nLLM utilisable : {ready} -> {message}")
    print(f"Racine projet  : {ROOT_DIR}")
