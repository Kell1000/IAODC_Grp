"""
Etape 3 du TP - Nettoyage et pretraitement du corpus.

IDEE CENTRALE A RETENIR
-----------------------
Le pretraitement depend de la TACHE. Il n'existe pas un "texte pretraite"
universel. Ce module produit donc DEUX representations de chaque document :

  1. text_clean   -> texte lisible, ponctuation et mots outils CONSERVES.
                     C'est ce texte qui sera donne au LLM comme contexte.
                     Si on enlevait les stop words ici, "Qui a invente le
                     transformeur ?" deviendrait "invente transformeur" :
                     le LLM perdrait la syntaxe et repondrait moins bien.

  2. tokens_norm  -> tokens minuscules, sans stop words, racinises.
                     C'est cette version qui alimente TF-IDF, ou les mots
                     outils ("le", "de", "et") sont du bruit statistique.

Erreur classique du debutant : nettoyer agressivement UNE seule fois, puis
donner ce texte mutile au LLM. Les reponses deviennent mauvaises et on croit
que le probleme vient du modele.

Utilisation
-----------
    python -m src.preprocess
    python -m src.preprocess --min-chars 500
    python -m src.preprocess --stats

Entree  : data/raw/articles.jsonl
Sortie  : data/processed/articles_clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

from src.collect_wikipedia import append_jsonl, read_jsonl
from src.config import CFG, CLEAN_ARTICLES, RAW_ARTICLES

# --------------------------------------------------------------------------
#  Expressions regulieres de nettoyage (compilees une fois = plus rapide)
# --------------------------------------------------------------------------
# Titres de section du format WIKI : "== Histoire ==" ou "=== Details ==="
RE_SECTION = re.compile(r"^\s*={2,}\s*(.+?)\s*={2,}\s*$", re.MULTILINE)
# Modeles wiki residuels {{...}} et tables {|...|}
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")

# Formules mathematiques rendues en TeX : "{\displaystyle ax+b}".
# Les articles scientifiques de Wikipedia en sont remplis. Le motif gere
# deux niveaux d'accolades imbriquees, ce qui couvre les cas reels du type
# "{\displaystyle (x_{i},f(x_{i}))}".
RE_TEX = re.compile(
    r"\{\\displaystyle(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
)

# Debris de formules rendues en texte brut. Wikipedia fournit souvent la
# formule DEUX fois : une version TeX (ci-dessus) et une version texte dont
# chaque symbole devient une pseudo-phrase :
#     "a. x. +. b."    "(. x. i.)."
# Ces fragments n'apportent rien et polluent le decoupage en phrases. On
# supprime les suites de 3 symboles courts ou plus suivis d'un point.
# Le separateur accepte une ponctuation apres le point ("x. i., f.") : sans
# cela une simple virgule interrompait le motif et le debris survivait.
RE_MATH_NOISE = re.compile(
    r"(?:(?:[^\W\d_]{1,2}|\d{1,3}|[^\w\s]{1,3})\.[,;:\s]+){3,}"
)
# Balises HTML residuelles
RE_HTML = re.compile(r"<[^>]+>")
# References du type [1], [12], [note 3]
RE_REF = re.compile(r"\[\s*(?:\d+|note\s+\d+|n\s+\d+)\s*\]", re.IGNORECASE)
# Espaces multiples (hors saut de ligne)
RE_SPACES = re.compile(r"[^\S\n]+")
# Plus de deux sauts de ligne consecutifs
RE_NEWLINES = re.compile(r"\n{3,}")
# Caracteres de controle invisibles (sauf \n et \t)
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Tokens : lettres unicode (accents compris), chiffres, apostrophes internes
RE_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)

# Sections de fin d'article qui n'apportent rien a un systeme de QA :
# bibliographies, listes de liens, notes. On coupe l'article avant.
TAIL_SECTIONS = {
    "voir aussi", "notes et references", "notes et références",
    "references", "références", "bibliographie", "liens externes",
    "annexes", "articles connexes", "notes", "sources",
    "see also", "external links", "further reading",
    "bibliography", "notes and references",
}

# Version sans accents, calculee une fois : la comparaison devient
# insensible aux accents ("References" == "Références").
TAIL_SECTIONS_FOLDED = {
    "".join(
        char
        for char in unicodedata.normalize("NFD", section)
        if unicodedata.category(char) != "Mn"
    )
    for section in TAIL_SECTIONS
}

# Stop words de secours, utilises si le corpus NLTK n'est pas telechargeable
# (machine hors ligne, image Docker minimale, CI sans reseau).
FALLBACK_STOPWORDS = {
    "fr": {
        "a", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de",
        "des", "du", "elle", "elles", "en", "est", "et", "eux", "il", "ils",
        "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "meme",
        "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par",
        "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur",
        "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre",
        "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y", "ete", "etre",
        "avoir", "plus", "aussi", "comme", "dont", "sans", "deux", "peut",
        "leurs", "ainsi", "entre", "autre", "autres", "cela", "fait", "sont",
    },
    "en": {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
        "from", "had", "has", "have", "he", "her", "his", "i", "in", "is",
        "it", "its", "of", "on", "or", "she", "that", "the", "their", "them",
        "there", "these", "they", "this", "to", "was", "were", "which",
        "with", "would", "you", "your", "also", "can", "such", "other",
    },
}

# Codes langue Wikipedia -> noms de langue attendus par NLTK
NLTK_LANG = {"fr": "french", "en": "english", "es": "spanish", "de": "german"}


# ==========================================================================
#  Dependances NLTK : telechargement tolerant aux pannes
# ==========================================================================
_NLTK_READY: dict[str, bool] = {}


def ensure_nltk(resource: str) -> bool:
    """Garantit la presence d'une ressource NLTK, sans faire planter le script.

    NLTK >= 3.9 a renomme 'punkt' en 'punkt_tab' : on essaie les deux. Si
    aucun telechargement n'est possible (pas de reseau), on renvoie False et
    l'appelant bascule sur une solution de secours en expressions regulieres.
    """
    if resource in _NLTK_READY:
        return _NLTK_READY[resource]

    try:
        import nltk
    except ImportError:
        _NLTK_READY[resource] = False
        return False

    paths = {
        "punkt": ["tokenizers/punkt_tab", "tokenizers/punkt"],
        "stopwords": ["corpora/stopwords"],
    }[resource]

    for path in paths:
        try:
            nltk.data.find(path)
            _NLTK_READY[resource] = True
            return True
        except LookupError:
            continue

    names = ["punkt_tab", "punkt"] if resource == "punkt" else ["stopwords"]
    for name in names:
        try:
            if nltk.download(name, quiet=True):
                _NLTK_READY[resource] = True
                return True
        except Exception:  # noqa: BLE001 - reseau coupe, proxy, etc.
            continue

    print(
        f"  [!] Ressource NLTK '{resource}' indisponible : "
        "utilisation de la solution de secours (regex / liste interne).",
        file=sys.stderr,
    )
    _NLTK_READY[resource] = False
    return False


# ==========================================================================
#  1. Nettoyage du texte
# ==========================================================================
def _heading_of(line: str) -> str | None:
    """Renvoie le titre de section porte par cette ligne, sinon None.

    DEUX FORMATS A GERER, et c'est un piege reel du TP :

      "== Histoire ==" : format wikitexte brut.
      "Histoire"       : c'est ce que renvoie reellement wikipedia-api avec
                         ExtractFormat.WIKI. Les titres de section arrivent
                         comme de simples lignes, SANS aucun marqueur.

    Si l'on ne gere que le premier format (l'erreur naturelle quand on lit
    la documentation du wikitexte), le filtrage des sections finales ne
    s'applique jamais : bibliographies, listes de liens et notes avec leurs
    ISBN et DOI restent dans le corpus et remontent dans les resultats de
    recherche. On a constate exactement cela en executant le TP.
    """
    stripped = line.strip()
    if not stripped:
        return None

    match = RE_SECTION.match(line)
    if match:
        return match.group(1).strip()

    # Ligne courte sans ponctuation finale : tres probablement un titre.
    #
    # MAIS il faut exiger un minimum de lettres. Les formules mathematiques de
    # Wikipedia arrivent sous forme d'UN SYMBOLE PAR LIGNE, fortement indente :
    #     "le reseau repond \n  \n   a\n   x\n   +\n   b\n  \n {\\displaystyle ax+b}"
    # Sans ce garde-fou, chacune de ces lignes etait prise pour un titre et
    # recevait un point, ce qui fabriquait le texte "a. x. +. b." - exactement
    # le debris qui polluait le passage le mieux classe du corpus.
    # Les vrais titres de section ("Notes", "Voir aussi") ont au moins 3 lettres.
    if len(stripped) < 80 and stripped[-1] not in ".!?,;:":
        letters = sum(1 for char in stripped if char.isalpha())
        if letters >= 3:
            return stripped
    return None


def strip_tail_sections(text: str) -> str:
    """Coupe l'article avant ses sections finales (bibliographie, liens...).

    Ces sections ne sont que des listes de references. Elles ne contiennent
    jamais la reponse a une question de fond, mais elles sont riches en noms
    propres et en chiffres : elles obtiennent donc de bons scores TF-IDF et
    polluent les resultats. On les supprime.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        heading = _heading_of(line)
        if heading and strip_accents(heading.lower()) in TAIL_SECTIONS_FOLDED:
            return "\n".join(lines[:index])
    return text


def mark_headings(text: str) -> str:
    """Ajoute un point final aux lignes de titre restantes.

    Sans cela, le decoupage en phrases colle le titre a la phrase suivante
    et produit des chunks comme "Bibliometrie La bibliometrie est...".
    Un point suffit a separer proprement les deux.
    """
    output: list[str] = []
    for line in text.split("\n"):
        heading = _heading_of(line)
        if heading and heading == line.strip() and not RE_SECTION.match(line):
            output.append(f"{heading}.")
        else:
            output.append(line)
    return "\n".join(output)


def clean_text(raw: str) -> str:
    """Nettoyage MINIMAL et non destructif, adapte au contexte d'un LLM.

    Ce qui est supprime : artefacts de balisage, references [1], caracteres
    de controle, espaces superflus.
    Ce qui est CONSERVE : accents, majuscules, ponctuation, mots outils.
    """
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)   # unifie les variantes unicode
    text = text.replace(" ", " ")          # espace insecable -> espace
    text = text.replace("­", "")           # trait d'union conditionnel
    text = RE_CONTROL.sub("", text)
    text = strip_tail_sections(text)
    text = RE_TEX.sub(" ", text)        # formules TeX avant les modeles {{...}}
    text = RE_TEMPLATE.sub("", text)
    text = RE_HTML.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_MATH_NOISE.sub("", text)  # debris de formules en texte brut
    # Titre de section "== Histoire ==" -> "Histoire." : on garde l'info de
    # structure, utile comme contexte, sans les signes de balisage.
    text = RE_SECTION.sub(lambda m: f"\n{m.group(1).strip()}.\n", text)
    # Et pour les titres arrivant sans marqueur (cas de wikipedia-api).
    text = mark_headings(text)
    text = RE_SPACES.sub(" ", text)
    text = RE_NEWLINES.sub("\n\n", text)

    # Retire les espaces avant une ponctuation anglaise, garde la typographie
    # francaise (espace avant ? ! : ; est correcte en francais).
    text = re.sub(r"\s+([,.)])", r"\1", text)
    return text.strip()


# ==========================================================================
#  2. Segmentation en phrases
# ==========================================================================
RE_SENTENCE_FALLBACK = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü0-9])")


def split_sentences(text: str, lang: str = "fr") -> list[str]:
    """Decoupe un texte en phrases.

    Utilise le modele punkt de NLTK s'il est disponible (il connait les
    abreviations comme "M." ou "etc."), sinon une regex de secours qui coupe
    apres . ! ? suivi d'une majuscule.
    """
    if not text.strip():
        return []

    if ensure_nltk("punkt"):
        try:
            from nltk.tokenize import sent_tokenize

            return [
                sentence.strip()
                for sentence in sent_tokenize(text, language=NLTK_LANG.get(lang, "french"))
                if sentence.strip()
            ]
        except Exception:  # noqa: BLE001 - langue non supportee par punkt
            pass

    return [part.strip() for part in RE_SENTENCE_FALLBACK.split(text) if part.strip()]


# ==========================================================================
#  3. Tokenisation et normalisation
# ==========================================================================
def tokenize(text: str) -> list[str]:
    """Decoupe en tokens : mots (accents compris) et nombres.

    On utilise une regex unicode plutot que text.split() : split() laisserait
    la ponctuation collee aux mots ("langage," != "langage").
    """
    return RE_TOKEN.findall(text)


def get_stopwords(lang: str = "fr") -> set[str]:
    """Liste de mots outils, depuis NLTK si possible, sinon liste interne."""
    if ensure_nltk("stopwords"):
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words(NLTK_LANG.get(lang, "french")))
        except Exception:  # noqa: BLE001
            pass
    return set(FALLBACK_STOPWORDS.get(lang, FALLBACK_STOPWORDS["fr"]))


def strip_accents(text: str) -> str:
    """Retire les accents : 'modele' et 'modèle' doivent correspondre."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


_STEMMERS: dict[str, object] = {}


def get_stemmer(lang: str = "fr"):
    """Racinisateur Snowball de NLTK (aucun telechargement necessaire).

    Pourquoi une racinisation (stemming) et non une lemmatisation ?
    Le lemmatiseur WordNet de NLTK ne gere que l'anglais. Pour le francais,
    une vraie lemmatisation demanderait spaCy + un modele a telecharger
    (environ 16 Mo). Snowball est integre, instantane, et suffit a faire
    correspondre "modeles", "modele" et "modeliser" pour du TF-IDF.
    Si vous voulez une vraie lemmatisation : voir --use-spacy ci-dessous.
    """
    if lang not in _STEMMERS:
        try:
            from nltk.stem.snowball import SnowballStemmer

            _STEMMERS[lang] = SnowballStemmer(NLTK_LANG.get(lang, "french"))
        except Exception:  # noqa: BLE001
            _STEMMERS[lang] = None
    return _STEMMERS[lang]


def normalize_tokens(
    text: str,
    lang: str = "fr",
    remove_stopwords: bool = True,
    stem: bool = True,
    min_len: int = 2,
) -> list[str]:
    """Produit les tokens normalises destines a TF-IDF.

    Chaine appliquee : minuscules -> sans accents -> filtrage des stop words
    -> filtrage des tokens trop courts -> racinisation.
    """
    tokens = [token.lower() for token in tokenize(text)]
    tokens = [strip_accents(token) for token in tokens]

    if remove_stopwords:
        stops = {strip_accents(word.lower()) for word in get_stopwords(lang)}
        tokens = [token for token in tokens if token not in stops]

    tokens = [token for token in tokens if len(token) >= min_len]

    if stem:
        stemmer = get_stemmer(lang)
        if stemmer is not None:
            tokens = [stemmer.stem(token) for token in tokens]
    return tokens


# ==========================================================================
#  4. Deduplication
# ==========================================================================
def content_fingerprint(text: str) -> str:
    """Empreinte stable du CONTENU d'un texte.

    On normalise avant de hacher : deux articles qui ne different que par la
    casse, les accents ou les espaces produisent la meme empreinte et sont
    donc reconnus comme doublons.
    """
    normalized = strip_accents(text.lower())
    normalized = re.sub(r"\W+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Supprime les doublons par titre, par pageid et par empreinte de contenu."""
    unique: list[dict] = []
    seen_titles: set[str] = set()
    seen_ids: set[object] = set()
    seen_prints: set[str] = set()
    removed = 0

    for record in records:
        title_key = strip_accents(record.get("title", "").lower().strip())
        page_id = record.get("pageid")
        fingerprint = content_fingerprint(record.get("text", "")[:5000])

        is_duplicate = (
            title_key in seen_titles
            or (page_id is not None and page_id in seen_ids)
            or fingerprint in seen_prints
        )
        if is_duplicate:
            removed += 1
            continue

        seen_titles.add(title_key)
        if page_id is not None:
            seen_ids.add(page_id)
        seen_prints.add(fingerprint)
        unique.append(record)

    return unique, removed


# ==========================================================================
#  5. Pipeline complet
# ==========================================================================
def process_record(record: dict, lang: str) -> dict:
    """Transforme un article brut en article pretraite (les 2 representations)."""
    cleaned = clean_text(record.get("text", ""))
    sentences = split_sentences(cleaned, lang)
    tokens_norm = normalize_tokens(cleaned, lang)

    return {
        "pageid": record.get("pageid"),
        "title": record.get("title", ""),
        "url": record.get("url", ""),
        "lang": record.get("lang", lang),
        "license": record.get("license", "CC BY-SA 4.0"),
        # --- representation 1 : pour le LLM et le RAG ---
        "text_clean": cleaned,
        "sentences": sentences,
        # --- representation 2 : pour TF-IDF ---
        "tokens_norm": tokens_norm,
        "norm_text": " ".join(tokens_norm),
        # --- metriques ---
        "n_chars_raw": len(record.get("text", "")),
        "n_chars_clean": len(cleaned),
        "n_sentences": len(sentences),
        "n_tokens": len(tokenize(cleaned)),
        "n_tokens_norm": len(tokens_norm),
    }


def run(
    in_path: Path = RAW_ARTICLES,
    out_path: Path = CLEAN_ARTICLES,
    lang: str | None = None,
    min_chars: int = 500,
) -> dict:
    """Execute tout le pretraitement et renvoie un rapport chiffre."""
    lang = lang or CFG.wiki_lang
    records = read_jsonl(in_path)
    if not records:
        print(
            f"  [X] Aucun article dans {in_path}.\n"
            "      Lancez d'abord :  python -m src.collect_wikipedia",
            file=sys.stderr,
        )
        return {"n_in": 0, "n_out": 0}

    print(f"  Articles lus : {len(records)}")

    records, n_duplicates = deduplicate(records)
    print(f"  Doublons supprimes : {n_duplicates}")

    if out_path.exists():
        out_path.unlink()  # le pretraitement est idempotent : on repart propre

    kept, too_short = 0, 0
    total_raw, total_clean = 0, 0

    for record in records:
        processed = process_record(record, lang)
        total_raw += processed["n_chars_raw"]
        total_clean += processed["n_chars_clean"]

        # Document trop court APRES nettoyage : ebauche ou page de service.
        if processed["n_chars_clean"] < min_chars:
            too_short += 1
            print(
                f"  [-] trop court apres nettoyage "
                f"({processed['n_chars_clean']} car.) : {processed['title']}"
            )
            continue

        append_jsonl(out_path, processed)
        kept += 1

    reduction = (1 - total_clean / total_raw) * 100 if total_raw else 0.0
    stats = {
        "n_in": len(records) + n_duplicates,
        "n_duplicates": n_duplicates,
        "n_too_short": too_short,
        "n_out": kept,
        "chars_raw": total_raw,
        "chars_clean": total_clean,
        "reduction_pct": round(reduction, 1),
    }

    print("\n=== Rapport de pretraitement ===")
    print(f"  Entree               : {stats['n_in']} article(s)")
    print(f"  Doublons retires     : {stats['n_duplicates']}")
    print(f"  Trop courts retires  : {stats['n_too_short']}")
    print(f"  Conserves            : {stats['n_out']}")
    print(f"  Caracteres avant     : {stats['chars_raw']:,}".replace(",", " "))
    print(f"  Caracteres apres     : {stats['chars_clean']:,}".replace(",", " "))
    print(f"  Reduction            : {stats['reduction_pct']} %")
    print(f"  Sortie               : {out_path}")

    if kept == 0:
        print("  [!] Aucun document conserve : baissez --min-chars.")
    return stats


def show_stats(path: Path = CLEAN_ARTICLES) -> None:
    """Affiche un apercu du corpus pretraite (controle visuel)."""
    records = read_jsonl(path)
    if not records:
        print(f"  [X] {path} est vide ou absent.")
        return

    print(f"=== Corpus pretraite : {len(records)} document(s) ===\n")
    for record in records[:3]:
        print(f"  Titre      : {record['title']}")
        print(f"  URL        : {record['url']}")
        print(f"  Phrases    : {record['n_sentences']}")
        print(f"  Tokens     : {record['n_tokens']} (normalises : {record['n_tokens_norm']})")
        print(f"  Extrait    : {record['text_clean'][:160]}...")
        print(f"  Tokens norm: {record['tokens_norm'][:12]}")
        print()

    total_sentences = sum(record["n_sentences"] for record in records)
    total_tokens = sum(record["n_tokens"] for record in records)
    print(f"  Total phrases : {total_sentences}")
    print(f"  Total tokens  : {total_tokens:,}".replace(",", " "))
    print(f"  Vocabulaire normalise : "
          f"{len({token for record in records for token in record['tokens_norm']}):,}"
          .replace(",", " "))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nettoyage et pretraitement NLP.")
    parser.add_argument("--in", dest="in_path", type=Path, default=RAW_ARTICLES)
    parser.add_argument("--out", dest="out_path", type=Path, default=CLEAN_ARTICLES)
    parser.add_argument("--lang", default=CFG.wiki_lang)
    parser.add_argument("--min-chars", type=int, default=500,
                        help="Longueur minimale apres nettoyage (defaut 500).")
    parser.add_argument("--stats", action="store_true",
                        help="Affiche le corpus deja pretraite, sans retraiter.")
    args = parser.parse_args(argv)

    if args.stats:
        show_stats(args.out_path)
        return 0

    print("=== Pretraitement NLP ===")
    stats = run(args.in_path, args.out_path, args.lang, args.min_chars)
    return 0 if stats.get("n_out") else 1


if __name__ == "__main__":
    raise SystemExit(main())
