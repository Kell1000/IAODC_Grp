"""
Etape 2 du TP - Collecte d'un petit corpus Wikipedia.

Objectif : constituer un corpus de 20 a 50 articles sur un theme donne, en
respectant les regles d'acces de Wikimedia, et de facon RELANCABLE : si vous
relancez le script, il ne retelecharge pas ce qui est deja sur le disque.

Regles de bonne conduite appliquees ici
---------------------------------------
1. User-Agent explicite et identifiable (exige par la politique Wikimedia).
   Une valeur generique ou absente peut entrainer un blocage HTTP 403.
2. Une pause entre deux requetes (WIKI_SLEEP) : on ne martele pas l'API.
3. Un plafond dur d'articles (MAX_ARTICLES) : on controle RAM et disque.
4. On conserve l'URL, la langue et la licence de chaque article, pour pouvoir
   citer les sources dans le chatbot (Wikipedia est sous licence CC BY-SA).

Utilisation
-----------
    python -m src.collect_wikipedia
    python -m src.collect_wikipedia --topics "Traitement automatique du langage naturel" "Apprentissage profond"
    python -m src.collect_wikipedia --max 40 --lang fr
    python -m src.collect_wikipedia --stats          # juste un rapport
    python -m src.collect_wikipedia --force          # repart de zero

Sortie : data/raw/articles.jsonl (un objet JSON par ligne)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from src.config import CFG, RAW_ARTICLES

# La licence de tout contenu textuel de Wikipedia. On la stocke dans chaque
# enregistrement : c'est ce qui rend l'attribution possible cote interface.
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"

# Themes par defaut : le TP porte sur le NLP, le corpus parle donc de NLP.
# Modifiable avec --topics sans toucher au code.
#
# ATTENTION AUX ACCENTS : les titres sont transmis tels quels a MediaWiki.
# "Reseau de neurones artificiels" est resolu automatiquement, mais
# "Grand modele de langage" NON (l'article s'appelle "Grand modele de
# langage" avec un accent sur le e de modele). En cas de doute, cherchez
# l'article sur fr.wikipedia.org et copiez son titre exact.
DEFAULT_TOPICS = [
    "Traitement automatique des langues",
    "Apprentissage automatique",
    "Apprentissage profond",
    "Intelligence artificielle",
    "Grand modèle de langage",
    "Réseau de neurones artificiels",
    "Linguistique informatique",
    "Traduction automatique",
]

# Longueur minimale d'un article utile. En dessous, c'est une ebauche ou une
# page d'homonymie : du bruit pour un systeme de questions-reponses.
MIN_ARTICLE_CHARS = 800

# Pages d'INFRASTRUCTURE BIBLIOGRAPHIQUE a ecarter.
#
# Pourquoi cette liste existe : le classement par co-citation (voir
# candidate_titles) favorise les pages citees par beaucoup d'articles. Or les
# modeles de references de Wikipedia pointent systematiquement vers "Digital
# Object Identifier", "ISBN", "ISSN"... Ces pages obtiennent donc un score
# tres eleve alors qu'elles ne parlent pas du tout du theme. Sans ce filtre,
# elles occupent les premieres places de n'importe quelle collecte.
SKIP_CONTAINS = (
    "international standard",
    "digital object identifier",
    "autorite (sciences de l'information)",
    "autorité (sciences de l'information)",
    "bibliotheque nationale",
    "bibliothèque nationale",
    "fichier d'autorite",
    "fichier d'autorité",
    "numero de controle",
    "numéro de contrôle",
    "identifiant",
    "systeme universitaire de documentation",
    "système universitaire de documentation",
    "gemeinsame normdatei",
    "virtual international authority file",
)

# Prefixes a ignorer : pages de service, listes, homonymie, portails...
SKIP_PREFIXES = (
    "Liste ",
    "Portail:",
    "Categorie:",
    "Catégorie:",
    "Modele:",
    "Modèle:",
    "Aide:",
    "Discussion",
    "Wikipedia:",
    "Wikipédia:",
    "Fichier:",
    "Projet:",
    "Reference:",
    "Référence:",
)


# ==========================================================================
#  Lecture / ecriture JSONL
# ==========================================================================
def read_jsonl(path: Path) -> list[dict]:
    """Lit un fichier JSONL. Renvoie [] si le fichier n'existe pas.

    Une ligne corrompue (ecriture interrompue par Ctrl+C) est ignoree avec
    un avertissement, plutot que de faire planter tout le pipeline.
    """
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"  [!] Ligne {line_number} illisible dans {path.name}, ignoree.",
                    file=sys.stderr,
                )
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Ajoute un enregistrement a la fin du fichier JSONL.

    On ecrit au fur et a mesure (et non tout a la fin) : si la collecte est
    interrompue, les articles deja recuperes sont conserves. C'est ce qui
    rend le script relancable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==========================================================================
#  Client Wikipedia
# ==========================================================================
def build_client(lang: str, user_agent: str):
    """Construit le client wikipedia-api avec un User-Agent conforme."""
    try:
        import wikipediaapi
    except ImportError as exc:  # pragma: no cover - probleme d'installation
        raise SystemExit(
            "La bibliotheque 'wikipedia-api' est absente.\n"
            "Installez-la :  python -m pip install wikipedia-api"
        ) from exc

    if "example.com" in user_agent:
        print(
            "  [!] WIKI_USER_AGENT contient encore 'example.com'.\n"
            "      Mettez un vrai contact dans .env : Wikimedia peut bloquer\n"
            "      les User-Agent generiques (erreur HTTP 403).",
            file=sys.stderr,
        )

    return wikipediaapi.Wikipedia(
        user_agent=user_agent,
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def should_skip_title(title: str, extra_excludes: tuple[str, ...] = ()) -> bool:
    """Ecarte les pages de service, listes, homonymies et pages d'identifiants."""
    if title.startswith(SKIP_PREFIXES):
        return True
    lowered = title.lower()
    if "homonymie" in lowered or "disambiguation" in lowered:
        return True
    if any(marker in lowered for marker in SKIP_CONTAINS):
        return True
    return any(marker.lower() in lowered for marker in extra_excludes if marker)


def fetch_page(client, title: str, retries: int = 3, sleep: float = 0.5):
    """Recupere une page avec gestion des erreurs reseau et re-essais.

    Les erreurs reseau sont normales (wifi instable, API momentanement
    saturee) : on re-essaie avec un delai croissant (backoff exponentiel)
    au lieu d'abandonner toute la collecte.
    """
    for attempt in range(1, retries + 1):
        try:
            page = client.page(title)
            # L'acces a .exists() declenche l'appel reseau reel.
            if not page.exists():
                return None
            return page
        except Exception as exc:  # noqa: BLE001 - on veut tout rattraper ici
            wait = sleep * (2 ** (attempt - 1))
            print(
                f"  [!] Erreur reseau sur '{title}' "
                f"(essai {attempt}/{retries}) : {type(exc).__name__}. "
                f"Nouvelle tentative dans {wait:.1f}s",
                file=sys.stderr,
            )
            if attempt == retries:
                return None
            time.sleep(wait)
    return None


def page_to_record(page, lang: str) -> dict:
    """Transforme une page wikipedia-api en enregistrement JSON serialisable."""
    return {
        "pageid": page.pageid,
        "title": page.title,
        "text": page.text,
        "summary": page.summary,
        "url": page.fullurl,
        "lang": lang,
        "n_chars": len(page.text),
        "license": WIKIPEDIA_LICENSE,
        "source": "wikipedia",
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ==========================================================================
#  Collecte
# ==========================================================================
def _title_words(titles: Iterable[str]) -> set[str]:
    """Mots significatifs des titres de themes, pour mesurer la proximite.

    On derive le vocabulaire du theme depuis les graines elles-memes : aucune
    liste de mots-cles n'est codee en dur, donc changer --topics change
    automatiquement ce qui est considere comme pertinent.
    """
    words: set[str] = set()
    for title in titles:
        for word in re.split(r"\W+", title.lower()):
            if len(word) > 3:   # ecarte "de", "des", "la", "un"...
                words.add(word)
    return words


def candidate_titles(
    client,
    topics: Iterable[str],
    sleep: float,
    extra_excludes: tuple[str, ...] = (),
) -> Iterator[str]:
    """Produit les titres a collecter : d'abord les themes, puis leurs liens.

    Strategie en deux vagues :
      - vague 1 : les articles "graines" demandes explicitement ;
      - vague 2 : les articles lies depuis ces graines, CLASSES PAR
        PERTINENCE et non par ordre alphabetique.

    POURQUOI PAS L'ORDRE ALPHABETIQUE ?
    C'est l'erreur a ne pas commettre. Un article comme "Intelligence
    artificielle" contient des centaines de liens ; pris par ordre
    alphabetique, on recolte "Alibaba.com", "Avion" et "Besancon" avant
    d'atteindre le moindre article de NLP. Le corpus devient incoherent et
    le chatbot repond a cote.

    CLASSEMENT UTILISE (deux signaux simples et gratuits) :
      1. co-citation : combien de pages graines pointent vers ce lien ?
         Un article cite par 4 pages de NLP est presque surement du NLP.
      2. recouvrement lexical avec les mots des titres de themes.
    Le tri final est deterministe (le titre sert de dernier critere), donc
    deux executions produisent le meme corpus.
    """
    seeds = [topic for topic in topics if topic.strip()]
    for title in seeds:
        yield title

    seed_words = _title_words(seeds)
    seen_in_seeds = {title.lower() for title in seeds}

    # link_votes[titre] = nombre de pages graines qui pointent vers ce titre
    link_votes: Counter[str] = Counter()
    for seed in seeds:
        page = fetch_page(client, seed, sleep=sleep)
        time.sleep(sleep)
        if page is None:
            continue
        for linked_title in page.links:
            if should_skip_title(linked_title, extra_excludes):
                continue
            if linked_title.lower() in seen_in_seeds:
                continue
            link_votes[linked_title] += 1

    def score(title: str) -> tuple[int, int, str]:
        overlap = len(_title_words([title]) & seed_words)
        # Negatifs pour trier en ordre decroissant, puis titre croissant.
        return (-link_votes[title], -overlap, title)

    for title in sorted(link_votes, key=score):
        yield title


def collect(
    topics: list[str],
    max_articles: int,
    lang: str,
    user_agent: str,
    sleep: float,
    out_path: Path,
    force: bool = False,
    extra_excludes: tuple[str, ...] = (),
) -> dict:
    """Collecte le corpus et renvoie un petit rapport chiffre."""
    if force and out_path.exists():
        backup = out_path.with_suffix(".jsonl.bak")
        out_path.replace(backup)
        print(f"  Ancien corpus deplace vers {backup.name}")

    existing = read_jsonl(out_path)
    # Deduplication sur deux cles : le titre ET le pageid. Le titre seul ne
    # suffit pas (redirections), le pageid seul non plus (pages renommees).
    seen_titles = {record.get("title", "") for record in existing}
    seen_ids = {record.get("pageid") for record in existing}

    print(f"  Corpus existant : {len(existing)} article(s)")
    if len(existing) >= max_articles:
        print(
            f"  Objectif deja atteint ({len(existing)} >= {max_articles}). "
            "Rien a telecharger. Utilisez --force pour repartir de zero."
        )
        return report(out_path)

    client = build_client(lang, user_agent)

    added = 0
    skipped_short = 0
    skipped_missing = 0
    target = max_articles - len(existing)
    print(f"  Objectif : +{target} article(s) (plafond {max_articles})\n")

    for title in candidate_titles(client, topics, sleep, extra_excludes):
        if added >= target:
            break
        if title in seen_titles or should_skip_title(title, extra_excludes):
            continue

        page = fetch_page(client, title, sleep=sleep)
        time.sleep(sleep)  # politesse : une pause APRES chaque appel reseau

        if page is None:
            skipped_missing += 1
            print(f"  [-] introuvable / erreur : {title}")
            continue
        if page.pageid in seen_ids:
            continue  # redirection vers un article deja collecte
        if len(page.text) < MIN_ARTICLE_CHARS:
            skipped_short += 1
            print(f"  [-] trop court ({len(page.text)} car.) : {page.title}")
            seen_titles.add(title)
            continue

        record = page_to_record(page, lang)
        append_jsonl(out_path, record)
        seen_titles.add(title)
        seen_titles.add(page.title)
        seen_ids.add(page.pageid)
        added += 1
        print(f"  [{len(existing) + added:3d}/{max_articles}] {page.title} "
              f"({record['n_chars']:,} car.)".replace(",", " "))

    print(
        f"\n  Ajoutes : {added} | ignores trop courts : {skipped_short} "
        f"| introuvables : {skipped_missing}"
    )
    return report(out_path)


# ==========================================================================
#  Rapport qualite
# ==========================================================================
def report(path: Path) -> dict:
    """Mesure le volume et la qualite du corpus collecte.

    Ces chiffres sont a RELEVER pendant le TP : ils dependent de vos themes,
    de la langue et du moment de la collecte.
    """
    records = read_jsonl(path)
    if not records:
        return {"n_articles": 0}

    lengths = sorted(record.get("n_chars", 0) for record in records)
    total = sum(lengths)
    titles = [record.get("title", "") for record in records]

    stats = {
        "n_articles": len(records),
        "n_unique_titles": len(set(titles)),
        "total_chars": total,
        "mean_chars": round(total / len(records)),
        "median_chars": lengths[len(lengths) // 2],
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "disk_kb": round(path.stat().st_size / 1024, 1),
        "langs": sorted({record.get("lang", "?") for record in records}),
    }

    print("\n=== Qualite du corpus ===")
    print(f"  Articles              : {stats['n_articles']}")
    print(f"  Titres uniques        : {stats['n_unique_titles']}")
    print(f"  Caracteres au total   : {stats['total_chars']:,}".replace(",", " "))
    print(f"  Moyenne / article     : {stats['mean_chars']:,}".replace(",", " "))
    print(f"  Mediane / article     : {stats['median_chars']:,}".replace(",", " "))
    print(f"  Plus court / long     : {stats['min_chars']} / {stats['max_chars']}")
    print(f"  Taille sur disque     : {stats['disk_kb']} Ko")
    print(f"  Langue(s)             : {', '.join(stats['langs'])}")

    if stats["n_articles"] != stats["n_unique_titles"]:
        print("  [!] Des titres sont dupliques : verifiez la deduplication.")
    if stats["n_articles"] < 20:
        print("  [!] Moins de 20 articles : relancez pour enrichir le corpus.")
    return stats


# ==========================================================================
#  Interface en ligne de commande
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collecte un petit corpus Wikipedia (relancable).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topics", nargs="+", default=DEFAULT_TOPICS,
        help="Articles graines (par defaut : themes NLP).",
    )
    parser.add_argument("--max", type=int, default=CFG.max_articles,
                        help=f"Plafond d'articles (defaut {CFG.max_articles}).")
    parser.add_argument("--lang", default=CFG.wiki_lang,
                        help=f"Langue Wikipedia (defaut {CFG.wiki_lang}).")
    parser.add_argument("--sleep", type=float, default=CFG.wiki_sleep,
                        help=f"Pause entre requetes (defaut {CFG.wiki_sleep}s).")
    parser.add_argument("--out", type=Path, default=RAW_ARTICLES,
                        help="Fichier JSONL de sortie.")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Fragments de titres a ecarter en plus "
                             "(ex. --exclude \"Jeux video\" Sport).")
    parser.add_argument("--force", action="store_true",
                        help="Sauvegarde le corpus existant et repart de zero.")
    parser.add_argument("--stats", action="store_true",
                        help="Affiche seulement le rapport qualite, sans collecter.")
    args = parser.parse_args(argv)

    if args.stats:
        stats = report(args.out)
        return 0 if stats.get("n_articles") else 1

    print("=== Collecte Wikipedia ===")
    print(f"  Langue      : {args.lang}")
    print(f"  Plafond     : {args.max} articles")
    print(f"  Themes      : {', '.join(args.topics)}")
    print(f"  User-Agent  : {CFG.wiki_user_agent}")
    print(f"  Sortie      : {args.out}\n")

    stats = collect(
        topics=args.topics,
        max_articles=args.max,
        lang=args.lang,
        user_agent=CFG.wiki_user_agent,
        sleep=args.sleep,
        out_path=args.out,
        force=args.force,
        extra_excludes=tuple(args.exclude),
    )
    return 0 if stats.get("n_articles") else 1


if __name__ == "__main__":
    raise SystemExit(main())
