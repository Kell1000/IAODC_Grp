"""
Etape 8 du TP - Evaluation mesuree du systeme.

REGLE DEONTOLOGIQUE DE CE MODULE
--------------------------------
Ce script ne contient AUCUN resultat ecrit a l'avance. Il MESURE, sur votre
corpus et votre machine, puis ecrit un rapport. Les chiffres que vous
mettrez dans votre compte rendu doivent sortir d'ici, pas d'un exemple
recopie : un meme code donne des scores differents selon le corpus, la
langue, le modele et le materiel.

Ce qui est mesure
-----------------
  1. Qualite de la recherche
       hit@k  : la source attendue est-elle dans les k passages renvoyes ?
       MRR    : a quel rang ? (1/rang, moyenne sur les questions)
  2. Couverture des mots cles attendus dans la reponse
  3. Fidelite aux sources (anti-hallucination)
       Pour chaque question hors corpus, le systeme refuse-t-il de repondre ?
       C'est la metrique la plus importante : un systeme RAG qui invente est
       plus dangereux qu'un systeme qui dit "je ne sais pas".
  4. Temps de reponse (recherche / generation / total)
  5. Consommation memoire approximative du processus

Utilisation
-----------
    python -m src.evaluate                       # retriever de .env
    python -m src.evaluate --retriever tfidf
    python -m src.evaluate --compare             # tfidf ET embeddings
    python -m src.evaluate --out eval/results_moi.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from src.config import CFG, EVAL_DIR
from src.rag import NO_ANSWER, RagPipeline, get_retriever, load_chunks

QUESTIONS_FILE = EVAL_DIR / "questions.json"

# Marqueurs d'un refus de repondre. On en liste plusieurs car le LLM peut
# reformuler legerement la phrase imposee par la consigne systeme.
REFUSAL_MARKERS = [
    "je ne trouve pas cette information",
    "ne contiennent pas",
    "pas cette information",
    "aucun extrait pertinent",
    "je ne sais pas",
]


def _fold(text: str) -> str:
    """Minuscules sans accents : comparer 'modele' et 'modèle' sans piege."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def is_refusal(answer: str) -> bool:
    """La reponse est-elle un refus (le systeme dit ne pas savoir) ?"""
    folded = _fold(answer)
    return any(_fold(marker) in folded for marker in REFUSAL_MARKERS)


def memory_mb() -> float:
    """Memoire approximative du processus, en Mo.

    psutil donne la vraie memoire resident (RSS) mais n'est pas dans nos
    dependances. On l'utilise s'il est present, sinon on se rabat sur
    tracemalloc, qui ne compte que les allocations Python (donc SOUS-ESTIME :
    il ignore la memoire de torch et des modeles). Le chiffre est indicatif.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        current, _peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"[X] Jeu de questions absent : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["questions"]


# ==========================================================================
#  Evaluation
# ==========================================================================
def evaluate(
    retriever_kind: str | None = None,
    top_k: int | None = None,
    questions: list[dict] | None = None,
) -> dict:
    """Execute toutes les questions et renvoie les metriques mesurees."""
    retriever_kind = retriever_kind or CFG.retriever
    top_k = top_k or CFG.top_k
    questions = questions or load_questions()
    # Seuil reellement applique pour CE moteur (les echelles de score de
    # TF-IDF et des embeddings sont differentes).
    threshold = CFG.min_score_for(retriever_kind)

    tracemalloc.start()
    memory_before = memory_mb()

    print("=" * 74)
    print(f"EVALUATION - retriever={retriever_kind}  top_k={top_k}  "
          f"provider={CFG.llm_provider}")
    print("=" * 74)

    chunks = load_chunks()
    if not chunks:
        raise SystemExit(
            "[X] Aucun chunk indexe. Lancez : python -m src.rag --build"
        )

    build_start = time.perf_counter()
    retriever = get_retriever(retriever_kind, chunks=chunks)
    pipeline = RagPipeline(retriever=retriever, chunks=chunks)
    build_time = time.perf_counter() - build_start
    memory_after_index = memory_mb()

    print(f"\n  Corpus          : {len(chunks)} chunks")
    print(f"  Index construit : {build_time:.2f}s")
    print(f"  Memoire         : {memory_after_index:.0f} Mo "
          f"(+{memory_after_index - memory_before:.0f} Mo depuis le demarrage)\n")

    rows: list[dict] = []
    in_corpus = [question for question in questions if question["type"] == "in_corpus"]
    out_corpus = [question for question in questions if question["type"] == "out_of_corpus"]

    # ---- Questions DANS le corpus -----------------------------------
    print("-" * 74)
    print("QUESTIONS DANS LE CORPUS (le systeme doit repondre)")
    print("-" * 74)

    for question in in_corpus:
        answer = pipeline.answer(question["question"], top_k=top_k)

        # hit@k : un titre attendu figure-t-il dans les passages renvoyes ?
        expected = [_fold(title) for title in question.get("expected_titles", [])]
        found_titles = [_fold(passage.title) for passage in answer.passages]
        if expected:
            hit = any(
                any(exp in title or title in exp for title in found_titles)
                for exp in expected
            )
            rank = next(
                (
                    position
                    for position, title in enumerate(found_titles, start=1)
                    if any(exp in title or title in exp for exp in expected)
                ),
                0,
            )
        else:
            hit, rank = None, 0   # pas de verite terrain pour cette question

        keywords = question.get("expected_keywords", [])
        folded_answer = _fold(answer.answer)
        found_keywords = [word for word in keywords if _fold(word) in folded_answer]
        coverage = len(found_keywords) / len(keywords) if keywords else None

        refused = is_refusal(answer.answer)

        rows.append(
            {
                "id": question["id"],
                "type": "in_corpus",
                "question": question["question"],
                "hit": hit,
                "rank": rank,
                "best_score": round(answer.best_score, 4),
                "keyword_coverage": None if coverage is None else round(coverage, 2),
                "keywords_found": found_keywords,
                "refused": refused,
                "retrieval_time_s": round(answer.retrieval_time_s, 4),
                "generation_time_s": round(answer.generation_time_s, 4),
                "total_time_s": round(answer.total_time_s, 4),
                "n_sources": len(answer.sources),
                "answer_preview": answer.answer[:200],
                "llm_ok": answer.llm_ok,
            }
        )

        hit_label = "?" if hit is None else ("OUI" if hit else "NON")
        cov_label = "-" if coverage is None else f"{coverage * 100:.0f}%"
        print(f"  {question['id']}  hit@{top_k}={hit_label:<3} rang={rank}  "
              f"score={answer.best_score:.3f}  mots-cles={cov_label:<5} "
              f"{answer.total_time_s:.2f}s")
        if refused:
            print(f"      [!] Le systeme a REFUSE de repondre a une question "
                  f"du corpus : seuil MIN_SCORE={threshold} trop eleve ?")

    # ---- Questions HORS corpus --------------------------------------
    print("\n" + "-" * 74)
    print("QUESTIONS HORS CORPUS (le systeme doit REFUSER)")
    print("-" * 74)

    for question in out_corpus:
        answer = pipeline.answer(question["question"], top_k=top_k)
        refused = is_refusal(answer.answer)
        rows.append(
            {
                "id": question["id"],
                "type": "out_of_corpus",
                "question": question["question"],
                "refused": refused,
                "best_score": round(answer.best_score, 4),
                "n_sources": len(answer.sources),
                "total_time_s": round(answer.total_time_s, 4),
                "answer_preview": answer.answer[:200],
                "llm_ok": answer.llm_ok,
            }
        )
        verdict = "OK (refuse)" if refused else "ECHEC (a repondu !)"
        print(f"  {question['id']}  {verdict:<20} score={answer.best_score:.3f}")
        if not refused:
            print(f"      Reponse donnee : {answer.answer[:120]}")
            print(f"      -> Risque d'hallucination. Augmentez MIN_SCORE "
                  f"(actuel : {threshold}).")

    # ---- Agregation --------------------------------------------------
    in_rows = [row for row in rows if row["type"] == "in_corpus"]
    out_rows = [row for row in rows if row["type"] == "out_of_corpus"]

    scored = [row for row in in_rows if row["hit"] is not None]
    hit_rate = (sum(1 for row in scored if row["hit"]) / len(scored)) if scored else None
    mrr = (
        sum(1 / row["rank"] for row in scored if row["rank"] > 0) / len(scored)
        if scored else None
    )
    covered = [row["keyword_coverage"] for row in in_rows
               if row["keyword_coverage"] is not None]
    refusal_rate = (
        sum(1 for row in out_rows if row["refused"]) / len(out_rows)
        if out_rows else None
    )
    false_refusals = sum(1 for row in in_rows if row["refused"])

    times_total = [row["total_time_s"] for row in rows]
    times_retrieval = [row.get("retrieval_time_s", 0) for row in in_rows]
    times_generation = [row.get("generation_time_s", 0) for row in in_rows]
    memory_end = memory_mb()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = {
        "retriever": retriever_kind,
        "llm_provider": CFG.llm_provider,
        "embedding_model": CFG.embedding_model if retriever_kind == "embeddings" else None,
        "vector_backend": CFG.vector_backend if retriever_kind == "embeddings" else None,
        "top_k": top_k,
        "min_score": threshold,
        "n_chunks": len(chunks),
        "n_questions": len(rows),
        "index_build_s": round(build_time, 3),
        "hit_at_k": None if hit_rate is None else round(hit_rate, 3),
        "mrr": None if mrr is None else round(mrr, 3),
        "keyword_coverage_mean": round(sum(covered) / len(covered), 3) if covered else None,
        "refusal_rate_out_of_corpus": None if refusal_rate is None else round(refusal_rate, 3),
        "false_refusals_in_corpus": false_refusals,
        "mean_total_time_s": round(sum(times_total) / len(times_total), 3),
        # ATTENTION : cette moyenne melange une premiere requete A FROID et
        # des requetes A CHAUD. Mesure reelle : 94,6 s pour la premiere
        # question (chargement du modele) contre 0,19 s pour les suivantes.
        # La moyenne seule est donc trompeuse ; on publie aussi le detail.
        "mean_retrieval_time_s": (
            round(sum(times_retrieval) / len(times_retrieval), 4) if times_retrieval else None
        ),
        "first_query_s": round(times_retrieval[0], 4) if times_retrieval else None,
        "warm_retrieval_time_s": (
            round(sorted(times_retrieval[1:])[len(times_retrieval[1:]) // 2], 4)
            if len(times_retrieval) > 1 else None
        ),
        "mean_generation_time_s": (
            round(sum(times_generation) / len(times_generation), 3) if times_generation else None
        ),
        "memory_mb_end": round(memory_end, 1),
        "memory_mb_python_peak": round(peak / 1024 / 1024, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print("\n" + "=" * 74)
    print("RESULTATS MESURES")
    print("=" * 74)
    for key, value in metrics.items():
        print(f"  {key:30s} : {value}")

    print("\n  COMMENT LIRE CES CHIFFRES")
    print("  - hit_at_k proche de 1.0        : la recherche trouve les bons articles.")
    print("  - refusal_rate... egal a 1.0    : aucune hallucination sur le hors-corpus.")
    print("  - false_refusals_in_corpus > 0  : MIN_SCORE est trop severe.")
    print("  - mean_generation_time eleve    : normal avec un LLM local sur CPU.")
    print("  - first_query_s >> warm_...     : le modele se charge a la 1re requete.")
    print("    C'est warm_retrieval_time_s qui represente le regime normal ;")
    print("    mean_retrieval_time_s melange les deux et sous-estime la vitesse.")

    return {"metrics": metrics, "rows": rows}


def save_report(report: dict, out_path: Path) -> None:
    """Ecrit le rapport en JSON et en Markdown lisible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = out_path.with_suffix(".md")
    metrics = report["metrics"]
    lines = [
        "# Rapport d'evaluation - WikiNLP Assistant",
        "",
        f"Mesure du {metrics['measured_at']} sur {metrics['platform']}",
        f"(Python {metrics['python']}).",
        "",
        "## Configuration mesuree",
        "",
        "| Parametre | Valeur |",
        "| --- | --- |",
    ]
    for key in ("retriever", "llm_provider", "embedding_model", "vector_backend",
                "top_k", "min_score", "n_chunks"):
        lines.append(f"| {key} | {metrics.get(key)} |")

    lines += [
        "",
        "## Metriques",
        "",
        "| Metrique | Valeur | Lecture |",
        "| --- | --- | --- |",
        f"| hit@{metrics['top_k']} | {metrics['hit_at_k']} | 1.0 = la source attendue est toujours trouvee |",
        f"| MRR | {metrics['mrr']} | 1.0 = la bonne source arrive toujours en premier |",
        f"| Couverture mots-cles | {metrics['keyword_coverage_mean']} | part des mots attendus presents |",
        f"| Refus hors corpus | {metrics['refusal_rate_out_of_corpus']} | 1.0 = aucune hallucination |",
        f"| Faux refus dans le corpus | {metrics['false_refusals_in_corpus']} | 0 = seuil bien regle |",
        f"| Temps moyen total | {metrics['mean_total_time_s']} s | |",
        f"| Temps moyen recherche | {metrics['mean_retrieval_time_s']} s | |",
        f"| Temps moyen generation | {metrics['mean_generation_time_s']} s | |",
        f"| Memoire processus | {metrics['memory_mb_end']} Mo | approximative |",
        "",
        "## Detail par question",
        "",
        "| id | type | refus | score | hit | rang | temps (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['type']} | {row['refused']} | "
            f"{row['best_score']} | {row.get('hit')} | {row.get('rank', '-')} | "
            f"{row['total_time_s']} |"
        )

    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Rapport JSON     : {out_path}")
    print(f"  Rapport Markdown : {markdown}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluation du systeme RAG.")
    parser.add_argument("--retriever", choices=["tfidf", "embeddings"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Evalue tfidf PUIS embeddings et compare.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.compare:
        reports = {}
        for kind in ("tfidf", "embeddings"):
            try:
                reports[kind] = evaluate(kind, args.top_k)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"\n[!] Evaluation '{kind}' impossible : {exc}", file=sys.stderr)

        if len(reports) == 2:
            print("\n" + "=" * 74)
            print("COMPARAISON LEXICAL / SEMANTIQUE")
            print("=" * 74)
            keys = ["hit_at_k", "mrr", "keyword_coverage_mean",
                    "refusal_rate_out_of_corpus", "mean_retrieval_time_s",
                    "mean_total_time_s", "memory_mb_end"]
            print(f"\n  {'Metrique':<30} {'tfidf':>12} {'embeddings':>12}")
            print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")
            for key in keys:
                left = reports["tfidf"]["metrics"].get(key)
                right = reports["embeddings"]["metrics"].get(key)
                print(f"  {key:<30} {str(left):>12} {str(right):>12}")

        out = args.out or EVAL_DIR / "results_compare.json"
        save_report({"comparison": {k: v["metrics"] for k, v in reports.items()},
                     "metrics": next(iter(reports.values()))["metrics"],
                     "rows": next(iter(reports.values()))["rows"]}, out)
        return 0

    report = evaluate(args.retriever, args.top_k)
    out = args.out or EVAL_DIR / f"results_{report['metrics']['retriever']}.json"
    save_report(report, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
