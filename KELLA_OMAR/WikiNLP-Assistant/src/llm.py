"""
Etape 4 (experience C) du TP - Couche d'abstraction sur les modeles de langue.

Quatre fournisseurs interchangeables, choisis par LLM_PROVIDER dans .env.
Aucun n'est obligatoire : l'application fonctionne toujours, meme sans cle.

  extractive  AUCUNE cle, AUCUN reseau. Ne "genere" rien : renvoie les
              meilleures phrases des passages trouves. C'est le mode
              demonstration, et la reference pour juger l'apport d'un LLM.

  groq        API gratuite (console.groq.com), tres rapide. Quotas limites.
              VERIFIEZ vos limites actuelles : console.groq.com/settings/limits
              Les quotas changent, aucune valeur n'est garantie dans le temps.

  gemini      API gratuite (aistudio.google.com/apikey).
              ATTENTION VIE PRIVEE : sur l'offre gratuite, Google indique
              pouvoir utiliser vos requetes pour ameliorer ses modeles, avec
              relecture humaine possible. N'y envoyez aucune donnee sensible.

  ollama      LLM local (ollama.com), aucune cle, aucune limite, fonctionne
              hors ligne. Un modele 1,5 B quantifie consomme ~1,5 Go de RAM :
              tenable sur 16 Go. Plus lent qu'une API sur CPU.

POURQUOI passer par requests plutot que par les SDK officiels ?
Chaque SDK (groq, google-genai, openai) ajoute des dependances et des
conflits de versions. Ici, trois appels HTTP suffisent : le projet reste
leger et aucun fournisseur ne devient structurellement obligatoire.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from src.config import CFG

# Consigne systeme commune. C'est la piece la plus importante contre
# l'hallucination : on interdit explicitement d'inventer, et on impose de
# dire "je ne sais pas" quand le contexte ne contient pas la reponse.
SYSTEM_PROMPT = (
    "Tu es un assistant documentaire qui repond EXCLUSIVEMENT a partir des "
    "extraits Wikipedia fournis.\n"
    "Regles imperatives :\n"
    "1. N'utilise QUE les informations presentes dans les extraits.\n"
    "2. Si les extraits ne contiennent pas la reponse, reponds exactement : "
    "\"Je ne trouve pas cette information dans les documents fournis.\"\n"
    "3. N'invente jamais de fait, de date, de chiffre ni de source.\n"
    "4. Cite les numeros des extraits utilises, au format [1], [2].\n"
    "5. Reponds en francais, de maniere concise (3 a 6 phrases)."
)


@dataclass
class LLMResult:
    """Resultat d'un appel LLM, succes ou echec."""

    text: str
    provider: str
    model: str
    latency_s: float
    ok: bool = True
    error: str = ""
    usage: dict | None = None


class LLMError(RuntimeError):
    """Erreur d'appel au modele de langue, avec un message exploitable."""


# ==========================================================================
#  Traduction des erreurs HTTP en messages comprehensibles
# ==========================================================================
def _explain_http_error(status: int, provider: str, body: str) -> str:
    """Transforme un code HTTP en consigne actionnable pour l'etudiant."""
    extract = body[:300].replace("\n", " ")
    if status in (401, 403):
        return (
            f"Cle API refusee par {provider} (HTTP {status}).\n"
            f"  - Verifiez la valeur dans .env (pas d'espace, pas de guillemets).\n"
            f"  - Verifiez que la cle n'a pas ete revoquee ou regeneree.\n"
            f"  Reponse du serveur : {extract}"
        )
    if status == 429:
        return (
            f"Quota depasse chez {provider} (HTTP 429).\n"
            f"  - Attendez quelques minutes : les quotas sont par minute ET par jour.\n"
            f"  - Ou basculez en local : LLM_PROVIDER=ollama\n"
            f"  - Ou en mode sans LLM : LLM_PROVIDER=extractive\n"
            f"  Reponse du serveur : {extract}"
        )
    if status == 404:
        return (
            f"Modele introuvable chez {provider} (HTTP 404).\n"
            f"  Le nom du modele a probablement change ou a ete retire.\n"
            f"  Consultez la liste des modeles du fournisseur et corrigez .env.\n"
            f"  Reponse du serveur : {extract}"
        )
    if status >= 500:
        return (
            f"Panne cote {provider} (HTTP {status}). Reessayez plus tard, "
            f"ou basculez sur LLM_PROVIDER=extractive.\n  Reponse : {extract}"
        )
    return f"Erreur {provider} HTTP {status} : {extract}"


def _post_json(url: str, provider: str, timeout: int, **kwargs) -> dict:
    """POST JSON avec gestion homogene des erreurs reseau."""
    try:
        response = requests.post(url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectTimeout as exc:
        raise LLMError(
            f"Delai de connexion depasse vers {provider}. Verifiez votre reseau."
        ) from exc
    except requests.exceptions.ReadTimeout as exc:
        raise LLMError(
            f"{provider} n'a pas repondu en {timeout}s. Augmentez LLM_TIMEOUT "
            "dans .env, ou reduisez LLM_MAX_TOKENS."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise LLMError(
            f"Connexion impossible vers {provider}.\n"
            "  - Si provider=ollama : Ollama est-il demarre ? (ollama serve)\n"
            "  - Sinon : verifiez votre connexion Internet / proxy."
        ) from exc

    if response.status_code != 200:
        raise LLMError(_explain_http_error(response.status_code, provider, response.text))

    try:
        return response.json()
    except ValueError as exc:
        raise LLMError(f"Reponse non JSON de {provider} : {response.text[:200]}") from exc


# ==========================================================================
#  Fournisseur 1 : mode extractif (sans LLM)
# ==========================================================================
def generate_extractive(question: str, context: str) -> LLMResult:
    """Construit une reponse SANS modele generatif.

    On ne fait que remonter les premieres phrases des passages retrouves.
    Interet pedagogique : c'est la reference ("baseline"). En comparant ce
    mode au mode groq/gemini, on voit exactement ce que le LLM apporte -
    la reformulation et la synthese - et ce qu'il n'apporte pas : le contenu,
    qui vient entierement de la recherche documentaire.
    """
    start = time.perf_counter()

    if not context.strip():
        text = "Je ne trouve pas cette information dans les documents fournis."
    else:
        # Garde les 2 premieres phrases de chaque extrait, dans l'ordre.
        pieces: list[str] = []
        for block in context.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            sentences = [s.strip() for s in block.replace("\n", " ").split(". ") if s.strip()]
            if sentences:
                summary = ". ".join(sentences[:2])
                if not summary.endswith("."):
                    summary += "."
                pieces.append(summary)

        text = (
            "Mode extractif (aucun LLM configure). Voici les passages les plus "
            "pertinents trouves dans le corpus :\n\n"
            + "\n\n".join(f"- {piece}" for piece in pieces[:4])
        )

    return LLMResult(
        text=text,
        provider="extractive",
        model="aucun (extraction de phrases)",
        latency_s=time.perf_counter() - start,
    )


# ==========================================================================
#  Fournisseur 2 : Groq (API compatible OpenAI)
# ==========================================================================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_groq(question: str, context: str) -> LLMResult:
    if not CFG.groq_api_key:
        raise LLMError(
            "GROQ_API_KEY est vide.\n"
            "  1. Creez une cle gratuite sur https://console.groq.com/keys\n"
            "  2. Collez-la dans .env :  GROQ_API_KEY=gsk_...\n"
            "  3. Relancez l'application."
        )

    start = time.perf_counter()
    payload = _post_json(
        GROQ_URL,
        provider="Groq",
        timeout=CFG.llm_timeout,
        headers={
            "Authorization": f"Bearer {CFG.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": CFG.groq_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question, context)},
            ],
            "temperature": CFG.llm_temperature,
            "max_tokens": CFG.llm_max_tokens,
        },
    )

    try:
        text = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Reponse Groq inattendue : {str(payload)[:300]}") from exc

    return LLMResult(
        text=text,
        provider="groq",
        model=CFG.groq_model,
        latency_s=time.perf_counter() - start,
        usage=payload.get("usage"),
    )


# ==========================================================================
#  Fournisseur 3 : Google Gemini
# ==========================================================================
def generate_gemini(question: str, context: str) -> LLMResult:
    if not CFG.gemini_api_key:
        raise LLMError(
            "GEMINI_API_KEY est vide.\n"
            "  1. Creez une cle gratuite sur https://aistudio.google.com/apikey\n"
            "  2. Collez-la dans .env :  GEMINI_API_KEY=...\n"
            "  3. Relancez l'application."
        )

    start = time.perf_counter()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{CFG.gemini_model}:generateContent"
    )
    payload = _post_json(
        url,
        provider="Gemini",
        timeout=CFG.llm_timeout,
        headers={
            "x-goog-api-key": CFG.gemini_api_key,
            "Content-Type": "application/json",
        },
        json={
            # Gemini gere la consigne systeme a part du message utilisateur.
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": _user_prompt(question, context)}]}
            ],
            "generationConfig": {
                "temperature": CFG.llm_temperature,
                "maxOutputTokens": CFG.llm_max_tokens,
            },
        },
    )

    try:
        candidate = payload["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            # Cas reel : reponse bloquee par les filtres de securite, ou
            # budget de tokens epuise avant d'avoir produit du texte.
            reason = candidate.get("finishReason", "inconnue")
            raise LLMError(
                f"Gemini n'a renvoye aucun texte (finishReason={reason}). "
                "Augmentez LLM_MAX_TOKENS ou reformulez la question."
            )
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Reponse Gemini inattendue : {str(payload)[:300]}") from exc

    return LLMResult(
        text=text,
        provider="gemini",
        model=CFG.gemini_model,
        latency_s=time.perf_counter() - start,
        usage=payload.get("usageMetadata"),
    )


# ==========================================================================
#  Fournisseur 4 : Ollama (local)
# ==========================================================================
def generate_ollama(question: str, context: str) -> LLMResult:
    start = time.perf_counter()
    payload = _post_json(
        f"{CFG.ollama_host.rstrip('/')}/api/chat",
        provider="Ollama",
        timeout=max(CFG.llm_timeout, 120),  # sur CPU, un LLM local est lent
        json={
            "model": CFG.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question, context)},
            ],
            "stream": False,
            "options": {
                "temperature": CFG.llm_temperature,
                "num_predict": CFG.llm_max_tokens,
            },
        },
    )

    text = payload.get("message", {}).get("content", "").strip()
    if not text:
        raise LLMError(
            f"Ollama n'a rien renvoye. Le modele '{CFG.ollama_model}' est-il "
            f"telecharge ?  ollama pull {CFG.ollama_model}"
        )

    return LLMResult(
        text=text,
        provider="ollama",
        model=CFG.ollama_model,
        latency_s=time.perf_counter() - start,
        usage={
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "eval_count": payload.get("eval_count"),
        },
    )


# ==========================================================================
#  Construction du prompt et aiguillage
# ==========================================================================
def _user_prompt(question: str, context: str) -> str:
    """Assemble le message utilisateur : extraits numerotes, puis question.

    L'ordre compte : le contexte AVANT la question. Les modeles suivent mieux
    la consigne quand la question arrive en dernier, juste avant la reponse.
    """
    if not context.strip():
        return (
            f"Aucun extrait pertinent n'a ete trouve dans le corpus.\n\n"
            f"Question : {question}\n\n"
            "Reponds exactement : \"Je ne trouve pas cette information dans "
            "les documents fournis.\""
        )
    return (
        f"Extraits Wikipedia :\n\n{context}\n\n"
        f"---\nQuestion : {question}\n\n"
        "Reponds en te basant uniquement sur les extraits ci-dessus, "
        "et cite les numeros utilises."
    )


_PROVIDERS = {
    "extractive": generate_extractive,
    "groq": generate_groq,
    "gemini": generate_gemini,
    "ollama": generate_ollama,
}


def generate(
    question: str,
    context: str,
    provider: str | None = None,
    fallback_to_extractive: bool = True,
) -> LLMResult:
    """Point d'entree unique utilise par le RAG et par l'interface web.

    Si le fournisseur distant echoue (quota, panne, cle invalide) et que
    fallback_to_extractive est vrai, on retombe sur le mode extractif plutot
    que de montrer une erreur brute a l'utilisateur. L'erreur reste visible
    dans le champ `error` : on n'efface jamais le diagnostic.
    """
    provider = (provider or CFG.llm_provider).lower()
    function = _PROVIDERS.get(provider)
    if function is None:
        raise LLMError(
            f"LLM_PROVIDER inconnu : '{provider}'. "
            f"Valeurs possibles : {sorted(_PROVIDERS)}"
        )

    try:
        return function(question, context)
    except LLMError as exc:
        if provider != "extractive" and fallback_to_extractive:
            result = generate_extractive(question, context)
            result.ok = False
            result.error = str(exc)
            result.provider = f"extractive (repli apres echec de {provider})"
            return result
        raise


if __name__ == "__main__":
    # Demonstration :  python -m src.llm
    print("=== Test du fournisseur LLM ===")
    ready, message = CFG.llm_is_ready()
    print(f"  Fournisseur : {CFG.llm_provider}")
    print(f"  Etat        : {message}\n")

    demo_context = (
        "[1] Titre : Traitement automatique des langues\n"
        "Le traitement automatique des langues est un domaine a la croisee de "
        "la linguistique et de l'informatique. Il vise a construire des "
        "programmes capables de traiter le langage humain.\n\n"
        "[2] Titre : Transformeur\n"
        "L'architecture Transformer a ete introduite en 2017 dans l'article "
        "Attention Is All You Need. Elle repose sur un mecanisme d'attention."
    )
    demo_question = "Qu'est-ce que le traitement automatique des langues ?"

    result = generate(demo_question, demo_context)
    print(f"  Question : {demo_question}")
    print(f"  Reponse  :\n{result.text}\n")
    print(f"  Fournisseur utilise : {result.provider}")
    print(f"  Modele              : {result.model}")
    print(f"  Latence             : {result.latency_s:.2f}s")
    if result.usage:
        print(f"  Consommation        : {result.usage}")
    if not result.ok:
        print(f"\n  [!] Echec du fournisseur demande :\n{result.error}")

    # Test anti-hallucination : sans contexte, le systeme doit refuser.
    print("\n=== Test anti-hallucination (contexte vide) ===")
    empty = generate("Quelle est la capitale du Burkina Faso ?", "")
    print(f"  Reponse : {empty.text[:200]}")
