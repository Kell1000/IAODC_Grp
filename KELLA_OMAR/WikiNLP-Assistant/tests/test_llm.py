"""
Tests de la couche LLM, SANS appel reseau reel.

On remplace requests.post par une fausse fonction (monkeypatch). On peut
ainsi verifier la gestion des erreurs - quota depasse, cle invalide, panne
serveur - sans cle API, sans Internet, et sans consommer de quota.

C'est une pratique professionnelle importante : un test qui appelle une vraie
API est lent, instable et coute de l'argent.
"""

from __future__ import annotations

import pytest
import requests

from src import llm
from src.llm import LLMError, generate, generate_extractive


class FakeResponse:
    """Imite juste ce que notre code utilise d'une reponse requests."""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(self._payload)

    def json(self) -> dict:
        return self._payload


# ==========================================================================
#  Construction du prompt
# ==========================================================================
class TestPrompt:
    def test_la_consigne_interdit_d_inventer(self):
        prompt = llm.SYSTEM_PROMPT.lower()
        assert "n'invente" in prompt or "invente jamais" in prompt
        assert "je ne trouve pas cette information" in prompt

    def test_le_contexte_precede_la_question(self):
        prompt = llm._user_prompt("Ma question ?", "[1] Titre : X\nUn contenu.")
        assert prompt.index("Un contenu") < prompt.index("Ma question ?")

    def test_contexte_vide_demande_un_refus(self):
        prompt = llm._user_prompt("Ma question ?", "")
        assert "Aucun extrait pertinent" in prompt
        assert "ne trouve pas" in prompt.lower()


# ==========================================================================
#  Mode extractif
# ==========================================================================
class TestExtractif:
    def test_fonctionne_sans_cle(self):
        result = generate_extractive("q", "[1] Titre : A\nPremiere phrase. Deuxieme.")
        assert result.ok
        assert result.provider == "extractive"

    def test_refuse_sans_contexte(self):
        assert "ne trouve pas" in generate_extractive("q", "").text.lower()


# ==========================================================================
#  Traduction des erreurs HTTP en messages utiles
# ==========================================================================
class TestMessagesErreur:
    @pytest.mark.parametrize("status", [401, 403])
    def test_cle_invalide(self, status):
        message = llm._explain_http_error(status, "Groq", "invalid api key")
        assert "Cle API refusee" in message
        assert ".env" in message

    def test_quota_depasse_propose_des_solutions(self):
        message = llm._explain_http_error(429, "Groq", "rate limit")
        assert "Quota depasse" in message
        assert "ollama" in message
        assert "extractive" in message

    def test_modele_introuvable(self):
        message = llm._explain_http_error(404, "Groq", "model not found")
        assert "Modele introuvable" in message

    def test_panne_serveur(self):
        assert "Panne" in llm._explain_http_error(503, "Gemini", "unavailable")


# ==========================================================================
#  Groq (simule)
# ==========================================================================
class TestGroq:
    def test_sans_cle_leve_une_erreur_explicite(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "")
        with pytest.raises(LLMError) as info:
            llm.generate_groq("q", "c")
        assert "GROQ_API_KEY" in str(info.value)
        assert "console.groq.com" in str(info.value)

    def test_appel_reussi(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": "Une reponse [1]."}}],
                    "usage": {"total_tokens": 42},
                },
            ),
        )
        result = llm.generate_groq("q", "[1] Titre : X\ncontenu")
        assert result.ok
        assert result.text == "Une reponse [1]."
        assert result.usage["total_tokens"] == 42

    def test_quota_depasse(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeResponse(429, {}, "rate limit")
        )
        with pytest.raises(LLMError) as info:
            llm.generate_groq("q", "c")
        assert "Quota depasse" in str(info.value)

    def test_reponse_mal_formee(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeResponse(200, {"inattendu": True})
        )
        with pytest.raises(LLMError):
            llm.generate_groq("q", "c")

    def test_reseau_coupe(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")

        def boom(*args, **kwargs):
            raise requests.exceptions.ConnectionError("pas de reseau")

        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(LLMError) as info:
            llm.generate_groq("q", "c")
        assert "Connexion impossible" in str(info.value)

    def test_delai_depasse(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")

        def boom(*args, **kwargs):
            raise requests.exceptions.ReadTimeout("trop lent")

        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(LLMError) as info:
            llm.generate_groq("q", "c")
        assert "LLM_TIMEOUT" in str(info.value)


# ==========================================================================
#  Gemini (simule)
# ==========================================================================
class TestGemini:
    def test_sans_cle(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "gemini_api_key", "")
        with pytest.raises(LLMError) as info:
            llm.generate_gemini("q", "c")
        assert "GEMINI_API_KEY" in str(info.value)

    def test_appel_reussi(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "gemini_api_key", "fausse_cle")
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(
                200,
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "Reponse Gemini [1]."}]}}
                    ],
                    "usageMetadata": {"totalTokenCount": 10},
                },
            ),
        )
        result = llm.generate_gemini("q", "c")
        assert result.text == "Reponse Gemini [1]."

    def test_reponse_bloquee_sans_texte(self, monkeypatch):
        """Cas reel : filtre de securite ou budget de tokens epuise."""
        monkeypatch.setattr(llm.CFG, "gemini_api_key", "fausse_cle")
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(
                200, {"candidates": [{"finishReason": "SAFETY", "content": {}}]}
            ),
        )
        with pytest.raises(LLMError) as info:
            llm.generate_gemini("q", "c")
        assert "aucun texte" in str(info.value)


# ==========================================================================
#  Ollama (simule)
# ==========================================================================
class TestOllama:
    def test_appel_reussi(self, monkeypatch):
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(
                200, {"message": {"content": "Reponse locale."}, "eval_count": 7}
            ),
        )
        assert llm.generate_ollama("q", "c").text == "Reponse locale."

    def test_serveur_eteint_message_clair(self, monkeypatch):
        def boom(*args, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(LLMError) as info:
            llm.generate_ollama("q", "c")
        assert "ollama serve" in str(info.value)


# ==========================================================================
#  Aiguillage et repli automatique
# ==========================================================================
class TestAiguillage:
    def test_fournisseur_inconnu(self):
        with pytest.raises(LLMError):
            generate("q", "c", provider="inexistant")

    def test_repli_extractif_quand_l_api_echoue(self, monkeypatch):
        """Comportement attendu de l'UI : degrader, pas planter."""
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeResponse(429, {}, "rate limit")
        )
        result = generate("q", "[1] Titre : X\nUn contenu de test.",
                          provider="groq", fallback_to_extractive=True)
        assert not result.ok                    # l'echec est signale...
        assert "Quota depasse" in result.error  # ...et diagnostique...
        assert result.text                      # ...mais une reponse existe.
        assert "extractive" in result.provider

    def test_sans_repli_l_erreur_remonte(self, monkeypatch):
        monkeypatch.setattr(llm.CFG, "groq_api_key", "gsk_fausse_cle")
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeResponse(429, {}, "rate limit")
        )
        with pytest.raises(LLMError):
            generate("q", "c", provider="groq", fallback_to_extractive=False)
