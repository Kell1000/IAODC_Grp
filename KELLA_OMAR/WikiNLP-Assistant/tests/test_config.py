"""
Tests de la configuration.

Deux objectifs :
  1. Verifier que les garde-fous rejettent une configuration invalide AVANT
     que l'application ne se comporte bizarrement.
  2. Verifier qu'aucune cle API ne fuite dans les affichages ou les logs :
     c'est une exigence de securite du TP.
"""

from __future__ import annotations

import pytest

from src.config import Config, ROOT_DIR


class TestValidation:
    def test_configuration_par_defaut_valide(self):
        assert Config().top_k >= 1

    def test_retriever_invalide_rejete(self, monkeypatch):
        monkeypatch.setenv("RETRIEVER", "magique")
        with pytest.raises(ValueError) as info:
            Config()
        assert "RETRIEVER" in str(info.value)

    def test_fournisseur_invalide_rejete(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "chatgpt5000")
        with pytest.raises(ValueError) as info:
            Config()
        assert "LLM_PROVIDER" in str(info.value)

    def test_backend_vectoriel_invalide_rejete(self, monkeypatch):
        monkeypatch.setenv("VECTOR_BACKEND", "oracle")
        with pytest.raises(ValueError):
            Config()

    def test_chevauchement_superieur_a_la_taille_rejete(self, monkeypatch):
        """Ce reglage ferait boucler le decoupage a l'infini."""
        monkeypatch.setenv("CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        with pytest.raises(ValueError) as info:
            Config()
        assert "CHUNK_OVERLAP" in str(info.value)

    def test_top_k_nul_rejete(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "0")
        with pytest.raises(ValueError):
            Config()


class TestLectureEnvironnement:
    def test_variable_vide_retombe_sur_le_defaut(self, monkeypatch):
        """Piege classique : TOP_K= (vide) ne doit pas casser int()."""
        monkeypatch.setenv("TOP_K", "")
        assert Config().top_k == 4

    def test_valeur_non_numerique_ne_plante_pas(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "beaucoup")
        assert Config().chunk_size == 900

    def test_les_valeurs_sont_bien_lues(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "7")
        monkeypatch.setenv("MIN_SCORE", "0.42")
        monkeypatch.setenv("WIKI_LANG", "en")
        config = Config()
        assert config.top_k == 7
        assert config.min_score == 0.42
        assert config.wiki_lang == "en"


class TestSeuilParMoteur:
    """Les scores TF-IDF et semantiques ne sont pas sur la meme echelle.

    Un seuil unique trop haut refuse toutes les questions legitimes : c'est
    le defaut qui a ete mesure puis corrige pendant le TP.
    """

    def test_seuils_par_defaut_distincts(self, monkeypatch):
        monkeypatch.delenv("MIN_SCORE", raising=False)
        config = Config()
        assert config.min_score_for("tfidf") < config.min_score_for("embeddings")

    def test_le_seuil_tfidf_laisse_passer_les_scores_observes(self, monkeypatch):
        """Mesures reelles : dans le corpus 0,151-0,205 ; hors corpus 0,081-0,121."""
        monkeypatch.delenv("MIN_SCORE", raising=False)
        seuil = Config().min_score_for("tfidf")
        assert seuil < 0.140, "le seuil refuserait une question legitime"
        assert seuil > 0.122, "le seuil accepterait une question hors corpus"

    def test_le_seuil_embeddings_laisse_passer_les_scores_observes(self, monkeypatch):
        """Mesures reelles : hors corpus 0,186-0,365 ; dans le corpus 0,495-0,897.

        L'estimation initiale de 0,30 tombait DANS la population hors corpus :
        une question hors sujet passait au score 0,3652 et
        refusal_rate_out_of_corpus valait 0,667 au lieu de 1,0.
        """
        monkeypatch.delenv("MIN_SCORE", raising=False)
        seuil = Config().min_score_for("embeddings")
        assert seuil > 0.366, "le seuil accepterait une question hors corpus"
        assert seuil < 0.494, "le seuil refuserait une question legitime"

    def test_min_score_suit_le_retriever_configure(self, monkeypatch):
        monkeypatch.delenv("MIN_SCORE", raising=False)
        monkeypatch.setenv("RETRIEVER", "tfidf")
        assert Config().min_score == Config().min_score_for("tfidf")
        monkeypatch.setenv("RETRIEVER", "embeddings")
        assert Config().min_score == Config().min_score_for("embeddings")

    def test_min_score_force_les_deux_moteurs(self, monkeypatch):
        monkeypatch.setenv("MIN_SCORE", "0.5")
        config = Config()
        assert config.min_score_for("tfidf") == 0.5
        assert config.min_score_for("embeddings") == 0.5

    def test_min_score_vide_revient_a_l_automatique(self, monkeypatch):
        monkeypatch.setenv("MIN_SCORE", "")
        config = Config()
        assert config.min_score_for("tfidf") == config.min_score_tfidf

    def test_min_score_non_numerique_rejete(self, monkeypatch):
        monkeypatch.setenv("MIN_SCORE", "beaucoup")
        with pytest.raises(ValueError) as info:
            Config()
        assert "MIN_SCORE" in str(info.value)


class TestSecuriteDesCles:
    def test_les_cles_sont_masquees_dans_le_resume(self, monkeypatch):
        """Le resume est affiche et logue : il ne doit jamais exposer la cle."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_tres_long_1234")
        resume = Config().summary()
        assert "gsk_secret_tres_long_1234" not in str(resume)
        assert resume["groq_api_key"].startswith("***")

    def test_cle_absente_affichee_comme_vide(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        assert Config().summary()["gemini_api_key"] == "(vide)"

    def test_le_fichier_env_n_est_pas_versionne(self):
        """Verification concrete : .gitignore protege bien .env."""
        gitignore = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
        lignes = [ligne.strip() for ligne in gitignore.splitlines()]
        assert ".env" in lignes, ".env doit figurer dans .gitignore"

    def test_env_example_ne_contient_aucune_vraie_cle(self):
        """.env.example est public : il ne doit contenir que des champs vides."""
        exemple = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
        for ligne in exemple.splitlines():
            ligne = ligne.strip()
            if ligne.startswith(("GROQ_API_KEY", "GEMINI_API_KEY")):
                cle, _, valeur = ligne.partition("=")
                assert valeur.strip() == "", (
                    f"{cle} contient une valeur dans .env.example : "
                    "ne commitez jamais une cle !"
                )


class TestEtatLLM:
    def test_mode_extractif_toujours_pret(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "extractive")
        ready, message = Config().llm_is_ready()
        assert ready
        assert "aucune cle" in message.lower()

    def test_groq_sans_cle_non_pret(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "")
        ready, message = Config().llm_is_ready()
        assert not ready
        assert "GROQ_API_KEY" in message

    def test_groq_avec_cle_pret(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        ready, _ = Config().llm_is_ready()
        assert ready
