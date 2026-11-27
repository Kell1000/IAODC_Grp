"""
Tests du systeme RAG (etape 8 du TP).

Ce que l'on verifie :
  - le decoupage en chunks respecte la taille et le chevauchement ;
  - la recherche remonte le BON document ;
  - le garde-fou refuse de repondre hors corpus (anti-hallucination) ;
  - les sources sont toujours presentes quand une reponse est donnee ;
  - le temps de reponse reste raisonnable.

Tout tourne hors ligne : TF-IDF + mode extractif (voir conftest.py).
"""

from __future__ import annotations

import time

import pytest

from src.rag import (
    NO_ANSWER,
    RagPipeline,
    TfidfRetriever,
    chunk_text,
)


# ==========================================================================
#  Decoupage en chunks
# ==========================================================================
class TestChunking:
    def test_texte_court_reste_entier(self):
        """Un texte plus court que `size` n'est pas decoupe.

        Le texte doit tout de meme etre de la prose : le filtre de qualite
        s'applique partout, y compris sur ce raccourci.
        """
        texte = (
            "Une phrase courte mais bien formee qui tient entierement dans "
            "un seul chunk sans avoir besoin du moindre decoupage."
        )
        assert chunk_text(texte, size=500, overlap=50) == [texte]

    def test_texte_long_est_decoupe(self):
        texte = ("Ceci est une phrase de test. " * 60).strip()
        morceaux = chunk_text(texte, size=300, overlap=50)
        assert len(morceaux) > 1

    def test_respecte_la_borne_taille_plus_chevauchement(self):
        """Borne reelle d'un chunk : size + overlap.

        Un chunk commence par le report de chevauchement (jusqu'a `overlap`
        caracteres), puis on y ajoute des phrases tant que le total reste
        sous `size`. Dans le pire des cas - une phrase de longueur `size`
        juste apres un report complet - on atteint size + overlap.
        C'est cette borne qui est garantie, pas `size` seul : savoir la
        calculer evite de dimensionner trop juste le contexte du LLM.
        """
        size, overlap = 300, 50
        texte = ("Phrase de longueur moyenne pour le test. " * 60).strip()
        for morceau in chunk_text(texte, size=size, overlap=overlap):
            assert len(morceau) <= size + overlap

    def test_le_chevauchement_cree_un_recouvrement(self):
        """Deux chunks consecutifs doivent partager du texte."""
        phrases = [f"Phrase numero {i} du document de test." for i in range(40)]
        texte = " ".join(phrases)
        morceaux = chunk_text(texte, size=250, overlap=100)
        assert len(morceaux) >= 2

        # On cherche un mot de la fin du chunk 0 dans le debut du chunk 1.
        fin = morceaux[0].split()[-5:]
        assert any(mot in morceaux[1] for mot in fin), (
            "Aucun recouvrement detecte : le chevauchement ne fonctionne pas."
        )

    def test_phrase_plus_longue_que_la_taille_est_coupee(self):
        """Un tableau sans ponctuation ne doit pas faire exploser un chunk."""
        morceaux = chunk_text("mot " * 300, size=200, overlap=20)
        assert all(len(morceau) <= 200 + 10 for morceau in morceaux)

    @pytest.mark.parametrize("entree", ["", "   ", "\n"])
    def test_texte_vide(self, entree):
        assert chunk_text(entree, size=100, overlap=10) == []

    def test_jette_les_fragments_minuscules(self):
        """Un chunk de 10 caracteres n'a aucune valeur informative."""
        assert all(len(morceau) > 50 for morceau in chunk_text("ok. " * 40, 200, 20))

    def test_jette_les_fragments_de_formules(self):
        """Les restes de formules mathematiques ne doivent pas etre indexes.

        Cas reel observe dans l'interface : le passage le mieux classe pour
        une question sur l'apprentissage profond etait "a. x." - un fragment
        de formule. TF-IDF favorise ces chunks courts, qui n'apprennent rien.
        """
        formules = "a. x. f(x). i=1. n. y. z. w(i). 2. 3. b. c. d. e."
        assert chunk_text(formules, size=300, overlap=50) == []

    def test_garde_la_prose(self):
        """A l'inverse, un paragraphe normal doit evidemment etre conserve."""
        prose = (
            "L'apprentissage profond est un ensemble de methodes fondees sur "
            "des reseaux de neurones artificiels comportant de nombreuses "
            "couches successives."
        )
        assert chunk_text(prose, size=300, overlap=50) == [prose]

    def test_produit_des_chunks_depuis_le_corpus(self, chunks):
        assert len(chunks) >= 3
        for chunk in chunks:
            for champ in ("id", "text", "norm_text", "title", "url"):
                assert champ in chunk
            assert chunk["text"].strip()


# ==========================================================================
#  Recherche TF-IDF
# ==========================================================================
class TestTfidfRetriever:
    def test_trouve_le_bon_document(self, chunks):
        """Une question sur le TAL doit remonter l'article sur le TAL."""
        hits = TfidfRetriever(chunks).search(
            "Qu'est-ce que le traitement automatique des langues ?", top_k=3
        )
        assert hits
        assert "Traitement automatique" in hits[0].title

    def test_distingue_deux_sujets_eloignes(self, chunks):
        """Une question de cuisine ne doit pas remonter un article de NLP."""
        hits = TfidfRetriever(chunks).search(
            "Quels sont les ingredients de la pizza napolitaine ?", top_k=2
        )
        assert hits
        assert "Cuisine" in hits[0].title

    def test_les_scores_sont_decroissants(self, chunks):
        hits = TfidfRetriever(chunks).search("reseaux de neurones", top_k=4)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_respecte_top_k(self, chunks):
        assert len(TfidfRetriever(chunks).search("langage", top_k=2)) == 2

    def test_les_metadonnees_de_source_sont_presentes(self, chunks):
        hit = TfidfRetriever(chunks).search("apprentissage profond", top_k=1)[0]
        assert hit.title
        assert hit.url.startswith("http")
        assert hit.chunk_id

    def test_question_sans_mot_utile_ne_plante_pas(self, chunks):
        """Une requete faite uniquement de stop words devient vide."""
        assert TfidfRetriever(chunks).search("le la de et", top_k=3) == []

    def test_score_faible_pour_question_hors_corpus(self, chunks):
        hits = TfidfRetriever(chunks).search(
            "Quel est le cours de l'action Tesla aujourd'hui ?", top_k=1
        )
        if hits:
            assert hits[0].score < 0.25, (
                "Un sujet absent du corpus obtient un score eleve : "
                "le garde-fou ne pourra pas le filtrer."
            )


# ==========================================================================
#  Pipeline RAG : garde-fou et sources
# ==========================================================================
class TestRagPipeline:
    def test_repond_a_une_question_du_corpus(self, pipeline):
        answer = pipeline.answer(
            "Qu'est-ce que le traitement automatique des langues ?",
            top_k=3, min_score=0.05,
        )
        assert answer.has_context
        assert answer.answer.strip()
        assert answer.answer != NO_ANSWER

    def test_cite_toujours_ses_sources_quand_il_repond(self, pipeline):
        """Regle du TP : pas de reponse sans source verifiable."""
        answer = pipeline.answer("apprentissage profond", top_k=3, min_score=0.05)
        assert answer.has_context
        assert answer.sources
        for source in answer.sources:
            assert source["url"].startswith("http")
            assert source["title"]
            assert source["license"] == "CC BY-SA 4.0"

    def test_refuse_de_repondre_hors_corpus(self, pipeline):
        """LE test le plus important : ne pas inventer."""
        answer = pipeline.answer(
            "Quel est le resultat du match de football d'hier soir ?",
            top_k=3, min_score=0.25,
        )
        assert not answer.has_context
        assert answer.answer == NO_ANSWER
        assert answer.sources == [], "Aucune source ne doit etre citee sans contexte."

    def test_le_seuil_eleve_declenche_le_garde_fou(self, pipeline):
        """Avec min_score=0.99, meme une bonne question doit etre refusee."""
        answer = pipeline.answer("traitement automatique des langues",
                                 top_k=3, min_score=0.99)
        assert not answer.has_context

    def test_le_seuil_bas_laisse_passer(self, pipeline):
        answer = pipeline.answer("traitement automatique des langues",
                                 top_k=3, min_score=0.0)
        assert answer.has_context

    def test_question_vide(self, pipeline):
        answer = pipeline.answer("")
        assert not answer.has_context
        assert answer.sources == []

    def test_les_sources_sont_dedupliquees(self, pipeline):
        """Deux chunks du meme article = UNE seule source affichee."""
        answer = pipeline.answer("traitement automatique des langues",
                                 top_k=5, min_score=0.0)
        urls = [source["url"] for source in answer.sources]
        assert len(urls) == len(set(urls))

    def test_le_contexte_est_numerote(self, pipeline):
        """La numerotation [1], [2] permet au LLM de citer ses sources."""
        hits = pipeline.retriever.search("langage", top_k=2)
        contexte = pipeline.build_context(hits)
        assert "[1]" in contexte
        assert "[2]" in contexte
        assert "Titre :" in contexte

    def test_les_temps_sont_mesures(self, pipeline):
        answer = pipeline.answer("apprentissage profond", top_k=2, min_score=0.0)
        assert answer.retrieval_time_s > 0
        assert answer.total_time_s >= answer.retrieval_time_s

    def test_le_temps_de_reponse_reste_raisonnable(self, pipeline):
        """En mode TF-IDF + extractif, une reponse doit etre quasi immediate."""
        start = time.perf_counter()
        pipeline.answer("traitement automatique des langues", top_k=3, min_score=0.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Reponse trop lente hors ligne : {elapsed:.2f}s"


# ==========================================================================
#  Mode extractif : aucun reseau, aucune invention
# ==========================================================================
class TestModeExtractif:
    def test_le_mode_extractif_ne_sort_pas_du_contexte(self, pipeline):
        """Le texte renvoye doit provenir des passages, pas d'ailleurs.

        On verifie qu'au moins un fragment significatif de la reponse se
        retrouve reellement dans le contexte transmis : c'est la definition
        operationnelle de la fidelite aux sources en mode extractif.
        """
        answer = pipeline.answer("apprentissage profond", top_k=2, min_score=0.0)
        contexte = " ".join(passage.text for passage in answer.passages).lower()
        mots_reponse = [
            mot for mot in answer.answer.lower().split()
            if len(mot) > 7 and mot.isalpha()
        ]
        communs = [mot for mot in mots_reponse if mot in contexte]
        assert communs, "La reponse extractive ne reprend rien du contexte."

    def test_contexte_vide_donne_un_refus(self):
        from src.llm import generate_extractive

        result = generate_extractive("une question", "")
        assert "ne trouve pas" in result.text.lower()

    def test_aucune_cle_api_necessaire(self):
        from src.llm import generate_extractive

        result = generate_extractive("question", "[1] Titre : X\nUn contenu de test.")
        assert result.ok
        assert result.provider == "extractive"
        assert result.latency_s >= 0
