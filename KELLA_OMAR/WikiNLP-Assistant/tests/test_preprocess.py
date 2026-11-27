"""
Tests du nettoyage et du pretraitement (etape 8 du TP).

Ce que l'on verifie ici, c'est la QUALITE DU NETTOYAGE demandee par le TP :
les artefacts disparaissent, le sens reste, les doublons sont retires, et
les deux representations (texte pour le LLM / tokens pour TF-IDF) sont bien
differentes.
"""

from __future__ import annotations

import pytest

from src.preprocess import (
    clean_text,
    content_fingerprint,
    deduplicate,
    normalize_tokens,
    process_record,
    split_sentences,
    strip_accents,
    tokenize,
)


# ==========================================================================
#  Nettoyage
# ==========================================================================
class TestCleanText:
    def test_supprime_les_titres_de_balisage(self):
        """Les '==' disparaissent mais le titre reste comme contexte."""
        cleaned = clean_text("== Definition ==\nUn texte simple.")
        assert "==" not in cleaned
        assert "Definition" in cleaned
        assert "Un texte simple." in cleaned

    def test_supprime_les_references_numerotees(self):
        cleaned = clean_text("Le TAL est un domaine [1] de recherche [12].")
        assert "[1]" not in cleaned
        assert "[12]" not in cleaned
        assert "domaine" in cleaned

    def test_supprime_les_formules_tex(self):
        """Les articles scientifiques de Wikipedia sont pleins de TeX."""
        cleaned = clean_text(
            "La droite a pour equation {\\displaystyle ax+b} dans le plan."
        )
        assert "displaystyle" not in cleaned
        assert "La droite" in cleaned
        assert "dans le plan" in cleaned

    def test_supprime_les_formules_tex_imbriquees(self):
        cleaned = clean_text(
            "Le couple {\\displaystyle (x_{i},f(x_{i}))} est hors de la droite."
        )
        assert "displaystyle" not in cleaned
        assert "{" not in cleaned
        assert "est hors de la droite" in cleaned

    def test_supprime_les_debris_de_formules(self):
        """Cas reel : "a. x. +. b." polluait le chunk le mieux classe."""
        cleaned = clean_text("a. x. +. b. Pour eviter le surapprentissage.")
        assert "Pour eviter le surapprentissage" in cleaned
        assert not cleaned.startswith("a.")

    def test_ne_prend_pas_une_formule_pour_un_titre(self):
        """Cas reel : Wikipedia rend les formules a raison d'UN SYMBOLE PAR LIGNE.

        Sans garde-fou, chaque ligne etait prise pour un titre de section et
        recevait un point, fabriquant le texte "a. x. +. b." qui polluait le
        passage le mieux classe du corpus.
        """
        brut = (
            "le reseau repond\n  \n    a\n    x\n    +\n    b\n  \n"
            "pour toute valeur de x."
        )
        cleaned = clean_text(brut)
        assert "a." not in cleaned
        assert "x." not in cleaned.replace("valeur de x.", "")
        assert "le reseau repond" in cleaned

    def test_un_vrai_titre_recoit_bien_un_point(self):
        """Le garde-fou ne doit pas empecher les vrais titres d'etre marques."""
        cleaned = clean_text("Histoire\nLe domaine est ancien.")
        assert "Histoire." in cleaned

    def test_supprime_les_debris_avec_ponctuation(self):
        """Variante reelle : une virgule interrompait le motif ("x. i., f.")."""
        cleaned = clean_text("x. i., f. Le reste de la phrase est utile.")
        assert "Le reste de la phrase est utile" in cleaned
        assert not cleaned.startswith("x.")

    def test_ne_casse_pas_une_phrase_normale(self):
        """Garde-fou : le filtre anti-formules ne doit pas manger du texte."""
        phrase = "Le modele apprend sur des donnees et generalise ensuite."
        assert clean_text(phrase) == phrase

    def test_supprime_html_et_modeles_wiki(self):
        cleaned = clean_text("Texte <b>gras</b> et {{modele}} residuel.")
        assert "<b>" not in cleaned
        assert "{{" not in cleaned
        assert "gras" in cleaned

    def test_normalise_les_espaces(self):
        cleaned = clean_text("Trop    d'espaces\n\n\n\n\net de lignes.")
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_supprime_espace_insecable_et_caracteres_de_controle(self):
        cleaned = clean_text("Mot suivant\x07 et­ coupure.")
        assert " " not in cleaned
        assert "\x07" not in cleaned
        assert "­" not in cleaned

    def test_coupe_les_sections_finales_inutiles(self):
        """Bibliographie et liens externes ne doivent pas polluer le corpus."""
        cleaned = clean_text(
            "Contenu utile a conserver.\n"
            "== Voir aussi ==\n"
            "Lien vers une autre page a jeter."
        )
        assert "Contenu utile" in cleaned
        assert "a jeter" not in cleaned

    def test_conserve_les_accents_et_la_ponctuation(self):
        """Point crucial : le texte destine au LLM ne doit pas etre mutile."""
        cleaned = clean_text("Le modele repond-il bien ? Oui, tres bien !")
        assert "?" in cleaned
        assert "," in cleaned
        assert "modele" in cleaned

    @pytest.mark.parametrize("entree", ["", "   ", "\n\n"])
    def test_texte_vide_ne_plante_pas(self, entree):
        assert clean_text(entree) == ""


# ==========================================================================
#  Segmentation et tokenisation
# ==========================================================================
class TestSegmentation:
    def test_decoupe_en_phrases(self):
        phrases = split_sentences(
            "Premiere phrase. Deuxieme phrase ! Troisieme ?", "fr"
        )
        assert len(phrases) >= 3

    def test_phrase_unique_sans_point_final(self):
        assert len(split_sentences("Une seule phrase sans point", "fr")) == 1

    def test_texte_vide(self):
        assert split_sentences("", "fr") == []

    def test_tokenise_en_separant_la_ponctuation(self):
        tokens = tokenize("Le langage, c'est important.")
        assert "langage" in tokens
        # La virgule ne doit pas rester collee au mot
        assert "langage," not in tokens

    def test_garde_les_nombres(self):
        assert "2017" in tokenize("Introduit en 2017 par Google.")

    def test_garde_les_accents_a_la_tokenisation(self):
        assert "modele" in tokenize("un modele simple")


# ==========================================================================
#  Normalisation pour TF-IDF
# ==========================================================================
class TestNormalisation:
    def test_retire_les_accents(self):
        assert strip_accents("modèle élève") == "modele eleve"

    def test_supprime_les_stop_words(self):
        tokens = normalize_tokens("le chat de la maison", "fr")
        assert "le" not in tokens
        assert "la" not in tokens
        assert "de" not in tokens
        # Les mots porteurs de sens subsistent (eventuellement racinises)
        assert any(token.startswith("chat") for token in tokens)

    def test_met_en_minuscules(self):
        assert all(token.islower() or token.isdigit()
                   for token in normalize_tokens("LANGAGE Naturel", "fr"))

    def test_peut_conserver_les_stop_words(self):
        tokens = normalize_tokens("le chat", "fr", remove_stopwords=False, stem=False)
        assert "le" in tokens

    def test_la_racinisation_rapproche_les_formes(self):
        """'modeles' et 'modele' doivent se ramener a la meme racine."""
        singulier = normalize_tokens("modele", "fr", remove_stopwords=False)
        pluriel = normalize_tokens("modeles", "fr", remove_stopwords=False)
        assert singulier == pluriel

    def test_filtre_les_tokens_trop_courts(self):
        tokens = normalize_tokens("a b cd efg", "fr", remove_stopwords=False,
                                  stem=False, min_len=3)
        assert "a" not in tokens
        assert "cd" not in tokens
        assert "efg" in tokens


# ==========================================================================
#  Deduplication
# ==========================================================================
class TestDeduplication:
    def test_empreinte_identique_malgre_casse_et_accents(self):
        assert content_fingerprint("Le Modèle") == content_fingerprint("le modele")

    def test_empreinte_differente_pour_contenus_differents(self):
        assert content_fingerprint("chat") != content_fingerprint("chien")

    def test_supprime_les_doublons_de_titre(self, raw_articles):
        doubles = raw_articles + [raw_articles[0]]
        unique, removed = deduplicate(doubles)
        assert removed == 1
        assert len(unique) == len(raw_articles)

    def test_detecte_un_doublon_de_contenu_sous_un_autre_titre(self, raw_articles):
        """Meme texte, autre titre, autre pageid : c'est quand meme un doublon."""
        clone = dict(raw_articles[0])
        clone["title"] = "TAL (copie)"
        clone["pageid"] = 999
        unique, removed = deduplicate(raw_articles + [clone])
        assert removed == 1

    def test_corpus_sans_doublon_reste_intact(self, raw_articles):
        unique, removed = deduplicate(raw_articles)
        assert removed == 0
        assert len(unique) == 3


# ==========================================================================
#  Pipeline complet : les DEUX representations
# ==========================================================================
class TestProcessRecord:
    def test_produit_les_champs_attendus(self, raw_articles):
        record = process_record(raw_articles[0], "fr")
        for champ in ("text_clean", "sentences", "tokens_norm", "norm_text",
                      "title", "url", "license", "n_chars_clean"):
            assert champ in record, f"champ manquant : {champ}"

    def test_les_deux_representations_sont_differentes(self, raw_articles):
        """C'est LA lecon du module : un seul texte ne sert pas les deux taches."""
        record = process_record(raw_articles[0], "fr")
        assert record["text_clean"] != record["norm_text"]
        # Le texte destine au LLM garde ses mots outils...
        assert " de " in record["text_clean"] or " la " in record["text_clean"]
        # ...alors que la version TF-IDF les a perdus.
        assert " de " not in f" {record['norm_text']} "

    def test_le_nettoyage_reduit_la_taille(self, raw_articles):
        record = process_record(raw_articles[0], "fr")
        assert record["n_chars_clean"] < record["n_chars_raw"]
        assert record["n_chars_clean"] > 0

    def test_les_metriques_sont_coherentes(self, raw_articles):
        record = process_record(raw_articles[0], "fr")
        assert record["n_sentences"] == len(record["sentences"])
        assert record["n_tokens_norm"] == len(record["tokens_norm"])
        # La normalisation supprime des tokens, elle n'en ajoute jamais.
        assert record["n_tokens_norm"] <= record["n_tokens"]
