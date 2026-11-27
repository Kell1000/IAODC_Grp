# WikiNLP Assistant

Assistant conversationnel qui repond en langage naturel a partir d'un corpus
d'articles Wikipedia, en citant ses sources. Projet pedagogique complet :
collecte de donnees, pretraitement NLP, recherche lexicale et semantique,
systeme RAG, interface web, tests, Docker, CI et deploiement.

**Contrainte de conception : 0 euro, 16 Go de RAM, pas de GPU.**
Aucune cle API n'est necessaire pour faire fonctionner le projet.

---

## Sommaire

- [Ce que fait le projet](#ce-que-fait-le-projet)
- [Architecture](#architecture)
- [Choix techniques et justifications](#choix-techniques-et-justifications)
- [Demarrage rapide](#demarrage-rapide)
- [Structure du projet](#structure-du-projet)
- [Commandes](#commandes)
- [Configuration](#configuration)
- [Docker](#docker)
- [Tests et evaluation](#tests-et-evaluation)
- [Deploiement](#deploiement)
- [Limites connues](#limites-connues)
- [Licence et attribution](#licence-et-attribution)

---

## Ce que fait le projet

L'utilisateur pose une question. Le systeme :

1. cherche les passages les plus pertinents dans un corpus Wikipedia local ;
2. verifie que ces passages sont suffisamment pertinents (sinon il refuse de
   repondre, au lieu d'inventer) ;
3. demande a un modele de langue de rediger une reponse **uniquement** a
   partir de ces passages ;
4. affiche la reponse, les sources Wikipedia utilisees, et le contexte exact
   transmis au modele.

### La distinction centrale du projet

| Approche | Contenu de la reponse | Sources | Risque |
| --- | --- | --- | --- |
| **LLM seul** | memoire du modele | aucune | invente avec aplomb |
| **Recherche documentaire** | extraits bruts | exactes | aucune redaction |
| **RAG** | redige depuis les extraits | exactes | faible, verifiable |

Le RAG n'est pas un LLM plus intelligent : c'est un LLM a qui on **interdit**
de parler d'autre chose que des documents fournis.

---

## Architecture

```
                         +--------------------------+
                         |   Wikipedia (MediaWiki)  |
                         +------------+-------------+
                                      | HTTPS, User-Agent identifie,
                                      | 1 requete / 0,5 s
            ETAPE 1                   v
       +---------------------------------------------------+
       |  src/collect_wikipedia.py                         |
       |  classement par co-citation, deduplication,        |
       |  relancable (reprend ou il s'est arrete)           |
       +------------------------+--------------------------+
                                v   data/raw/articles.jsonl
            ETAPE 2
       +---------------------------------------------------+
       |  src/preprocess.py                                |
       |  PRODUIT DEUX REPRESENTATIONS :                    |
       |    text_clean  -> lisible, pour le LLM             |
       |    tokens_norm -> normalise, pour TF-IDF           |
       +------------------------+--------------------------+
                                v   data/processed/articles_clean.jsonl
            ETAPE 3
       +---------------------------------------------------+
       |  src/rag.py : chunk_text()                        |
       |  ~900 caracteres, chevauchement par PHRASES        |
       +------------------------+--------------------------+
                                v   data/processed/chunks.jsonl
            ETAPE 4                  (deux moteurs au choix)
       +----------------------------+----------------------------+
       |  TfidfRetriever            |  EmbeddingRetriever        |
       |  scikit-learn              |  sentence-transformers     |
       |  lexical, 0 Mo, instantane |  semantique, ~470 Mo       |
       +----------------------------+-------------+--------------+
                                                  v
                                    +--------------------------+
                                    |  src/vector_store.py     |
                                    |  numpy (exact) | chroma  |
                                    +------------+-------------+
                                                 v  data/index/
            ETAPE 5
       +---------------------------------------------------+
       |  src/rag.py : RagPipeline                         |
       |  recherche -> GARDE-FOU (min_score) -> prompt      |
       +------------------------+--------------------------+
                                v
            ETAPE 6             |  src/llm.py (4 fournisseurs)
       +--------------+---------+--------+--------------+
       |  extractive  |  groq   | gemini |   ollama     |
       |  sans cle    | gratuit | gratuit|  local       |
       +--------------+---------+--------+--------------+
                                v
       +---------------------------------------------------+
       |  app.py (Streamlit)  |  app_gradio.py (HF Spaces) |
       |  reponse + sources + contexte + mesures            |
       +---------------------------------------------------+
```

### Le garde-fou anti-hallucination

C'est la piece la plus importante du systeme :

```
question --> recherche --> meilleur score < MIN_SCORE ?
                               |                |
                             OUI               NON
                               |                |
                   "Je ne trouve pas       appel au LLM
                    cette information"     avec le contexte
                               |                |
                      aucune source      reponse + sources
```

Le refus intervient **avant** l'appel au LLM. Deux benefices : aucun quota
consomme pour rien, et suppression de la premiere cause d'hallucination (un
LLM a qui l'on fournit un contexte hors sujet brode quand meme).

---

## Choix techniques et justifications

| Besoin | Choix | Pourquoi celui-la |
| --- | --- | --- |
| Collecte | `wikipedia-api` | respecte la politique Wikimedia, exige un User-Agent identifie |
| Phrases / stop words | NLTK | leger, solution de secours en regex si le telechargement echoue |
| Lemmatisation FR | Snowball (stemming) | le lemmatiseur WordNet de NLTK est anglophone ; spaCy FR demanderait un modele en plus |
| Recherche lexicale | scikit-learn TF-IDF | reference indispensable pour mesurer l'apport du semantique |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 470 Mo, 384 dim., **multilingue** : le corpus est en francais |
| Base vectorielle | NumPy par defaut, ChromaDB en option | a 1 400 chunks, la recherche exhaustive est exacte ET plus rapide qu'un index approximatif |
| LLM | Groq / Gemini / Ollama / extractif | aucun fournisseur obligatoire, repli automatique |
| Interface | Streamlit + Gradio | Streamlit en local et Docker ; Gradio pour la seule voie gratuite de HF Spaces |
| Appels LLM | `requests` brut | evite 3 SDK et leurs conflits de versions |

### Pourquoi `all-MiniLM-L6-v2` n'est pas le defaut

Ce modele est tres populaire et quatre fois plus leger (90 Mo), mais il est
**entraine sur l'anglais uniquement**. Sur un corpus francais, sa qualite
s'ecroule. Il reste disponible dans `.env` (`EMBEDDING_MODEL`) et c'est le
bon choix si vous mettez `WIKI_LANG=en`.

---

## Demarrage rapide

### Prerequis

| Outil | Version | Verification |
| --- | --- | --- |
| Python | 3.10 a 3.12 | `python --version` |
| Git | 2.x | `git --version` |
| Docker Desktop | optionnel | `docker --version` |

> **Windows :** si `python --version` ouvre le Microsoft Store, Python n'est
> pas installe. Voir l'etape 1 du TP (`TP_WikiNLP_Assistant.md`).

### Installation (PowerShell)

```powershell
git clone https://github.com/<votre-compte>/WikiNLP-Assistant.git
cd WikiNLP-Assistant

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env          # renseignez WIKI_USER_AGENT avec un contact reel
```

### Le pipeline complet

```powershell
python -m src.collect_wikipedia --max 30    # 1. collecte (~2 min)
python -m src.preprocess                    # 2. nettoyage (~10 s)
python -m src.rag --build                   # 3. index (~2 min la 1re fois)
streamlit run app.py                        # 4. interface web
```

Ouvrez <http://localhost:8501>.

### Version minimale, sans aucun telechargement de modele

```powershell
python -m pip install -r requirements-lite.txt
# dans .env :  RETRIEVER=tfidf
python -m src.collect_wikipedia --max 20
python -m src.preprocess
python -m src.rag --build
streamlit run app.py
```

---

## Structure du projet

```
WikiNLP-Assistant/
├── app.py                     Interface Streamlit (local, Docker)
├── app_gradio.py              Interface Gradio (Hugging Face Spaces)
├── requirements.txt           Dependances completes (~900 Mo)
├── requirements-lite.txt      Sans torch (~60 Mo) : CI et hebergement gratuit
├── requirements-gradio.txt    Pour un Space Gradio
├── requirements.lock.txt      Versions exactes mesurees (reproduction a l'identique)
├── Dockerfile                 Image legere par defaut, complete sur option
├── compose.yaml               Orchestration + service Ollama optionnel
├── .env.example               Modele de configuration (aucun secret)
├── .gitignore                 Protege .env, les donnees et les modeles
├── pyproject.toml             Configuration de pytest
├── run_all.ps1                Lance tout le pipeline d'un coup
├── data/
│   ├── raw/                   articles.jsonl (collecte brute)
│   ├── processed/             articles_clean.jsonl, chunks.jsonl
│   └── index/                 index vectoriel persiste
├── src/
│   ├── config.py              Configuration centrale validee
│   ├── collect_wikipedia.py   Etape 2 : collecte
│   ├── preprocess.py          Etape 3 : nettoyage, 2 representations
│   ├── nlp_models.py          Etape 4 : experiences A, B, B2
│   ├── embeddings.py          Embeddings + cache disque
│   ├── vector_store.py        Base vectorielle (numpy | chroma)
│   ├── llm.py                 4 fournisseurs LLM + gestion d'erreurs
│   ├── rag.py                 Chunking, retrievers, pipeline RAG
│   └── evaluate.py            Etape 8 : evaluation mesuree
├── tests/                     117 tests, hors ligne
├── eval/questions.json        Jeu de questions d'evaluation
├── docs/DEPLOYMENT.md         Deploiement et solutions de secours
└── .github/workflows/tests.yml  CI : tests + construction Docker
```

**Separation stricte** : `src/` contient le code (versionne), `data/` contient
les donnees generees (ignorees par git, sauf le mini-corpus de demonstration).

---

## Commandes

### Pipeline

```powershell
python -m src.collect_wikipedia --max 30                 # collecter
python -m src.collect_wikipedia --stats                  # qualite du corpus
python -m src.collect_wikipedia --topics "Climat" "Oceanographie"
python -m src.collect_wikipedia --force                  # repartir de zero
python -m src.preprocess                                 # nettoyer
python -m src.preprocess --stats                         # apercu
python -m src.rag --build                                # indexer
python -m src.rag --info                                 # etat du pipeline
```

### Interroger

```powershell
python -m src.rag --ask "Qu'est-ce que l'apprentissage profond ?"
python -m src.rag --compare "Qu'est-ce qu'un reseau de neurones ?"
```

`--compare` affiche cote a cote TF-IDF, recherche semantique et RAG complet.

### Experiences

```powershell
python -m src.nlp_models --exp-a        # NLP classique
python -m src.nlp_models --exp-b        # embeddings + comparaison
python -m src.nlp_models --sentiment    # classification (+270 Mo)
```

### Interfaces

```powershell
streamlit run app.py
streamlit run app.py --server.port 8502     # si 8501 est occupe
python app_gradio.py                        # variante Gradio
```

---

## Configuration

Tout se regle dans `.env`. Variables principales :

| Variable | Defaut | Role |
| --- | --- | --- |
| `WIKI_LANG` | `fr` | langue du corpus |
| `WIKI_USER_AGENT` | - | **a personnaliser** : exige par Wikimedia |
| `MAX_ARTICLES` | `30` | plafond d'articles |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | decoupage |
| `TOP_K` | `4` | passages recuperes |
| `MIN_SCORE_TFIDF` | `0.13` | **seuil du garde-fou** en recherche lexicale |
| `MIN_SCORE_EMBEDDINGS` | `0.43` | **seuil du garde-fou** en recherche semantique |
| `MIN_SCORE` | *(vide)* | force le meme seuil pour les deux moteurs |
| `RETRIEVER` | `embeddings` | `tfidf` ou `embeddings` |
| `EMBEDDING_MODEL` | multilingue | modele de phrases |
| `VECTOR_BACKEND` | `numpy` | `numpy` ou `chroma` |
| `LLM_PROVIDER` | `extractive` | `extractive`, `groq`, `gemini`, `ollama` |
| `HF_HOME` | `./hf_cache` | cache des modeles (evite de remplir `C:`) |

### Regler le seuil du garde-fou

**Il y a un seuil par moteur, et ce n'est pas un detail.** Les scores de
TF-IDF et des embeddings ne sont pas sur la meme echelle. Les deux moteurs ont
ete mesures sur le corpus de reference (1 345 chunks, jeu de 8 questions) :

| | `tfidf` | `embeddings` |
| --- | --- | --- |
| Questions **dans** le corpus | 0,151 a 0,205 | 0,821 a 0,897 |
| Questions **hors** corpus | 0,081 a 0,121 | 0,230 a 0,365 |
| Seuil retenu | **0,13** | **0,43** |

Les deux seuils ci-dessus ont ete choisis **entre** les deux populations de
chaque moteur :

- un seuil unique de `0.25` se situait **au-dessus des deux populations** de
  TF-IDF : le systeme refusait les 5 questions legitimes
  (`false_refusals_in_corpus = 5`) ;
- la valeur `0.30` d'abord retenue pour les embeddings tombait **dans** leur
  population hors corpus : une question hors sujet obtenait 0,3652 et passait
  le garde-fou (`refusal_rate_out_of_corpus = 0,667`). Mesure refaite a
  `0.43` : **1.0 et 0 simultanement**.

**Mais 8 questions ne suffisent pas a placer un seuil.** Une sonde
complementaire de 20 questions (10 dans, 10 hors ; gardee hors du jeu
d'evaluation pour ne pas en fausser les metriques) change la lecture du
tableau ci-dessus :

| Populations elargies (28 questions) | `tfidf` | `embeddings` |
| --- | --- | --- |
| Questions **dans** le corpus | 0,105 a 0,234 | 0,495 a 0,897 |
| Questions **hors** corpus | 0,076 a 0,139 | 0,186 a 0,365 |
| Separation | **recouvrement** | ecart utile **0,365 - 0,495** |

Deux conclusions, et la premiere est la plus importante du projet :

1. **les deux populations de TF-IDF se recouvrent.** La question legitime
   « Comment un ordinateur apprend-il a partir d'exemples ? » obtient 0,1054,
   soit **moins** que la question hors sujet « Comment soigner une angine chez
   l'enfant ? » a 0,1390. **Aucun seuil ne les separe parfaitement** :
   `MIN_SCORE_TFIDF=0.13` est le meilleur compromis disponible, juste sur le
   jeu officiel, et rien de plus. C'est l'argument chiffre le plus solide en
   faveur de la recherche semantique — plus solide que la comparaison de
   rangs, qui reste equivoque (voir [Limites connues](#limites-connues)) ;
2. en semantique l'ecart subsiste, et `0.43` en est le **milieu exact**
   (0,43 - 0,365 = 0,495 - 0,43 = 0,065). Ne montez pas ce seuil vers 0,55 :
   la question « Qu'est-ce que le TAL ? », legitime, obtient **0,4954** et
   serait refusee.

| Symptome | Correction |
| --- | --- |
| repond a des questions hors sujet | **augmenter** le seuil du moteur actif |
| refuse des questions legitimes | **baisser** le seuil du moteur actif |

`python -m src.evaluate` mesure les deux erreurs (faux refus et absences de
refus) : reglez le seuil d'apres **vos** mesures, pas d'apres ces valeurs.

### Proteger les cles API

```powershell
Copy-Item .env.example .env     # .env est deja dans .gitignore
git check-ignore -v .env        # doit afficher la regle qui l'exclut
```

Ne mettez **jamais** de cle dans le code, dans `.env.example`, ni dans une
image Docker (elle reste lisible dans l'historique des couches). La CI echoue
volontairement si un `.env` ou une cle en dur est detecte.

---

## Docker

```powershell
# Image legere (865 Mo mesure, TF-IDF)
docker build -t wikinlp:lite .

# Image complete (recherche semantique, torch inclus)
docker build --build-arg REQUIREMENTS=requirements.txt -t wikinlp:full .

# Avec Compose (recommande)
docker compose up -d --build
docker compose logs -f app
docker compose ps
docker compose down                 # arrete, GARDE les donnees
docker compose down -v              # arrete et SUPPRIME les volumes
```

Ouvrez <http://localhost:8501>.

- **Persistance** : `./data` est monte en volume (corpus et index survivent) ;
  le cache des modeles est dans le volume nomme `hf_cache`.
- **Secrets** : injectes via `env_file: .env`, jamais copies dans l'image.
- **LLM local** : `docker compose --profile local-llm up -d`, puis
  `docker compose exec ollama ollama pull qwen2.5:1.5b-instruct`, et dans
  `.env` : `LLM_PROVIDER=ollama` et `OLLAMA_HOST=http://ollama:11434`.

---

## Tests et evaluation

```powershell
python -m pytest                     # 117 tests, aucun reseau requis
python -m pytest -v -k garde_fou
python -m src.evaluate               # mesure sur votre corpus
python -m src.evaluate --compare     # tfidf contre embeddings
```

Les tests tournent hors ligne (`RETRIEVER=tfidf`, `LLM_PROVIDER=extractive`) :
ils sont donc rapides et deterministes, y compris dans la CI.

`src/evaluate.py` mesure hit@k, MRR, couverture des mots cles, **taux de refus
sur les questions hors corpus**, temps de reponse et memoire, puis ecrit un
rapport dans `eval/`.

### Les deux moteurs, mesures cote a cote

Sortie de `python -m src.evaluate --compare` sur le corpus de reference
(30 articles, 1 345 chunks, `extractive`, **une seule execution pour les deux
moteurs**) :

| Metrique | `tfidf` | `embeddings` | Lecture |
| --- | --- | --- | --- |
| `hit_at_k` (k=4) | **1.0** | **1.0** | egalite : les deux trouvent le bon article |
| `mrr` | **0,875** | 0,583 | TF-IDF classe mieux l'article **attendu** |
| `keyword_coverage_mean` | **0,834** | 0,768 | |
| `refusal_rate_out_of_corpus` | **1.0** | **1.0** | aucune hallucination, seuil propre a chaque moteur |
| `false_refusals_in_corpus` | **0** | **0** | |
| `first_query_s` (a froid) | 2,22 | **94,61** | la 1re question charge le modele |
| Recherche **a chaud**, par question | 0,015 | 0,19 | le regime normal, ~13x |
| `mean_retrieval_time_s` | 0,456 | 13,47 | **ne citez pas cette moyenne seule** : voir la note |
| `mean_total_time_s` | 0,291 | 8,48 | idem |
| `memory_mb_end` | 103,0 | 165,8 | `tracemalloc` : **sous-estime**, il ignore PyTorch |
| Pic de RSS reel (`VmHWM`) * | 312 Mo | **1 495 Mo** | vraie empreinte memoire |
| Index sur disque | aucun | 5,2 Mo | `data/index/` |

\* mesure dans deux executions separees : `--compare` evaluant les deux
moteurs dans le meme processus, leurs pics de memoire y seraient confondus.

**Le resultat surprend, et c'est l'interet de mesurer.** Sur ce corpus, le
moteur lexical est meilleur sur les metriques de classement, environ 13 fois
plus rapide a chaud dans ce conteneur et presque 5 fois plus leger. Le moteur
semantique gagne ailleurs : il separe nettement les questions legitimes des
questions hors sujet (voir
[Regler le seuil du garde-fou](#regler-le-seuil-du-garde-fou)) et comprend
les sigles et les reformulations, la ou TF-IDF echoue.

> **Attention a `mean_retrieval_time_s` : cette moyenne ne decrit aucune
> situation reelle.** Elle melange la 1re question, qui charge le modele
> (**94,61 s**), et les suivantes (**0,19 s**) — un facteur 500. C'etait un
> vrai defaut, pas un artefact de mesure : l'index semblait pret en une
> seconde et c'est l'utilisateur qui payait la minute et demie a sa premiere
> question. `EmbeddingRetriever.warmup()` charge donc le modele a la
> construction du retriever, dans le `spinner` de `app.py` : mesure en natif
> apres correction, la 1re question passe a **0,108 s**. `src/evaluate.py`
> publie maintenant `first_query_s` et `warm_retrieval_time_s` separement.
>
> Le biais ote, l'ecart honnete est bien plus faible qu'annonce : en natif,
> **0,024 s** en lexical contre **0,117 s** en semantique, soit **4,9x** — et
> non les ~30x que suggerait `mean_total_time_s`. Chargement de l'index
> vectoriel depuis le disque : 0,17 s.

> Les chiffres dependent de votre corpus et de votre machine. Ce depot ne
> contient volontairement aucun resultat preenregistre : mesurez les votres.

---

## Deploiement

Voir **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** pour la procedure detaillee.

Resume des contraintes constatees en octobre 2026 (**a reverifier avant de
deployer, ces offres changent**) :

| Plateforme | Etat pour ce projet |
| --- | --- |
| Hugging Face Spaces | Gradio/Docker demandent un compte payant ; un compte gratuit peut heberger jusqu'a **2 Spaces Gradio sur ZeroGPU** -> utiliser `app_gradio.py` |
| Streamlit Community Cloud | gratuit, 1 app privee / apps publiques illimitees, RAM limitee, mise en veille apres inactivite -> utiliser `app.py` avec `RETRIEVER=tfidf` |
| Local / Docker | sans limite, et la seule option vraiment maitrisee |

Aucune de ces offres ne garantit une disponibilite permanente.

---

## Limites connues

- **Corpus limite volontairement** a 20-50 articles : au-dela, la collecte est
  longue et la memoire grimpe. Le systeme ne sait que ce qu'il a collecte.
- **TF-IDF ignore les synonymes.** Mesure faite pendant le TP : sur
  « Qu'est-ce que le traitement automatique des langues ? », le passage
  contenant la **definition** n'arrive qu'en **4e** position (score 0,1344),
  derriere une etude bibliometrique (0,1969) qui repete simplement les mots
  de la question. Le passage de definition, lui, emploie les sigles « TAL »
  et « NLP », que TF-IDF traite comme des mots sans rapport. La recherche
  semantique, mesuree sur la meme question, remonte ce passage en **3e**
  position a 0,7773 (soit **95 %** du meilleur score, contre 68 % en lexical)
  et chasse du top 4 le passage hors sujet tire d'un autre article. Elle ne
  le place toutefois **pas premier** : ses deux premiers passages viennent de
  « Linguistique informatique », un article voisin.
- **Le moteur semantique coute cher pour un gain partiel.** Mesure :
  1 495 Mo de pic memoire contre 312 Mo, 94,6 s de chargement du modele au
  premier appel, 3 min 02 s pour encoder 1 345 chunks sur CPU, et un `mrr`
  **inferieur** a celui de TF-IDF (0,583 contre 0,875) sur ce corpus. Son
  avantage mesure est ailleurs : la separation nette des questions legitimes
  et hors sujet, et les sigles. Sur « Qu'est-ce que le TAL ? », TF-IDF donne
  0,1347 pour un seuil a 0,13 — il accepte de justesse, alors qu'une question
  hors sujet comme « Comment soigner une angine chez l'enfant ? » obtient
  **0,1390**, soit davantage. Les embeddings donnent 0,4954 contre 0,2508.
  Plus grave : a ce score de 0,1347, TF-IDF ne ramene pas du tout le bon
  article (son 1er passage vient d'*Exploration de donnees*) tout en restant
  **au-dessus de son propre seuil**. Il repond donc avec aplomb depuis un
  passage hors sujet, sans que le garde-fou se declenche. La recherche
  semantique place le bon passage 3e — et pour un RAG, ce qui compte est
  d'etre dans les `top_k` passages transmis au modele, pas d'etre premier.
- **Le stemming n'est pas une lemmatisation** : « modeles » et « modeliser »
  se confondent. Suffisant pour TF-IDF, pas pour une analyse linguistique fine.
- **Pas de memoire conversationnelle** : chaque question est traitee seule.
  « Et son inventeur ? » ne sera pas comprise comme une question de suivi.
- **Qualite tributaire de Wikipedia** : erreurs et biais de la source se
  retrouvent dans les reponses.
- **Mode extractif** : il n'y a pas de redaction, seulement des extraits.
- **Les quotas des API gratuites changent sans preavis.** Verifiez-les dans la
  console de chaque fournisseur ; le repli local reste disponible.

---

## Licence et attribution

Le code de ce projet est fourni a des fins pedagogiques.

Les contenus textuels proviennent de **Wikipedia** et sont diffuses sous
licence **CC BY-SA 4.0**. Chaque enregistrement collecte conserve son titre,
son URL et sa licence, et l'interface affiche systematiquement les sources
utilisees, conformement aux exigences d'attribution.

Les reponses sont generees automatiquement : **verifiez toujours les sources
citees** avant de reutiliser une information.
