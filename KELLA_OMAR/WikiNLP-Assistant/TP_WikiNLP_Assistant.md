# TP — WikiNLP Assistant

### Construire, tester, conteneuriser et deployer un assistant NLP/RAG complet

**Public :** debutant avance a intermediaire en Python.
**Duree estimee :** 5 seances de 2 a 3 heures.
**Materiel cible :** Windows 11, 16 Go de RAM, 50 Go de disque, **sans GPU**.
**Budget :** 0 euro. Aucune cle API n'est obligatoire.

---

## Comment lire ce document

Chaque etape suit toujours le meme plan en 10 points :

1. Objectif pedagogique
2. Explication simple
3. Prerequis
4. Commandes exactes
5. Code complet
6. Explication du code
7. Resultat attendu
8. Test de validation
9. Erreurs frequentes et corrections
10. Livrable de l'etape

> **Au sujet du point 5 « Code complet ».**
> Tous les scripts de ce TP sont livres **entiers et fonctionnels** dans le
> dossier du projet : il n'y a nulle part de pseudo-code ni de `...` a
> completer. Le document pointe vers chaque fichier et en commente les
> passages decisifs. Les fichiers eux-memes sont la reference : ils sont
> abondamment commentes, et c'est voulu — un TP doit s'expliquer tout seul.

> **Au sujet des resultats chiffres.**
> Ce document ne contient **aucun resultat experimental invente**. Les
> chiffres cites comme « mesure de reference » ont ete reellement obtenus en
> executant ce TP (environnement precise en annexe C). Les votres differeront :
> le corpus Wikipedia evolue, et votre machine n'est pas la machine de test.
> **Relevez toujours vos propres mesures** : c'est l'objet de l'etape 8.

---

## Table des matieres

| Seance | Contenu | Etapes |
| --- | --- | --- |
| **0** | [Architecture et choix techniques](#seance-0--architecture-et-choix-techniques) | — |
| **1** | [Environnement, collecte, pretraitement](#seance-1--des-donnees-propres) | 1 a 3 |
| **2** | [Experiences NLP : classique et neuronal](#seance-2--experimenter-les-modeles-nlp) | 4 |
| **3** | [Systeme RAG et chatbot web](#seance-3--rag-et-interface-web) | 5 a 7 |
| **4** | [Tests, evaluation, Docker](#seance-4--qualite-et-conteneurisation) | 8 a 9 |
| **5** | [Git, GitHub, deploiement](#seance-5--publier-et-deployer) | 10 a 11 |
| **Annexes** | [Erreurs, checklist, grille, mesures](#annexe-a--tableau-des-erreurs-courantes) | 12 |

---

# Seance 0 — Architecture et choix techniques

## 0.1 Le probleme a resoudre

On veut poser une question en francais et obtenir une reponse **fiable et
sourcee** a partir d'un ensemble d'articles Wikipedia.

Trois approches existent. Les confondre est l'erreur la plus repandue chez les
debutants en IA, et c'est exactement ce que ce TP vous apprend a distinguer.

| | Ce qu'on fait | Contenu de la reponse | Sources | Risque principal |
| --- | --- | --- | --- | --- |
| **1. LLM seul** | on envoie la question au modele | sa memoire d'entrainement | **aucune** | il **invente** avec assurance |
| **2. Recherche documentaire** | on renvoie les passages proches | extraits bruts | exactes | **rien n'est redige** |
| **3. RAG** | on cherche, **puis** on fait rediger | redige **depuis les extraits** | exactes | faible, et **verifiable** |

**Le RAG n'est pas un LLM plus intelligent.** C'est un LLM a qui l'on
**interdit** de parler d'autre chose que des documents qu'on lui fournit. Le
contenu vient de la recherche documentaire ; le modele n'apporte que la
reformulation. Cette phrase est le coeur du TP.

## 0.2 Architecture retenue

```
                         +--------------------------+
                         |   Wikipedia (MediaWiki)  |
                         +------------+-------------+
                                      | HTTPS, User-Agent identifie,
                                      | 1 requete / 0,5 s
            ETAPE 2                   v
       +---------------------------------------------------+
       |  src/collect_wikipedia.py                         |
       |  classement par co-citation, deduplication,        |
       |  relancable (reprend ou il s'est arrete)           |
       +------------------------+--------------------------+
                                v   data/raw/articles.jsonl
            ETAPE 3
       +---------------------------------------------------+
       |  src/preprocess.py                                |
       |  PRODUIT DEUX REPRESENTATIONS :                    |
       |    text_clean  -> lisible, pour le LLM             |
       |    tokens_norm -> normalise, pour TF-IDF           |
       +------------------------+--------------------------+
                                v   data/processed/articles_clean.jsonl
            ETAPE 5a
       +---------------------------------------------------+
       |  src/rag.py : chunk_text()                        |
       |  ~900 caracteres, chevauchement par PHRASES        |
       +------------------------+--------------------------+
                                v   data/processed/chunks.jsonl
            ETAPES 4 et 5b           (deux moteurs comparables)
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
            ETAPE 5c
       +---------------------------------------------------+
       |  src/rag.py : RagPipeline                         |
       |  recherche -> GARDE-FOU (min_score) -> prompt      |
       +------------------------+--------------------------+
                                v
            ETAPE 4c             |  src/llm.py (4 fournisseurs)
       +--------------+---------+--------+--------------+
       |  extractive  |  groq   | gemini |   ollama     |
       |  sans cle    | gratuit | gratuit|  local       |
       +--------------+---------+--------+--------------+
                                v
            ETAPE 6
       +---------------------------------------------------+
       |  app.py (Streamlit)  |  app_gradio.py (HF Spaces) |
       |  reponse + sources + contexte + mesures            |
       +---------------------------------------------------+
```

## 0.3 Le garde-fou anti-hallucination

C'est la piece la plus importante, et celle que les tutoriels oublient :

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

Le refus se produit **avant** l'appel au LLM. Deux gains concrets : on ne
consomme pas de quota pour rien, et on supprime la premiere cause
d'hallucination — un LLM a qui on donne un contexte hors sujet brode quand
meme une reponse plausible.

## 0.4 Technologies et justification de chaque choix

| Besoin | Choix | Pourquoi pas autrement |
| --- | --- | --- |
| Collecte | `wikipedia-api` | encapsule l'API MediaWiki et **impose** un User-Agent, comme l'exige Wikimedia. Scraper le HTML serait plus fragile et moins respectueux. |
| Phrases, stop words | NLTK | leger ; et le projet prevoit un repli en regex si le telechargement echoue. |
| « Lemmatisation » FR | Snowball (**stemming**) | le lemmatiseur WordNet de NLTK est **anglophone**. spaCy francais ferait mieux mais ajoute un modele a telecharger. Pour du TF-IDF, le stemming suffit. |
| Recherche lexicale | scikit-learn TF-IDF | c'est la **reference** obligatoire : sans elle, impossible de prouver que le semantique apporte quelque chose. |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (470 Mo) | **multilingue**. `all-MiniLM-L6-v2` est 4x plus leger mais **anglais uniquement** : sur un corpus francais, sa qualite s'effondre. |
| Base vectorielle | **NumPy** par defaut, ChromaDB en option | a ~1 400 chunks, la recherche exhaustive est **exacte** et prend ~1 ms. Un index approximatif serait de la sur-ingenierie. ChromaDB est quand meme fourni, car savoir brancher une vraie base vectorielle est une competence attendue. |
| LLM | Groq / Gemini / Ollama / extractif | **aucun n'est obligatoire**, et le repli est automatique. Une offre gratuite peut disparaitre : l'architecture doit y survivre. |
| Appels LLM | `requests` brut | trois SDK officiels = trois jeux de dependances et des conflits de versions. Trois requetes HTTP suffisent. |
| Interface | Streamlit **et** Gradio | Streamlit en local et en Docker ; Gradio parce que c'est la **seule voie gratuite** sur Hugging Face Spaces (verifie, voir etape 11). |
| Tests | pytest | 117 tests qui tournent **hors ligne**, donc utilisables en CI. |

## 0.5 Budget memoire et disque

> Les valeurs precedees de `~` sont des **estimations** destinees a verifier
> que le projet tient dans 16 Go de RAM et 50 Go de disque. Seules deux
> valeurs ont ete mesurees pendant la redaction : le chemin lexical
> (**101 Mo** de memoire processus mesures par `src/evaluate.py`) et l'image
> Docker legere (**865 Mo** mesures par `docker images`). Mesurez les votres.

| Poste | Disque | RAM a l'execution |
| --- | --- | --- |
| Python + venv + dependances legeres | ~200 Mo | ~80 Mo |
| PyTorch (CPU) + sentence-transformers | ~900 Mo | ~400 Mo |
| Modele d'embeddings multilingue | ~470 Mo | ~300 Mo |
| Corpus de 30 articles (brut + traite + chunks) | ~8 Mo | ~60 Mo |
| Index vectoriel (1 345 x 384 flottants 32 bits) | ~2 Mo | ~2 Mo |
| Streamlit | ~60 Mo | ~150 Mo |
| **Total chemin semantique** | **~1,7 Go** | **~1 Go** |
| **Total chemin lexical (`RETRIEVER=tfidf`)** | **~300 Mo** | **~250 Mo** |

Ces chiffres sont des **estimations de budget**, faites avant l'execution.
Elles ont ete confrontees a la mesure depuis (annexe C, environnements C1 et
C2), et l'une d'elles etait trop optimiste :

| Poste | Estime ci-dessus | **Mesure** |
| --- | --- | --- |
| RAM, chemin semantique | ~1 Go | **1 495 Mo** (pic de RSS reel) |
| RAM, chemin lexical | ~250 Mo | **312 Mo** (pic de RSS reel) |
| Index vectoriel sur disque | ~2 Mo | **5,2 Mo** (vecteurs + cache + metadonnees) |
| Roue PyTorch CPU telechargee | — | **196,3 Mo** |
| Image Docker complete | — | **1,89 Go** |

La RAM du chemin semantique depasse de 50 % l'estimation : c'est pourquoi la
limite memoire de `compose.yaml` doit etre montee a 4 Go pour cette variante,
et pourquoi l'hebergement gratuit (~1 Go) ne peut pas l'accueillir.

Confortable sur 16 Go de RAM et 50 Go de disque. **Deux precautions** qui
figurent dans le projet :

- on ne charge **qu'un seul** modele a la fois ;
- `HF_HOME=./hf_cache` deplace le cache des modeles dans le projet, au lieu de
  `C:\Users\<vous>\.cache\huggingface`. Sur beaucoup de portables, `C:` est la
  partition la plus remplie : c'est une cause classique de « disque plein » en
  pleine installation.

## 0.6 Progression : d'abord ca marche, ensuite c'est bien

L'ordre des seances est volontaire. On obtient une **version minimale
fonctionnelle** des la seance 1 (collecte + nettoyage), puis on ajoute une
brique a la fois. A aucun moment vous n'aurez un projet a moitie casse.

```
Seance 1 : donnees propres          -> on peut deja chercher du texte
Seance 2 : TF-IDF puis embeddings   -> on sait comparer deux moteurs
Seance 3 : RAG + interface web      -> le produit existe
Seance 4 : tests + Docker           -> le produit est fiable et portable
Seance 5 : Git + deploiement        -> le produit est publiable
```

---

# Seance 1 — Des donnees propres

## Etape 1 — Preparer l'environnement

### 1. Objectif pedagogique

Installer Python, Git et Docker ; creer un environnement virtuel isole et
comprendre **pourquoi** on en passe par la.

### 2. Explication simple

Un **environnement virtuel** est un dossier qui contient sa propre copie de
Python et de ses bibliotheques. Sans lui, toutes vos installations
s'empilent dans le Python du systeme : deux projets qui veulent deux versions
differentes de la meme bibliotheque finissent par se casser mutuellement.
Avec un venv, chaque projet a ses propres versions, et supprimer un projet se
reduit a supprimer son dossier.

### 3. Prerequis

- Windows 10/11 avec PowerShell.
- Une connexion Internet.
- Environ 2 Go d'espace libre (5 Go si vous construisez l'image Docker complete).

### 4. Commandes exactes

**a) Verifier ce qui est deja installe**

```powershell
python --version
git --version
docker --version
```

> **Piege Windows tres courant.** Si `python --version` ouvre le Microsoft
> Store, ou affiche « Python est introuvable », alors Python **n'est pas
> installe** : `C:\Users\<vous>\AppData\Local\Microsoft\WindowsApps\python.exe`
> n'est qu'un raccourci. Verifiez-le :

```powershell
(Get-Command python -ErrorAction SilentlyContinue).Source
```

S'il affiche un chemin contenant `WindowsApps`, passez a l'installation.

**b) Installer Python 3.12**

```powershell
winget install -e --id Python.Python.3.12
```

Puis **fermez et rouvrez PowerShell** (le `PATH` n'est relu qu'au demarrage
d'un terminal) et verifiez :

```powershell
python --version
python -m pip --version
```

> **Verifie sur une machine reellement depourvue de Python.** Cette
> installation a ete executee pendant la redaction du TP ; voici ce qui se
> passe exactement, car c'est une source de confusion classique.
>
> `winget` installe Python 3.12.10 dans
> `%LOCALAPPDATA%\Programs\Python\Python312` et place ces chemins **avant**
> `WindowsApps` dans le `PATH` utilisateur :
>
> ```
> 1. ...\Programs\Python\Python312\Scripts\
> 2. ...\Programs\Python\Python312\
> 3. ...\Programs\Python\Launcher\
> 4. ...\Microsoft\WindowsApps          <- le raccourci du Store
> ```
>
> Un terminal **neuf** trouve donc le bon Python. Mais dans un terminal
> **deja ouvert** avant l'installation, `python` continue de pointer vers le
> raccourci du Store, et vous obtenez toujours « Python est introuvable » —
> ce qui donne l'impression que l'installation a echoue. **Elle n'a pas
> echoue : il faut simplement un nouveau terminal.**
>
> Pour en avoir la preuve immediate, sans rouvrir de terminal :
>
> ```powershell
> & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version
> ```
>
> Si, meme dans un terminal neuf, le raccourci du Store gagne encore,
> desactivez les alias : *Parametres > Applications > Alias d'execution
> d'application*, puis desactivez `python.exe` et `python3.exe`.

> Pourquoi 3.12 et pas la toute derniere version ? Les bibliotheques
> scientifiques publient leurs paquets precompiles (« wheels ») avec quelques
> mois de retard. Sur une version trop recente, `pip` tente de **compiler**
> les paquets et echoue faute de compilateur C++. La version n-1 est le choix
> sur.

**c) Installer Git et Docker Desktop** (si absents)

```powershell
winget install -e --id Git.Git
winget install -e --id Docker.DockerDesktop
```

Docker Desktop demande un redemarrage et l'activation de WSL2. Il n'est
**necessaire qu'a l'etape 9** : vous pouvez faire les seances 1 a 4 sans lui.

**d) Creer le projet et son environnement virtuel**

```powershell
cd D:\TP_NLP
mkdir WikiNLP-Assistant
cd WikiNLP-Assistant

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Votre invite de commande doit maintenant commencer par `(.venv)` :

```
(.venv) PS D:\TP_NLP\WikiNLP-Assistant>
```

> **Si vous obtenez une erreur de politique d'execution** (`Activate.ps1
> cannot be loaded because running scripts is disabled`) :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`-Scope Process` ne vaut que pour cette fenetre : rien n'est modifie
durablement sur la machine, ce qui est le bon reflexe de securite.

**e) Installer les dependances**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comptez 3 a 8 minutes : PyTorch represente a lui seul environ 900 Mo.

**Vous etes presse, ou votre disque est juste ?** Installez la version legere :

```powershell
python -m pip install -r requirements-lite.txt
```

Elle n'embarque pas PyTorch (~60 Mo au total). Il faudra alors travailler avec
`RETRIEVER=tfidf`. Vous pourrez installer le reste plus tard, sans rien
casser.

**f) Creer votre fichier de configuration**

```powershell
Copy-Item .env.example .env
notepad .env
```

Renseignez **au minimum** `WIKI_USER_AGENT` avec une adresse de contact
reelle. Wikimedia exige un User-Agent identifiable et peut renvoyer une
erreur `403` sur des valeurs generiques.

### 5. Code complet

Fichiers concernes, livres complets dans le projet :

| Fichier | Role |
| --- | --- |
| `requirements.txt` | dependances completes, en **plages de versions** |
| `requirements-lite.txt` | sans PyTorch : CI et hebergement gratuit |
| `.env.example` | modele de configuration, **aucun secret** |
| `.gitignore` | protege `.env`, les donnees et les modeles |
| `src/config.py` | configuration centrale **validee** |
| `run_all.ps1` | lance tout le pipeline d'un coup |

**Pourquoi des plages de versions (`>=1.3,<2.0`) et non des versions figees ?**
Une version figee aujourd'hui sera obsolete dans six mois, et un paquet retire
rendrait le projet ininstallable. Les plages evitent les ruptures d'API tout
en laissant passer les corrections de securite. Pour une reproduction **a
l'identique**, on genere un verrou :

```powershell
python -m pip freeze > requirements.lock.txt
```

### 6. Explication du code

Le point important de cette etape est `src/config.py`. Trois idees :

**a) Un seul endroit pour tous les reglages.** Les autres modules font
`from src.config import CFG`. Changer de modele, de langue ou de fournisseur
LLM ne demande donc de modifier **aucune** ligne de code metier.

**b) Les chemins sont calcules depuis le fichier lui-meme :**

```python
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
```

Ainsi le projet fonctionne quel que soit le repertoire courant — ce qui
importe en Docker (ou le dossier de travail est `/app`) et sous Streamlit.
Des chemins relatifs du type `"data/raw"` casseraient des que l'on lance le
script depuis un autre dossier.

**c) Les reglages invalides echouent tot, avec un message clair :**

```python
if self.chunk_overlap >= self.chunk_size:
    raise ValueError(
        f"CHUNK_OVERLAP ({self.chunk_overlap}) doit etre strictement "
        f"inferieur a CHUNK_SIZE ({self.chunk_size}), sinon le "
        "decoupage boucle indefiniment."
    )
```

C'est le principe du *fail fast* : mieux vaut une erreur immediate et
explicite qu'un programme qui tourne une heure avant de produire n'importe
quoi.

Noter aussi la fonction `_env()`, qui traite une variable **vide** comme
absente. Sans cela, une ligne `TOP_K=` dans `.env` ferait planter `int("")`.

### 7. Resultat attendu

```powershell
python -m src.config
```

```
=== Configuration WikiNLP Assistant ===
  wiki_lang          = fr
  max_articles       = 30
  chunk_size         = 900
  retriever          = embeddings
  llm_provider       = extractive
  groq_api_key       = (vide)
  gemini_api_key     = (vide)
  ...
LLM utilisable : True -> Mode extractif : aucune cle API necessaire.
Racine projet  : D:\TP_NLP\WikiNLP-Assistant
```

Les cles API apparaissent **masquees** (`(vide)` ou `***abcd`) : ce resume est
concu pour pouvoir etre affiche ou journalise sans fuite de secret.

### 8. Test de validation

```powershell
python -c "import sklearn, nltk, streamlit, dotenv; print('dependances OK')"
python -m src.config
git check-ignore -v .env
```

La derniere commande doit afficher la regle de `.gitignore` qui exclut `.env`.
**Si elle n'affiche rien, arretez-vous** : votre fichier de secrets serait
publiable sur GitHub.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `python` ouvre le Microsoft Store | Python non installe (raccourci WindowsApps) | `winget install -e --id Python.Python.3.12` puis rouvrir PowerShell | `(Get-Command python).Source` ne contient plus `WindowsApps` |
| « Python est introuvable » **apres** une installation reussie | le terminal etait deja ouvert : il garde l'ancien `PATH` | **ouvrir un nouveau terminal** (verifie : le `PATH` utilisateur est correct, seul le processus en cours est perime) | `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version` repond |
| `Activate.ps1 cannot be loaded` | politique d'execution PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` | l'invite affiche `(.venv)` |
| `pip` installe hors du venv | venv non active | `.\.venv\Scripts\Activate.ps1` | `python -c "import sys; print(sys.prefix)"` pointe vers `.venv` |
| `error: Microsoft Visual C++ 14.0 required` | pip compile faute de wheel | utiliser Python 3.12, ou `requirements-lite.txt` | `pip install` se termine sans compilation |
| `No space left on device` | cache pip/HF sur `C:` sature | `$env:PIP_CACHE_DIR="D:\pipcache"` ; `HF_HOME=./hf_cache` dans `.env` | `Get-PSDrive C` |
| `ModuleNotFoundError: No module named 'src'` | lance depuis un autre dossier | se placer a la racine du projet et utiliser `python -m src.xxx` | `Get-Location` |

> **Pourquoi `python -m src.collect_wikipedia` et non
> `python src/collect_wikipedia.py` ?** La forme `-m` ajoute la racine du
> projet au chemin d'import, ce qui rend `from src.config import CFG`
> resolvable. La forme directe ne le fait pas et provoque le
> `ModuleNotFoundError` ci-dessus.

### 10. Livrable de l'etape

- [ ] `python --version` affiche 3.10, 3.11 ou 3.12 (pas un raccourci Store)
- [ ] L'invite PowerShell affiche `(.venv)`
- [ ] `python -m src.config` s'execute sans erreur
- [ ] `.env` existe, `WIKI_USER_AGENT` est personnalise
- [ ] `git check-ignore -v .env` confirme l'exclusion

---

## Etape 2 — Collecter les donnees depuis Wikipedia

### 1. Objectif pedagogique

Constituer un corpus thematique de 20 a 50 articles, en respectant les regles
d'acces de Wikimedia, et ecrire un script **relancable** et **deduplique**.

### 2. Explication simple

On interroge l'API officielle de Wikipedia via `wikipedia-api`. On part de
quelques articles « graines » sur notre theme, puis on suit leurs liens pour
elargir le corpus.

Chaque article est ecrit **immediatement** dans un fichier JSONL (un objet
JSON par ligne). Interet : si la collecte est interrompue (Ctrl+C, coupure
reseau), tout ce qui est deja telecharge est conserve, et relancer le script
reprend ou il s'etait arrete.

### 3. Prerequis

- Etape 1 terminee, venv actif.
- `WIKI_USER_AGENT` renseigne dans `.env`.

### 4. Commandes exactes

```powershell
python -m src.collect_wikipedia --max 30
python -m src.collect_wikipedia --stats
```

Autres usages utiles :

```powershell
# Changer de theme (les titres doivent exister sur Wikipedia)
python -m src.collect_wikipedia --topics "Changement climatique" "Oceanographie" --max 25

# Repartir de zero (l'ancien corpus est sauvegarde en .bak)
python -m src.collect_wikipedia --force --max 30

# Ecarter des sujets parasites
python -m src.collect_wikipedia --exclude "Jeux video" Sport
```

### 5. Code complet

Fichier : **`src/collect_wikipedia.py`** (livre complet, ~330 lignes).

Ses responsabilites :

| Fonction | Role |
| --- | --- |
| `build_client()` | client `wikipedia-api` avec User-Agent conforme |
| `should_skip_title()` | ecarte listes, homonymies, portails, pages d'identifiants |
| `fetch_page()` | recuperation avec **3 re-essais** et delai croissant |
| `candidate_titles()` | choisit les articles a collecter, **classes par pertinence** |
| `collect()` | boucle principale, deduplication, ecriture incrementale |
| `report()` | rapport qualite chiffre |

### 6. Explication du code

**a) Le User-Agent n'est pas une formalite.**

```python
return wikipediaapi.Wikipedia(
    user_agent=user_agent,
    language=lang,
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)
```

La politique Wikimedia impose un User-Agent identifiable. Le script vous
avertit explicitement si vous avez laisse `example.com`.

**b) Le classement par pertinence : l'erreur a ne pas commettre.**

Premiere version naive : prendre les liens par ordre alphabetique.

```python
for linked_title in sorted(page.links.keys()):   # MAUVAIS
    yield linked_title
```

**C'est ce que nous avons fait au premier essai, et voici le resultat reel :**

```
[  5/25] 2013 en science
[  8/25] Alibaba.com
[ 21/25] Avion
[ 23/25] Besançon
[ 24/25] Bibliométrie
```

L'article « Intelligence artificielle » contient des centaines de liens ; par
ordre alphabetique, on recolte les `A` et les `B` avant le moindre article de
NLP. Le corpus devient incoherent, et le chatbot repond a cote.

La version retenue classe les candidats sur deux signaux gratuits :

```python
# link_votes[titre] = nombre de pages graines qui pointent vers ce titre
link_votes: Counter[str] = Counter()
...
def score(title: str) -> tuple[int, int, str]:
    overlap = len(_title_words([title]) & seed_words)
    return (-link_votes[title], -overlap, title)
```

1. **la co-citation** : un article cite par 4 pages de NLP est presque
   certainement du NLP ;
2. **le recouvrement lexical** avec les mots des titres de themes.

Le titre sert de dernier critere de tri : deux executions produisent donc le
**meme** corpus (reproductibilite).

Le vocabulaire thematique est **derive des graines**, sans liste codee en
dur :

```python
def _title_words(titles):
    words = set()
    for title in titles:
        for word in re.split(r"\W+", title.lower()):
            if len(word) > 3:
                words.add(word)
    return words
```

Changer `--topics` change donc automatiquement ce qui est juge pertinent.

**c) Le filtre des pages d'identifiants.** Le classement par co-citation a un
effet de bord previsible : les modeles de references de Wikipedia pointent
tous vers « Digital Object Identifier », « ISBN », « ISSN ». Ces pages
obtiennent donc un score tres eleve sans parler du theme. **Nous l'avons
observe** lors de la deuxieme execution, d'ou `SKIP_CONTAINS`.

**d) Double deduplication.**

```python
seen_titles = {record.get("title", "") for record in existing}
seen_ids = {record.get("pageid") for record in existing}
```

Le titre seul ne suffit pas (plusieurs titres redirigent vers un meme
article) ; le `pageid` seul non plus (pages renommees). On verifie les deux.

**e) Re-essais avec delai croissant.**

```python
wait = sleep * (2 ** (attempt - 1))
```

Une coupure reseau passagere ne doit pas annuler 20 minutes de collecte.

### 7. Resultat attendu

Mesure de reference obtenue sur l'environnement d'annexe C :

```
=== Collecte Wikipedia ===
  Langue      : fr
  Plafond     : 30 articles
  Themes      : Traitement automatique des langues, Apprentissage automatique, ...

  Corpus existant : 0 article(s)
  Objectif : +30 article(s) (plafond 30)

  [  1/30] Traitement automatique des langues (15 861 car.)
  [  2/30] Apprentissage automatique (71 155 car.)
  [  3/30] Apprentissage profond (57 455 car.)
  [  4/30] Intelligence artificielle (132 752 car.)
  [  5/30] Grand modèle de langage (33 393 car.)
  ...
  [ 30/30] Théorie de l'information (30 829 car.)

  Ajoutes : 30 | ignores trop courts : 0 | introuvables : 0

=== Qualite du corpus ===
  Articles              : 30
  Titres uniques        : 30
  Caracteres au total   : 1 137 769
  Moyenne / article     : 37 926
  Mediane / article     : 33 845
  Plus court / long     : 1261 / 132752
  Taille sur disque     : 1209.5 Ko
  Langue(s)             : fr
```

**Comment juger la qualite du corpus :**

| Indicateur | Bon signe | Mauvais signe |
| --- | --- | --- |
| Articles / titres uniques | les deux nombres sont **egaux** | des doublons subsistent |
| Moyenne de caracteres | > 5 000 | beaucoup d'ebauches |
| Les titres listes | tous lies a votre theme | sujets hors sujet → ajustez `--topics` / `--exclude` |
| Taille sur disque | 0,5 a 3 Mo pour 30 articles | > 20 Mo : vous avez collecte trop large |

### 8. Test de validation

```powershell
# 1. Le fichier existe et contient le bon nombre de lignes
(Get-Content data\raw\articles.jsonl | Measure-Object -Line).Lines

# 2. Les champs obligatoires sont presents
python -c "import json; r=json.loads(open('data/raw/articles.jsonl',encoding='utf-8').readline()); print(sorted(r.keys()))"

# 3. La relance ne retelecharge rien
python -m src.collect_wikipedia --max 30
```

Le point 3 doit afficher :

```
  Objectif deja atteint (30 >= 30). Rien a telecharger.
```

C'est la preuve que le script est bien **relancable**.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `HTTP 403 Forbidden` | User-Agent generique ou absent | mettre un contact reel dans `WIKI_USER_AGENT` | la collecte repart |
| `[-] introuvable : Grand modele de langage` | **accent manquant** dans le titre | le vrai titre est `Grand modèle de langage` ; chercher l'article sur fr.wikipedia.org et copier son titre exact | l'article est collecte |
| `HTTP 429 Too Many Requests` | requetes trop rapprochees | augmenter `WIKI_SLEEP` a `1.0` dans `.env` | plus d'erreur 429 |
| Corpus hors sujet (Avion, Besançon) | themes trop generalistes | `--topics` plus precis, et `--exclude` | les titres listes redeviennent coherents |
| `ModuleNotFoundError: wikipediaapi` | paquet absent | `python -m pip install wikipedia-api` (nom du paquet avec un tiret, import **sans**) | `python -c "import wikipediaapi"` |
| 0 article collecte | pas de reseau, ou proxy | tester `curl https://fr.wikipedia.org` | la collecte avance |

> **Piege de nommage a connaitre :** le paquet s'installe sous le nom
> `wikipedia-api` (avec tiret) mais s'importe sous le nom `wikipediaapi`
> (sans tiret). Il existe par ailleurs un paquet different appele `wikipedia`,
> ancien et non maintenu : ne le confondez pas.

### 10. Livrable de l'etape

- [ ] `data/raw/articles.jsonl` contient 20 a 50 lignes
- [ ] Chaque ligne a les champs `title`, `text`, `url`, `lang`, `license`
- [ ] Articles et titres uniques en nombre **egal**
- [ ] Les titres sont coherents avec votre theme
- [ ] Une relance n'ajoute rien
- [ ] Vos chiffres de `--stats` sont notes dans votre compte rendu

---

## Etape 3 — Pretraitement NLP

### 1. Objectif pedagogique

Nettoyer les textes, et comprendre **la lecon centrale du pretraitement** :
il depend de la tache, et un nettoyage trop agressif degrade les resultats.

### 2. Explication simple

Il n'existe pas un « texte pretraite » universel. Nos deux usages ont des
besoins **opposes** :

| Usage | Besoin | Les mots outils (« le », « de ») sont... |
| --- | --- | --- |
| Contexte donne au **LLM** | texte lisible, syntaxe intacte | **indispensables** |
| Recherche **TF-IDF** | mots porteurs de sens | du **bruit** statistique |

Le script produit donc **deux representations** de chaque document :

- `text_clean` : nettoye en douceur — accents, majuscules, ponctuation et
  mots outils **conserves**. C'est ce qui ira au LLM.
- `tokens_norm` : minuscules, sans accents, sans stop words, racinise. C'est
  ce qui alimente TF-IDF.

> **L'erreur classique du debutant :** nettoyer agressivement **une seule
> fois**, puis donner ce texte mutile au LLM. « Qui a invente le
> transformeur ? » devient « invente transformeur » : le modele perd la
> syntaxe, repond mal, et l'on accuse le modele alors que le probleme est
> dans le pretraitement.

### 3. Prerequis

- `data/raw/articles.jsonl` existe (etape 2).

### 4. Commandes exactes

```powershell
python -m src.preprocess
python -m src.preprocess --stats
python -m src.preprocess --min-chars 300     # si trop d'articles sont ecartes
```

### 5. Code complet

Fichier : **`src/preprocess.py`** (livre complet, ~480 lignes).

| Fonction | Role |
| --- | --- |
| `ensure_nltk()` | telecharge les ressources NLTK **sans faire planter** le script |
| `_heading_of()` | detecte un titre de section (deux formats !) |
| `strip_tail_sections()` | coupe bibliographie, liens externes, notes |
| `clean_text()` | nettoyage **minimal**, non destructif |
| `split_sentences()` | segmentation NLTK, repli en regex |
| `tokenize()` / `normalize_tokens()` | tokenisation, puis normalisation TF-IDF |
| `content_fingerprint()` / `deduplicate()` | detection de doublons de contenu |
| `process_record()` | produit les **deux** representations |

### 6. Explication du code

**a) Le bug qu'il faut avoir vu une fois : les titres de section.**

En lisant la documentation du wikitexte, on code naturellement ceci :

```python
RE_SECTION = re.compile(r"^\s*={2,}\s*(.+?)\s*={2,}\s*$", re.MULTILINE)
```

Puis on constate que **le filtrage des sections finales ne s'applique
jamais**. Diagnostic effectue pendant la redaction de ce TP :

```
=== has '==' markers? False
=== around 'Bibliographie' ===
"...etc..\n\nVoir aussi\nBibliographie\n(en) Dan Jurafsky, Speech and
Language Processing, Stanford, Pearson, 2008, 320 p. (ISBN 9780131873216)..."
```

`wikipedia-api` avec `ExtractFormat.WIKI` renvoie les titres de section comme
de **simples lignes de texte, sans aucun marqueur `==`**.

Consequence concrete avant correction : bibliographies et listes de liens
restaient dans le corpus, et comme elles sont riches en noms propres et en
chiffres, elles obtenaient d'excellents scores TF-IDF. La premiere reponse a
« Qu'est-ce que le traitement automatique des langues ? » etait :

```
- [3] ... Marcel Cori, Le traitement automatique des langues en question:
  des machines qui comprennent le français?, Paris, Editions Cassini, 2020,
  248p. (ISBN 978-2-84225-255-7)...
- [4] ... (ISBN 978-2-84867-824-5 et 978-2-84867-062-1, ISSN 2967-8080 ...
  DOI 10.4000/BOOKS.PUFC.29912, lire en ligne).
```

La correction gere **les deux** formats :

```python
def _heading_of(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    match = RE_SECTION.match(line)
    if match:
        return match.group(1).strip()
    # Ligne courte sans ponctuation finale : tres probablement un titre.
    if len(stripped) < 80 and stripped[-1] not in ".!?,;:":
        return stripped
    return None
```

Effet mesure : la reduction du nettoyage passe de **6,9 % a 11,4 %**, et le
nombre de chunks de **1 681 a 1 559**. Les ISBN et DOI ont disparu des
resultats.

**Lecon generale : ne faites jamais confiance au format que vous *supposez*
recevoir. Affichez les 300 premiers caracteres de vos donnees reelles.**

**b bis) Les formules mathematiques, ou la meme lecon une seconde fois.**

Apres les corrections ci-dessus, le passage le mieux classe pour « A quoi
sert l'apprentissage profond ? » etait encore :

```
- [1] Titre : Reseau de neurones artificiels a. x.
```

Un chunk dont le debut est « a. x. ». En affichant le texte brut autour de la
zone fautive, on decouvre **deux** artefacts distincts :

```
"le reseau repond \n  \n    \n      \n        a\n        x\n        +\n
 b\n      \n    \n    {\\displaystyle ax+b}\n  \n pour toute valeur de ..."
```

1. Wikipedia livre la formule **en TeX** : `{\displaystyle ax+b}`.
2. Il la livre **aussi** en texte, a raison d'**un symbole par ligne**,
   fortement indente — et **sans aucun point**.

Le plus instructif : les points de « a. x. +. b. » ne venaient pas de
Wikipedia, mais de **notre propre code**. La fonction `mark_headings()`
prenait chaque ligne courte pour un titre de section et lui ajoutait un
point. Notre correction du point (b) avait donc cree un nouveau probleme.

Trois corrections, toutes dans `src/preprocess.py` :

```python
# 1. Les formules TeX, avec deux niveaux d'accolades imbriquees
RE_TEX = re.compile(r"\{\\displaystyle(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}")

# 2. Les debris deja ponctues ("x. i., f.")
RE_MATH_NOISE = re.compile(
    r"(?:(?:[^\W\d_]{1,2}|\d{1,3}|[^\w\s]{1,3})\.[,;:\s]+){3,}"
)
```

```python
# 3. Un titre de section a au moins 3 lettres : "a", "x" et "+" n'en sont pas
if len(stripped) < 80 and stripped[-1] not in ".!?,;:":
    letters = sum(1 for char in stripped if char.isalpha())
    if letters >= 3:
        return stripped
return None
```

Et dans `src/rag.py`, les chunks sont ramenes a une seule ligne, sinon les
sauts de ligne de Wikipedia se retrouvent dans le prompt du LLM sous forme de
longues suites d'espaces :

```python
text = re.sub(r"\s+", " ", text).strip()
```

Resultat final mesure : reduction **12,7 %**, **1 345 chunks**, et le passage
precedemment fautif est devenu de la prose lisible :

```
- [1] Titre : Reseau de neurones artificiels ( x i, f ( x i)) positionne en
  dehors de la droite va influencer la decision : il aura appris le bruit en
  plus, ce qui n'est pas souhaitable. Pour eviter le surapprentissage, il
  existe une methode simple : il suffit de partager la base d'exemples en
  deux sous-ensembles.
```

**Deuxieme lecon, plus dure : une correction peut en casser une autre.** Sans
les tests de non-regression (`tests/test_preprocess.py`), la correction (3)
aurait pu reintroduire le probleme des bibliographies. C'est exactement pour
cela que le test `test_un_vrai_titre_recoit_bien_un_point` existe a cote de
`test_ne_prend_pas_une_formule_pour_un_titre`.

**b) `clean_text()` est volontairement peu agressif.**

```python
text = unicodedata.normalize("NFKC", raw)   # unifie les variantes unicode
text = text.replace("\u00a0", " ")          # espace insecable
text = text.replace("\u00ad", "")           # trait d'union conditionnel
text = RE_CONTROL.sub("", text)
text = strip_tail_sections(text)
...
```

On supprime les **artefacts** (balisage, references `[12]`, caracteres de
controle). On **ne touche pas** aux accents, a la casse ni a la ponctuation :
ce texte doit rester lisible par un humain et par un LLM.

**c) Tout a un repli.** NLTK a besoin de telecharger `punkt` et `stopwords`.
Si c'est impossible (hors ligne, proxy, CI sans reseau), on ne plante pas :

```python
RE_SENTENCE_FALLBACK = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00c0-\u00dc0-9])")
```

De meme, `FALLBACK_STOPWORDS` contient une liste interne de mots outils
francais et anglais. **Un TP doit fonctionner dans le train.**

**d) Stemming plutot que lemmatisation, et c'est assume.**

```python
from nltk.stem.snowball import SnowballStemmer
_STEMMERS[lang] = SnowballStemmer(NLTK_LANG.get(lang, "french"))
```

Le lemmatiseur WordNet de NLTK ne gere que l'anglais. Pour le francais, une
vraie lemmatisation demanderait spaCy et son modele (~16 Mo). Snowball est
integre, instantane, et suffit a rapprocher « modeles » et « modele » pour du
TF-IDF. **Limite a connaitre :** il confond aussi « modeles » et
« modeliser ». Acceptable pour de la recherche, insuffisant pour une analyse
linguistique fine.

**e) Deduplication par empreinte de contenu.**

```python
normalized = strip_accents(text.lower())
normalized = re.sub(r"\W+", " ", normalized).strip()
return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
```

On normalise **avant** de hacher : deux articles qui ne different que par la
casse, les accents ou les espaces donnent la meme empreinte. On detecte ainsi
un meme contenu publie sous deux titres differents.

### 7. Resultat attendu

```
=== Pretraitement NLP ===
  Articles lus : 30
  Doublons supprimes : 0

=== Rapport de pretraitement ===
  Entree               : 30 article(s)
  Doublons retires     : 0
  Trop courts retires  : 0
  Conserves            : 30
  Caracteres avant     : 1 137 769
  Caracteres apres     : 993 576
  Reduction            : 12.7 %
  Sortie               : data/processed/articles_clean.jsonl
```

Une reduction de **5 a 20 %** est normale. Au-dela de 40 %, votre nettoyage
est trop agressif : verifiez quelles sections sont coupees.

```powershell
python -m src.preprocess --stats
```

affiche, pour les premiers documents, un extrait du texte propre **et** les
premiers tokens normalises — vous voyez ainsi les deux representations cote
a cote.

### 8. Test de validation

```powershell
python -m pytest tests/test_preprocess.py -v
```

38 tests doivent passer. Le plus parlant :

```python
def test_les_deux_representations_sont_differentes(self, raw_articles):
    record = process_record(raw_articles[0], "fr")
    assert record["text_clean"] != record["norm_text"]
    # Le texte destine au LLM garde ses mots outils...
    assert " de " in record["text_clean"] or " la " in record["text_clean"]
    # ...alors que la version TF-IDF les a perdus.
    assert " de " not in f" {record['norm_text']} "
```

Verification manuelle que plus aucune bibliographie ne traine :

```powershell
python -c "import json; d=[json.loads(l) for l in open('data/processed/articles_clean.jsonl',encoding='utf-8')]; n=sum('ISBN' in x['text_clean'] for x in d); print(f'documents contenant encore ISBN : {n}/{len(d)}')"
```

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `LookupError: Resource punkt not found` | ressource NLTK absente | `python -m nltk.downloader punkt punkt_tab stopwords` | plus d'avertissement |
| `[!] Ressource NLTK 'punkt' indisponible` | pas de reseau | **aucune action** : le repli regex prend le relais | le script se termine |
| 0 document conserve | `--min-chars` trop eleve | `python -m src.preprocess --min-chars 300` | `Conserves` > 0 |
| Reduction > 40 % | une section utile est coupee | relire `TAIL_SECTIONS` | reduction redescend |
| Des ISBN subsistent | heuristique de titre non declenchee | verifier `_heading_of()` ; le titre fait peut-etre plus de 80 caracteres | le compteur ISBN tombe |
| `UnicodeDecodeError` | fichier lu sans encodage | toujours `encoding="utf-8"` | la lecture passe |

### 10. Livrable de l'etape

- [ ] `data/processed/articles_clean.jsonl` existe
- [ ] Chaque enregistrement a `text_clean` **et** `tokens_norm`
- [ ] Reduction entre 5 et 20 %
- [ ] `pytest tests/test_preprocess.py` : tout passe
- [ ] Vous savez expliquer pourquoi il y a **deux** representations

---

# Seance 2 — Experimenter les modeles NLP

## Etape 4 — Trois experiences comparables

### 1. Objectif pedagogique

Mesurer **concretement** ce que chaque famille de modeles apporte et coute :
NLP classique, embeddings de phrases, LLM. Savoir choisir en connaissance de
cause, plutot que prendre « le plus moderne ».

### 2. Explication simple

Trois experiences, de la plus simple a la plus lourde :

| | Technologie | Telechargement | Ce que ca comprend |
| --- | --- | --- | --- |
| **A** | TF-IDF (scikit-learn) | 0 Mo | des **mots** |
| **B** | embeddings de phrases | ~470 Mo | du **sens** |
| **C** | LLM (API ou local) | 0 Mo ou ~1 Go | **redige** du texte |

Toutes les experiences posent **les memes questions** : c'est la seule facon
de comparer honnetement.

### 3. Prerequis

- Etape 3 terminee.
- Pour B : `pip install -r requirements.txt` (avec PyTorch).
- Pour C : rien d'obligatoire — le mode `extractive` fonctionne sans cle.

### 4. Commandes exactes

```powershell
python -m src.nlp_models --exp-a          # NLP classique
python -m src.rag --build                 # construit les chunks (requis par B.2)
python -m src.nlp_models --exp-b          # embeddings + comparaison
python -m src.nlp_models --sentiment      # classification (+270 Mo, optionnel)
python -m src.embeddings                  # demonstration isolee des embeddings
python -m src.llm                         # demonstration isolee du LLM
```

### 5. Code complet

| Fichier | Contenu |
| --- | --- |
| `src/nlp_models.py` | experiences A, B et B2 |
| `src/embeddings.py` | modele de phrases + cache disque |
| `src/llm.py` | 4 fournisseurs LLM + gestion fine des erreurs |

### 6. Explication du code

#### Experience A — NLP classique

**A.1 Frequences de mots.** Le script affiche deux colonnes, sans et avec
pretraitement. Sans pretraitement, le classement est domine par « de », « la »,
« le », « et » : les mots les plus frequents de la langue, qui ne disent
**rien** du sujet. C'est la justification empirique des stop words.

**A.2 TF-IDF.** La ponderation combine deux idees :

- **TF** (*term frequency*) : plus un mot apparait dans un document, plus il
  compte ;
- **IDF** (*inverse document frequency*) : plus un mot est present dans
  **tous** les documents, moins il compte.

Un mot frequent **ici** mais rare **ailleurs** obtient un score eleve : c'est
un bon descripteur. Le script affiche aussi la densite de la matrice :

```python
density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
```

Elle est de l'ordre de 0,1 % : une matrice TF-IDF est **tres creuse**, d'ou le
stockage `sparse` de SciPy. En dense, la meme matrice occuperait des centaines
de Mo pour rien.

Deux reglages du vectoriseur merites d'etre compris :

```python
TfidfVectorizer(
    ngram_range=(1, 2),   # mots seuls ET paires : capte "reseau neurone"
    min_df=1,
    sublinear_tf=True,    # log(tf) : un mot repete 10x ne vaut pas 10x plus
)
```

**A.3 Similarite cosinus.** Elle mesure l'**angle** entre deux vecteurs, pas
leur longueur : un article long et un article court sur le meme sujet restent
proches. On met la diagonale a zero, sinon chaque document est trivialement
son propre plus proche voisin.

**A.4 Recherche.** La requete subit **exactement** la meme normalisation que
les documents — condition absolue pour que les mots se correspondent :

```python
normalized = " ".join(normalize_tokens(query, CFG.wiki_lang))
scores = cosine_similarity(vectorizer.transform([normalized]), matrix).ravel()
```

#### Experience B — Embeddings

Un embedding est un vecteur qui represente le **sens** d'un texte. Le script
compare trois paires de phrases :

| Paire | TF-IDF | Embedding |
| --- | --- | --- |
| « Le chat dort sur le canape » / « Un felin se repose sur le sofa » | tres bas | **eleve** |
| « L'apprentissage profond utilise des reseaux de neurones » / « Le deep learning repose sur des reseaux neuronaux » | bas | **eleve** |
| « Paris est la capitale de la France » / « La tarte aux pommes est delicieuse » | bas | bas |

Les deux premieres paires ont le **meme sens avec des mots differents** :
TF-IDF ne voit rien, l'embedding voit la proximite. La troisieme est basse
pour les deux : l'embedding ne dit pas « tout se ressemble ».

**Deux details d'implementation qui comptent :**

```python
vectors = self.model.encode(
    texts,
    batch_size=BATCH_SIZE,       # 16 : on menage la RAM
    normalize_embeddings=True,   # cosinus == produit scalaire
)
return np.asarray(vectors, dtype=np.float32)
```

- `normalize_embeddings=True` ramene chaque vecteur a une norme de 1. La
  similarite cosinus devient un simple produit scalaire : une multiplication
  matricielle, sans division.
- `float32` au lieu du `float64` par defaut de NumPy : **deux fois moins de
  RAM** pour l'index, sans perte utile de precision.

**Le cache disque des embeddings.** Encoder ~1 400 chunks sur CPU prend une a
trois minutes. On ne refait pas ce travail a chaque lancement :

```python
hasher.update(model_name.encode("utf-8"))
hasher.update(str(len(texts)).encode("utf-8"))
for text in texts:
    hasher.update(text.encode("utf-8"))
```

La cle depend du corpus **et** du modele : changer un seul chunk ou changer de
modele invalide automatiquement le cache. Impossible de reutiliser par erreur
des vecteurs calcules pour autre chose — un bug tres desagreable a
diagnostiquer.

#### Experience C — LLM

Quatre fournisseurs derriere **une seule** fonction `generate()` :

| Fournisseur | Cle | Reseau | Remarque |
| --- | --- | --- | --- |
| `extractive` | non | **non** | renvoie les passages ; c'est la **reference** |
| `groq` | oui (gratuite) | oui | tres rapide |
| `gemini` | oui (gratuite) | oui | **lire l'avertissement de confidentialite** |
| `ollama` | non | local | fonctionne hors ligne |

**Le mode `extractive` a une vraie valeur pedagogique.** Il ne genere rien :
il remonte les premieres phrases des passages trouves. En comparant ce mode
au mode `groq`, vous voyez exactement ce que le LLM apporte — la
reformulation et la synthese — et ce qu'il n'apporte **pas** : le contenu, qui
vient entierement de la recherche.

**La consigne systeme est la principale defense contre l'hallucination :**

```python
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
```

**L'ordre du prompt n'est pas indifferent** : le contexte **avant** la
question. Les modeles suivent mieux la consigne quand la question arrive en
dernier, juste avant la reponse attendue.

**Les erreurs sont traduites en consignes actionnables.** Un `429` brut
n'apprend rien a un etudiant ; ceci, oui :

```python
if status == 429:
    return (
        f"Quota depasse chez {provider} (HTTP 429).\n"
        f"  - Attendez quelques minutes : les quotas sont par minute ET par jour.\n"
        f"  - Ou basculez en local : LLM_PROVIDER=ollama\n"
        f"  - Ou en mode sans LLM : LLM_PROVIDER=extractive\n"
    )
```

**Le repli automatique** garantit qu'une panne de fournisseur ne casse pas
l'application, **sans masquer le diagnostic** :

```python
except LLMError as exc:
    if provider != "extractive" and fallback_to_extractive:
        result = generate_extractive(question, context)
        result.ok = False          # l'echec est signale
        result.error = str(exc)    # et le diagnostic conserve
        return result
    raise
```

#### Comment choisir son modele sans toucher au code

Tout passe par `.env` :

```ini
RETRIEVER=embeddings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
```

`src/config.py` **valide** ces valeurs au demarrage : une faute de frappe
provoque un message explicite, pas un comportement etrange.

### 7. Resultat attendu

L'experience A affiche notamment :

```
  SANS pretraitement               | AVEC pretraitement
  -------------------------------- | --------------------------------
  de                     XXXXX     | apprentissag            XXXX
  la                     XXXXX     | model                   XXXX
  les                    XXXXX     | donne                   XXXX
  ...
```

> Les nombres exacts dependent de **votre** corpus. Ce qui doit etre vrai chez
> tout le monde : la colonne de gauche est remplie de mots outils, celle de
> droite de mots du domaine.

L'experience B affiche la matrice de similarite, puis la comparaison
TF-IDF / semantique question par question, avec le **recouvrement** des
resultats et les temps moyens.

**Interpretation du recouvrement :** s'il est faible (1 ou 2 passages sur 3),
les deux methodes trouvent des choses **differentes** — c'est precisement
l'interet de les comparer, et un argument pour la recherche hybride.

### 8. Test de validation

```powershell
python -m src.embeddings
```

Doit se terminer par :

```
  Phrases 1 et 2 (meme sens, mots differents) : 0.9xx
  Phrases 1 et 3 (sens different)              : 0.1xx

  [OK] Le modele capte bien le sens, pas seulement les mots.
```

Si la similarite « meme sens » n'est **pas** superieure a « sens different »,
votre modele est inadapte a la langue du corpus : verifiez `EMBEDDING_MODEL`
(c'est le symptome typique de `all-MiniLM-L6-v2` sur du francais).

```powershell
python -m src.llm
```

Le second test (contexte vide) doit afficher un **refus** de repondre.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `ModuleNotFoundError: sentence_transformers` | version legere installee | `pip install sentence-transformers`, **ou** `RETRIEVER=tfidf` | `python -m src.embeddings` |
| Telechargement du modele tres lent | 470 Mo au premier appel | patienter ; ou choisir `all-MiniLM-L6-v2` (90 Mo) si corpus **anglais** | 2e lancement instantane |
| `No space left on device` | cache HF sur `C:` sature | `HF_HOME=./hf_cache` dans `.env` | `Get-PSDrive C` |
| Scores semantiques tous proches de 0,9 | modele anglophone sur du francais | passer au modele **multilingue** | les scores se differencient |
| `MemoryError` a l'encodage | lot trop gros | baisser `BATCH_SIZE` (16 → 8) | l'encodage se termine |
| `401` sur l'API | cle absente ou mal copiee | recoller la cle dans `.env`, sans guillemets ni espace | `python -m src.llm` |
| `429` | quota depasse | attendre, ou `LLM_PROVIDER=extractive` | la reponse revient |
| `404` sur le modele | nom de modele retire par le fournisseur | relever un nom valide dans sa documentation, corriger `GROQ_MODEL` | la reponse revient |
| `ConnectionError` vers Ollama | serveur non demarre | `ollama serve`, puis `ollama pull qwen2.5:1.5b-instruct` | `ollama list` |

### 10. Livrable de l'etape

- [ ] Experience A executee ; les deux colonnes de frequences sont notees
- [ ] Vous savez expliquer TF et IDF avec vos mots
- [ ] Experience B executee ; la matrice de similarite est notee
- [ ] Le recouvrement TF-IDF / semantique est releve
- [ ] Au moins un fournisseur LLM teste (meme `extractive`)
- [ ] Tableau personnel « avantages / limites » rempli pour A, B et C

---

# Seance 3 — RAG et interface web

## Etape 5 — Construire le systeme RAG

### 1. Objectif pedagogique

Assembler la chaine complete : decoupage, indexation, recherche, construction
du prompt, generation, affichage des sources. Et surtout : **faire refuser le
systeme quand il ne sait pas**.

### 2. Explication simple

Un article Wikipedia fait 20 000 a 130 000 caracteres. On ne peut ni le
donner entier a un petit LLM, ni l'indexer tel quel : l'embedding d'un
article complet « moyenne » tous ses sujets et ne ressemble plus a rien de
precis. On le decoupe donc en **chunks** d'environ 900 caracteres, qui parlent
chacun d'une idee.

Les chunks se **chevauchent** : sans cela, une phrase cle coupee en deux perd
son sens dans les deux morceaux.

### 3. Prerequis

- Etape 3 terminee.

### 4. Commandes exactes

```powershell
python -m src.rag --build                                   # chunks + index
python -m src.rag --info                                    # etat du pipeline
python -m src.rag --ask "Qu'est-ce que l'apprentissage profond ?"
python -m src.rag --compare "Qu'est-ce qu'un reseau de neurones ?"
python -m src.rag --rebuild                                 # force la reconstruction
python -m src.vector_store                                  # teste la base vectorielle
```

### 5. Code complet

| Fichier | Contenu |
| --- | --- |
| `src/rag.py` | `chunk_text`, `TfidfRetriever`, `EmbeddingRetriever`, `RagPipeline` |
| `src/vector_store.py` | `NumpyVectorStore`, `ChromaVectorStore` |

### 6. Explication du code

#### a) Le decoupage — et un deuxieme bug instructif

Version naive du chevauchement : reprendre les `overlap` derniers
**caracteres** du chunk precedent.

```python
tail = current[-overlap:] if overlap and current else ""   # MAUVAIS
```

**Resultat reel observe pendant ce TP :**

```
- [1] Titre : Traitement automatique des langues
      st l'utilisation du traitement automatique des langues sur...
- [2] Titre : Traitement automatique des langues
      tégorisation de documents : Activite qui consiste a classer...
```

« st l'utilisation », « tégorisation » : les chunks commencent **au milieu
d'un mot**. C'est illisible pour un humain, et surtout l'embedding d'un
fragment tronque represente mal son sens — la recherche semantique s'en
trouve degradee.

La correction prend le chevauchement **par phrases entieres** :

```python
def flush() -> list[str]:
    if current:
        chunks.append(" ".join(current).strip())
    tail: list[str] = []
    tail_len = 0
    for sentence in reversed(current):
        if tail_len + len(sentence) + 1 > overlap:
            break
        tail.insert(0, sentence)
        tail_len += len(sentence) + 1
    return tail
```

Effet mesure : 1 559 → **1 372 chunks**, taille maximale ramenee de 1 047 a
**900** caracteres, et plus aucun debut tronque. Les nettoyages
supplementaires decrits a l'etape 3 (point « b bis » : formules
mathematiques et blancs) amenent le corpus final a **1 345 chunks**.

**La borne reelle d'un chunk est `size + overlap`**, et non `size` : un chunk
commence par le report (jusqu'a `overlap`) puis recoit des phrases tant que le
total reste sous `size`. Le test correspondant verifie cette borne exacte —
la connaitre evite de dimensionner trop juste le contexte du LLM.

#### b) Les deux moteurs de recherche

**TF-IDF** reutilise le `norm_text` calcule a l'etape 3 : on ne refait pas le
travail de normalisation.

```python
scores = linear_kernel(query_vector, self.matrix).ravel()
```

Les lignes TF-IDF de scikit-learn sont normalisees L2 : le produit scalaire
(`linear_kernel`) **est** deja la similarite cosinus. Pas besoin de
`cosine_similarity`, qui recalculerait les normes.

**Embeddings** : le vecteur de la question est compare a l'index.

```python
best = np.argpartition(-scores, top_k - 1)[:top_k]
best = best[np.argsort(-scores[best])]
```

`argpartition` trouve les `k` meilleurs **sans trier** tout le tableau, puis
on ne trie que ces `k` valeurs : en O(n) au lieu de O(n log n). Sur 1 345
chunks c'est anecdotique ; sur un million, c'est decisif. Prenez l'habitude.

**Le repli automatique** evite l'ecran d'erreur :

```python
try:
    return EmbeddingRetriever(chunks, rebuild=rebuild)
except Exception as exc:
    print("  [!] Recherche semantique indisponible ... Repli sur TF-IDF.")
    return TfidfRetriever(chunks)
```

Mieux vaut une recherche lexicale qui fonctionne qu'une application morte.

#### c) La base vectorielle : pourquoi NumPy par defaut

```python
scores = self._vectors @ query
```

Une ligne. Les vecteurs du corpus sont deja normalises, donc le produit
matriciel donne directement les similarites cosinus. Sur 1 345 x 384
flottants, cela prend environ 1 ms et le resultat est **exact**.

ChromaDB est fourni (`VECTOR_BACKEND=chroma`) parce que savoir brancher une
vraie base vectorielle est une competence attendue en entreprise. Noter la
conversion necessaire :

```python
# Chroma renvoie une DISTANCE cosinus ; on repasse en SIMILARITE
# pour que les scores soient comparables au backend NumPy.
distance = float(result["distances"][0][rank - 1])
... score=1.0 - distance
```

Oublier cette conversion donne un classement **inverse**, un bug classique.

#### d) Le garde-fou : le coeur du TP

```python
# Si le meilleur passage est trop faible, la question est hors corpus.
# On refuse AVANT d'appeler le LLM. Double benefice : on ne consomme
# pas de quota, et on supprime la principale source d'hallucination
# (un LLM a qui on donne un contexte hors sujet brode quand meme).
if not passages or best_score < min_score:
    return RagAnswer(
        question=question,
        answer=NO_ANSWER,
        sources=[],            # AUCUNE source citee
        has_context=False,
        ...
    )
```

Trois proprietes a retenir :

1. le refus est **anterieur** a l'appel du LLM ;
2. `sources=[]` : on ne cite **jamais** de source quand on ne repond pas — une
   source affichee a cote d'un « je ne sais pas » serait trompeuse ;
3. `has_context=False` permet a l'interface d'afficher un message distinct.

#### e) Le prompt numerote

```python
blocks.append(f"[{passage.rank}] Titre : {passage.title}\n{passage.text}")
```

La numerotation `[1]`, `[2]` sert deux fins : le LLM peut citer ses sources,
et l'utilisateur peut verifier chaque affirmation. L'interface affiche
d'ailleurs le contexte **exact** transmis au modele : toute information de la
reponse absente de ce contexte est, par definition, une hallucination.

### 7. Resultat attendu

```
=== Construction de l'index RAG ===
  30 article(s) -> 1345 chunk(s)
  Taille des chunks : min 145 / moyenne 774 / max 900 car.
  Sortie : data/processed/chunks.jsonl
```

Avec `--compare`, les trois approches s'affichent cote a cote. **Observation
reelle sur le corpus de reference**, et c'est la plus instructive du TP.

Question posee : « Qu'est-ce que le traitement automatique des langues ? »
Classement **mesure** de TF-IDF :

| Rang | Score | Passage |
| --- | --- | --- |
| 1 | 0,1969 | « Etude bibliometrique du traitement automatique des langues... » |
| 2 | 0,1849 | « Systemes de tutorat intelligents : ... » (liste d'applications) |
| 3 | 0,1374 | « Methodes d'apprentissage supervise. Boosting... » (**autre article**) |
| **4** | **0,1344** | « **Le traitement automatique des langues (TAL ou TALN), en anglais...** » |

Le passage qui contient la **definition** — exactement ce que demande la
question — n'arrive qu'en **quatrieme** position, derriere une etude
bibliometrique, une liste d'applications, et meme une liste de methodes
tiree d'un **autre article**.

**Pourquoi ?** TF-IDF compte les mots. Le passage bibliometrique repete
« traitement automatique des langues » plusieurs fois : il gagne. Le passage
de definition ne l'ecrit qu'une fois, puis utilise « TAL », « TALN » et
« NLP » — des sigles que TF-IDF traite comme des mots **sans aucun rapport**
avec la question.

Les embeddings, eux, savent que « TAL » et « traitement automatique des
langues » designent la meme chose. Classement **mesure** de la recherche
semantique, sur la **meme question** et le **meme corpus** :

| Rang | Score | Passage |
| --- | --- | --- |
| 1 | 0,8213 | « Elle joue un role central dans le traitement automatique du langage naturel (TALN)... » (*Linguistique informatique*) |
| 2 | 0,8027 | « La linguistique informatique, ou linguistique computationnelle contribue... » (*Linguistique informatique*) |
| **3** | **0,7773** | « **Le traitement automatique des langues (TAL ou TALN), en anglais...** » |
| 4 | 0,7384 | « Champ d'etude. Trois domaines principaux... » (*Linguistique informatique*) |

Passages communs aux deux methodes : **1 sur 4**.

**Le resultat est un progres reel, mais partiel — et il faut le dire ainsi.**

Ce qui s'ameliore, mesure :

1. le passage de definition **remonte de la 4e a la 3e place** ;
2. surtout, il passe de **68 %** du meilleur score (0,1344 / 0,1969) a
   **95 %** (0,7773 / 0,8213). En lexical, il etait loin derriere ; en
   semantique, il est pratiquement a egalite avec la tete du classement ;
3. le passage **hors sujet** tire d'un autre article (« Methodes
   d'apprentissage supervise », 3e en lexical) **disparait du top 4**. Les
   quatre passages semantiques traitent tous du sujet demande.

Ce qui ne s'ameliore pas :

4. la definition **n'arrive toujours pas premiere**. Les deux premieres places
   sont prises par l'article voisin *Linguistique informatique*, qui parle du
   TALN sans le definir ;
5. et comme `eval/questions.json` attend le titre « Traitement automatique des
   langues » pour cette question, ce classement fait **baisser le `mrr`** de
   0,875 a 0,583 (etape 8). La metrique juge le **titre de l'article**, pas la
   pertinence du **passage** : ici, elle penalise un resultat qui est
   pourtant meilleur a la lecture. C'est une limite de la metrique, pas du
   moteur — et c'est exactement le genre de nuance qu'une evaluation chiffree
   oblige a regarder.

### Et sur une question vraiment acronymique ?

Le raisonnement ci-dessus invoque les sigles, mais la question testee
contenait encore le nom complet. Posons donc la question **purement
acronymique**, celle ou TF-IDF ne peut rien faire — mesure sur le meme corpus :

**« Qu'est-ce que le TAL ? »**

| | Rang du bon article | Top 1 obtenu | Score du top 1 |
| --- | --- | --- | --- |
| TF-IDF | **absent du top 5** | *Exploration de donnees* | 0,1347 |
| Semantique | **3e** | *Anglais* (phonetique) | 0,4954 |

Deux enseignements, et le second est le plus important du TP.

1. **Le semantique trouve ce que le lexical ne trouve pas du tout.** « TAL »
   n'apparait presque jamais en clair : TF-IDF ne ramene pas le bon article,
   meme en cherchant jusqu'au 5e rang. Le semantique le place 3e. C'est bien
   l'argument attendu — mais notez que son top 1 est *Anglais*, un passage de
   phonetique : **aucun des deux moteurs ne classe la bonne reponse premiere.**

2. **TF-IDF repond quand meme, et c'est le pire scenario.** Son meilleur score
   vaut 0,1347, soit **au-dessus** de `MIN_SCORE_TFIDF = 0,13`. Le garde-fou
   ne se declenche donc pas : l'assistant repond avec assurance a partir d'un
   passage sur l'exploration de donnees. Le semantique, lui, fait entrer le
   bon passage dans le contexte transmis au LLM.
   **Pour un RAG, ce qui compte n'est pas d'etre premier, c'est d'etre dans
   le top-k** : les 4 passages partent ensemble dans le prompt, et le LLM
   choisit. Un moteur qui place la bonne reponse 3e sur 4 fait le travail ;
   un moteur qui ne la ramene pas du tout, mais passe le seuil, produit une
   reponse fausse et sourcee — exactement ce que ce TP cherche a eviter.

**Consequence pour la calibration du seuil.** Le 0,4954 qui sert de borne
basse a `MIN_SCORE_EMBEDDINGS` (voir l'etape 8) est le score d'un passage
**hors sujet**, pas celui de la bonne reponse. Garder le seuil sous cette
valeur reste le bon choix — c'est ce qui permet au bon passage, classe 3e,
d'atteindre le LLM — mais il faut savoir que l'on calibre sur le score du
**meilleur passage retourne**, quelle que soit sa pertinence, puisque c'est
exactement ce que teste le garde-fou.

Pour completer, deux variantes mesurees :

| Question | Bon article en TF-IDF | Bon article en semantique |
| --- | --- | --- |
| « Qu'est-ce que le TAL ? » | absent du top 5 | **3e** |
| « Qu'est-ce que le traitement automatique des langues ? » | **1er** (article), definition 4e | 3e |
| « Que signifie NLP ? » | **1er** | **1er** |

Le semantique gagne quand la question est **acronymique ou reformulee**, et
perd quand elle **reprend litteralement** les mots du corpus. Cela explique le
`mrr` plus faible : les questions de `eval/questions.json` emploient surtout
les noms complets, le terrain favorable a TF-IDF.

**Pourquoi le semantique perd-il en moyenne ?** Une hypothese, presentee comme
telle — elle n'a pas ete testee, et c'est une bonne piste d'approfondissement :
nos chunks font 900 caracteres et contiennent souvent **plusieurs sujets**. Un
embedding resume tout le passage en **un seul** vecteur de 384 dimensions : sur
un chunk melange, ce vecteur devient une moyenne floue, proche de beaucoup de
questions sans etre tres proche d'aucune. TF-IDF, lui, profite au contraire de
la repetition litterale d'un terme dans un passage long. On s'attendrait donc
a voir l'avantage s'inverser avec des chunks plus courts et mono-sujet
(`CHUNK_SIZE=400`), ou avec un modele plus grand. **Mesurez-le** plutot que de
me croire : c'est une experience de 5 minutes avec `--compare`.

**La morale du TP n'est donc pas « le semantique corrige le lexical », mais
« mesurez les deux ».** L'avantage le plus net des embeddings sur ce corpus
n'est pas ce classement : c'est leur capacite a **separer** les questions
legitimes des questions hors sujet, et a comprendre les sigles. Sur
« Qu'est-ce que le TAL ? », TF-IDF donne 0,1347 — a peine au-dessus de son
seuil de 0,13 — tandis qu'une question hors sujet sur l'angine obtient 0,1390,
soit **plus**. Les embeddings donnent 0,4954 contre 0,2508. Le detail chiffre
est a l'etape 8.

**Verifiez-le sur votre corpus avec `--compare`, et comparez les deux
classements** : c'est le resultat le plus parlant du TP.

### 8. Test de validation

```powershell
python -m pytest tests/test_rag.py -v
python -m src.vector_store
```

Le test decisif :

```python
def test_refuse_de_repondre_hors_corpus(self, pipeline):
    """LE test le plus important : ne pas inventer."""
    answer = pipeline.answer(
        "Quel est le resultat du match de football d'hier soir ?",
        top_k=3, min_score=0.25,
    )
    assert not answer.has_context
    assert answer.answer == NO_ANSWER
    assert answer.sources == []
```

Verification manuelle :

```powershell
python -m src.rag --ask "Quelle est la recette de la tarte tatin ?"
```

Doit repondre « Je ne trouve pas cette information... » et afficher un score
nettement **inferieur** a `MIN_SCORE`.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `[X] Aucun chunk disponible` | index non construit | `python -m src.rag --build` | `python -m src.rag --info` |
| `Index incoherent (vecteurs != enregistrements)` | index corrompu | `python -m src.rag --rebuild` | l'index se recharge |
| Repond a **tout**, meme hors sujet | seuil trop bas | monter celui du moteur **actif** : mesure `0.13` en `tfidf`, `0.43` en `embeddings` | la tarte tatin est refusee |
| Refuse des questions **legitimes** | seuil trop haut | descendre celui du moteur **actif** ; ne descendez pas sous 0,12 (`tfidf`) ni sous 0,37 (`embeddings`) | la question passe |
| Scores incoherents avec ChromaDB | distance non convertie en similarite | verifier `1.0 - distance` | les scores redeviennent dans [0,1] |
| `chromadb` ne s'installe pas (Windows) | compilation native | **rester sur** `VECTOR_BACKEND=numpy` | l'index fonctionne |
| Index reconstruit a chaque lancement | taille du corpus modifiee | normal ; sinon verifier le cache `emb_*.npy` | 2e lancement rapide |

> **Les seuils TF-IDF et semantiques n'ont pas la meme echelle.** Reglez
> `MIN_SCORE` **apres** avoir choisi votre moteur, et mesurez les deux types
> d'erreur avec `python -m src.evaluate` (etape 8).

### 10. Livrable de l'etape

- [ ] `data/processed/chunks.jsonl` existe
- [ ] Taille maximale des chunks <= `CHUNK_SIZE + CHUNK_OVERLAP`
- [ ] `--compare` execute : les trois approches sont observees
- [ ] Une question hors corpus est **refusee**, **sans** source
- [ ] `MIN_SCORE` regle et justifie dans votre compte rendu

---

## Etape 6 — Developper le chatbot web

### 1. Objectif pedagogique

Construire une interface utilisable : saisie, reponse, **sources**,
historique, indicateur de chargement, et gestion propre des erreurs.

### 2. Explication simple

On utilise **Streamlit** : on ecrit du Python, il produit la page web. Pas de
HTML ni de JavaScript a maintenir, ce qui garde le TP centre sur le NLP.

Une particularite a comprendre tout de suite : **Streamlit re-execute le
script entier a chaque interaction**. Tout etat doit donc vivre dans
`st.session_state`, sinon il disparait a chaque clic.

### 3. Prerequis

- Etape 5 terminee.

### 4. Commandes exactes

```powershell
streamlit run app.py
streamlit run app.py --server.port 8502    # si 8501 est occupe
python app_gradio.py                       # variante Gradio (port 7860)
```

Pour arreter : `Ctrl+C` dans le terminal.

### 5. Code complet

| Fichier | Role |
| --- | --- |
| `app.py` | interface Streamlit (local, Docker) |
| `app_gradio.py` | interface Gradio (Hugging Face Spaces) |

### 6. Explication du code

**a) Le cache de ressources evite de recharger 470 Mo a chaque clic.**

```python
@st.cache_resource(show_spinner=False)
def get_pipeline(retriever_kind: str):
    ...
```

`retriever_kind` est un **argument** et non une lecture de `CFG` : Streamlit
invalide ainsi le cache automatiquement quand l'utilisateur change de moteur
dans la barre laterale. Sans ce cache, chaque question rechargerait le modele.

**b) `set_page_config` doit etre le premier appel Streamlit**, sinon
l'application leve une exception.

**c) Les couleurs s'adaptent au theme clair/sombre.**

```css
background: rgba(128, 128, 128, 0.10);
```

Un gris translucide reste lisible sur fond blanc comme sur fond noir. Fixer
`background: #ffffff` donnerait du texte blanc sur blanc en theme sombre —
erreur tres fréquente.

**d) Trois etats d'affichage distincts**, et c'est le point important :

| Etat | Condition | Affichage |
| --- | --- | --- |
| Reponse normale | `has_context` | reponse + sources + contexte + mesures |
| **Aucun passage pertinent** | `not has_context` | bandeau orange + score obtenu + conseils |
| Erreur technique | `fatal_error` | message rouge + type d'exception |

```python
if not turn["has_context"]:
    st.markdown('<div class="msg-bot no-answer">...')
    st.caption(
        f"Meilleur score obtenu : {turn['best_score']:.3f} "
        f"(seuil requis : {min_score}). Baissez le seuil dans la barre "
        "laterale, reformulez, ou collectez des articles sur ce theme."
    )
```

On ne dit pas seulement « je ne sais pas » : on dit **pourquoi**, avec le
score, et **comment** corriger. C'est ce qui distingue une interface utile.

**e) L'echec d'un LLM n'efface pas le diagnostic.**

```python
if not turn.get("llm_ok", True):
    with st.expander("Le fournisseur LLM a echoue (repli extractif)"):
        st.warning(turn.get("llm_error", "Erreur inconnue"))
```

L'utilisateur obtient quand meme une reponse (repli extractif), **et** peut
deplier la cause exacte.

**f) La transparence du RAG.** Un panneau depliable montre les passages
exacts transmis au modele :

```python
st.caption(
    "C'est exactement le contexte recu par le modele. Si la reponse "
    "contient une information absente d'ici, c'est une hallucination."
)
```

**g) L'interface ne doit jamais planter.**

```python
except Exception as exc:  # noqa: BLE001 - l'UI ne doit jamais crasher
```

Attraper `Exception` est generalement une mauvaise pratique — ici c'est la
frontiere de l'application : on preferera toujours un message d'erreur lisible
a une trace de pile sur la page.

**h) Le mode demonstration sans cle** fonctionne parce que
`LLM_PROVIDER=extractive` est le **defaut** du projet. On peut faire la demo
du RAG complet sans aucun compte chez personne.

### 7. Resultat attendu

Au lancement :

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

La page affiche l'en-tete, le nombre de passages charges, trois questions
suggerees et le formulaire. Apres une question : la reponse, les sources
Wikipedia cliquables avec leur licence, le contexte depliable, et la ligne de
mesures (fournisseur, temps de recherche, temps de generation, score).

### 8. Test de validation — tests manuels a faire dans l'ordre

| # | Action | Resultat attendu |
| --- | --- | --- |
| 1 | Cliquer une question suggeree | reponse + au moins une source |
| 2 | Demander « Quelle est la recette de la tarte tatin ? » | bandeau orange, **aucune** source, score affiche |
| 3 | Deplier « Voir les N passages » | les passages correspondent a la reponse |
| 4 | Poser 3 questions de suite | l'historique s'empile, la plus recente en haut |
| 5 | Cliquer « Effacer la conversation » | l'historique se vide |
| 6 | Basculer `embeddings` → `tfidf` | le corpus se recharge, les scores changent |
| 7 | Mettre `MIN_SCORE` a 0,95 puis reposer la question 1 | elle est maintenant refusee |
| 8 | Choisir `groq` sans cle dans `.env` | avertissement en barre laterale, puis repli extractif annonce |
| 9 | Cliquer une URL de source | l'article Wikipedia s'ouvre |
| 10 | Reduire la fenetre a une largeur de telephone | la mise en page reste lisible |

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `Port 8501 is already in use` | une instance tourne deja | `--server.port 8502`, ou `Get-Process -Name streamlit \| Stop-Process` | la page s'ouvre |
| `set_page_config() can only be called once` | appel non place en premier | remonter l'appel en tete de fichier | l'app demarre |
| L'app recharge le modele a chaque clic | `@st.cache_resource` absent | ajouter le decorateur | 2e question instantanee |
| La **premiere** question reste bloquee ~1,5 min, les suivantes sont instantanees | le modele d'embeddings se chargeait paresseusement, a la 1re requete (mesure : 94,6 s contre 0,19 s ensuite) | corrige : `EmbeddingRetriever.warmup()` charge le modele pendant l'ecran de chargement | `first_query_s` tombe de 94,61 a 0,108 dans `src.evaluate` |
| Texte illisible en theme sombre | couleurs codees en dur | utiliser `rgba(128,128,128,0.1)` | lisible dans les deux themes |
| `Aucun corpus trouve` | pipeline non execute | `collect` → `preprocess` → `rag --build` | le compteur de passages > 0 |
| L'historique disparait | etat hors de `session_state` | tout stocker dans `st.session_state` | l'historique persiste |
| Page blanche | exception avant le premier rendu | lire le terminal | la trace indique la ligne |

### 10. Livrable de l'etape

- [ ] L'application repond a une question avec ses sources
- [ ] Les 10 tests manuels passent
- [ ] Le message « aucun passage pertinent » s'affiche correctement
- [ ] L'historique fonctionne
- [ ] Le mode demonstration marche **sans** cle API
- [ ] Capture d'ecran jointe au compte rendu

---

## Etape 7 — Organiser le projet

### 1. Objectif pedagogique

Adopter une arborescence professionnelle, et comprendre la regle qui la
gouverne : **le code se versionne, les donnees generees ne se versionnent
pas**.

### 2. Explication simple

Un lecteur exterieur doit comprendre votre projet en lisant l'arborescence.
Et un depot Git doit contenir ce qui est **source** — pas ce qui est
**produit** par le code.

### 3. Prerequis

- Etapes 1 a 6 terminees.

### 4. Commandes exactes

```powershell
tree /F /A | Select-Object -First 60
Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum
```

### 5. Code complet

Voir la section « Structure du projet » du `README.md`, qui documente chaque
fichier.

### 6. Explication du code

**La separation code / donnees**, dans `.gitignore` :

```gitignore
data/raw/*
data/processed/*
data/index/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/index/.gitkeep

# Exception : le mini-corpus de demonstration est versionne (petit, < 1 Mo)
!data/processed/demo_chunks.jsonl
```

Trois idees :

1. le contenu de `data/` est **exclu** : il est reproductible en relancant le
   pipeline, et il pese vite plusieurs Mo ;
2. les fichiers `.gitkeep` sont **conserves** : Git ne versionne pas les
   dossiers vides, mais l'application a besoin que ces dossiers existent ;
3. **une exception assumee** : `demo_chunks.jsonl` (environ 600 Ko) est
   versionne, car l'hebergement gratuit n'a pas de disque persistant et doit
   pouvoir demarrer avec un corpus (voir etape 11).

**Pourquoi un fichier `src/config.py` separe ?** Parce que toute la
configuration est lue a **un seul** endroit. Comparez :

```python
# Sans config centrale : chaque module relit l'environnement a sa facon
model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")   # dispersé, non validé

# Avec config centrale
from src.config import CFG
model = CFG.embedding_model                                  # validé une fois
```

**Pourquoi deux applications (`app.py` et `app_gradio.py`) ?** Ce n'est pas de
la duplication gratuite : les deux ne font que de l'affichage et appellent le
**meme** `RagPipeline`. La logique metier n'existe qu'une fois. La raison est
exposee a l'etape 11 : Streamlit n'est plus un SDK de Hugging Face Spaces.

### 7. Resultat attendu

L'arborescence du `README.md`, avec `src/` (le code), `data/` (les donnees
generees, ignorees), `tests/`, `eval/`, `docs/` et `.github/`.

### 8. Test de validation

```powershell
# Aucun fichier de donnees volumineux ne doit etre suivi par git
git status --short
git ls-files data/ | ForEach-Object { "$_  $((Get-Item $_).Length / 1KB) Ko" }
```

Seuls les `.gitkeep` et `demo_chunks.jsonl` doivent apparaitre.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| Depot de plusieurs centaines de Mo | `data/` versionne | `git rm -r --cached data/` puis committer | `git count-objects -vH` |
| `data/raw` absent apres un clone | dossier vide non versionne | les `.gitkeep` le resolvent ; `config.py` cree les dossiers au besoin | le clone fonctionne |
| `ModuleNotFoundError: src` | `src/__init__.py` manquant | le creer (il est fourni) | `python -c "import src"` |
| Des `__pycache__` dans le depot | `.gitignore` incomplet | ajouter `__pycache__/` | `git status` propre |

### 10. Livrable de l'etape

- [ ] L'arborescence correspond au `README.md`
- [ ] `git status` ne montre aucun fichier de donnees volumineux
- [ ] Les `.gitkeep` sont presents
- [ ] Chaque fichier de `src/` a un role unique et documente

---

# Seance 4 — Qualite et conteneurisation

## Etape 8 — Tests et evaluation

### 1. Objectif pedagogique

Ecrire des tests automatises qui tournent **hors ligne**, et **mesurer** la
qualite du systeme au lieu de l'estimer a l'oeil.

### 2. Explication simple

Deux choses differentes :

- **Les tests** (`pytest`) verifient que le **code** fait ce qu'on a ecrit.
  Ils sont binaires : ils passent ou ils echouent.
- **L'evaluation** (`src/evaluate.py`) mesure la **qualite des reponses** sur
  un jeu de questions. Elle produit des nombres a interpreter.

Regle d'or : **les tests ne doivent jamais dependre du reseau ni d'un modele
telecharge.** Sinon ils echouent dans la CI, ou dans le train. D'ou, dans
`tests/conftest.py` :

```python
# /!\ Doit venir avant les imports de src.*
os.environ.setdefault("RETRIEVER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "extractive")
```

Ces lignes sont **avant** les imports parce que la configuration est lue une
seule fois, a l'import du module.

### 3. Prerequis

- Etapes 1 a 5 terminees.

### 4. Commandes exactes

```powershell
python -m pytest                          # tous les tests
python -m pytest -v                       # detail
python -m pytest tests/test_rag.py -v     # un fichier
python -m pytest -k "garde_fou or refuse" # par mot-cle
python -m pytest --tb=long                # traces completes

python -m src.evaluate                    # evaluation (moteur de .env)
python -m src.evaluate --retriever tfidf
python -m src.evaluate --compare          # tfidf CONTRE embeddings
```

### 5. Code complet

| Fichier | Contenu |
| --- | --- |
| `tests/conftest.py` | mini-corpus et fixtures partagees |
| `tests/test_preprocess.py` | nettoyage, tokenisation, deduplication (38 tests) |
| `tests/test_rag.py` | chunking, recherche, **garde-fou** (32 tests) |
| `tests/test_llm.py` | 4 fournisseurs simules, erreurs HTTP (24 tests) |
| `tests/test_config.py` | validation, **securite des cles**, seuils (22 tests) |
| `eval/questions.json` | jeu de questions avec verite terrain |
| `src/evaluate.py` | harnais de mesure |

### 6. Explication du code

**a) Simuler une API sans l'appeler.** On remplace `requests.post` :

```python
class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(self._payload)
    def json(self):
        return self._payload
```

On peut alors tester le quota depasse, la cle invalide, la panne serveur, le
delai depasse — **sans cle API, sans Internet, sans consommer de quota**. Un
test qui appelle une vraie API est lent, instable et coute de l'argent.

**b) Le test le plus important du projet :**

```python
def test_refuse_de_repondre_hors_corpus(self, pipeline):
    """LE test le plus important : ne pas inventer."""
    answer = pipeline.answer(
        "Quel est le resultat du match de football d'hier soir ?",
        top_k=3, min_score=0.25,
    )
    assert not answer.has_context
    assert answer.answer == NO_ANSWER
    assert answer.sources == []
```

**c) Un test qui protege vos secrets** — il verifie le projet, pas le code :

```python
def test_env_example_ne_contient_aucune_vraie_cle(self):
    exemple = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    for ligne in exemple.splitlines():
        if ligne.strip().startswith(("GROQ_API_KEY", "GEMINI_API_KEY")):
            cle, _, valeur = ligne.partition("=")
            assert valeur.strip() == "", (
                f"{cle} contient une valeur dans .env.example : "
                "ne commitez jamais une cle !"
            )
```

**d) Les metriques mesurees**, et comment les lire :

| Metrique | Definition | Cible |
| --- | --- | --- |
| `hit_at_k` | la source attendue est-elle dans les `k` passages ? | proche de **1.0** |
| `mrr` | 1/rang de la bonne source, en moyenne | proche de **1.0** |
| `keyword_coverage_mean` | part des mots attendus presents dans la reponse | eleve |
| **`refusal_rate_out_of_corpus`** | part des questions hors corpus **refusees** | **exactement 1.0** |
| **`false_refusals_in_corpus`** | questions legitimes refusees a tort | **exactement 0** |
| `mean_retrieval_time_s` | temps de recherche | < 0,5 s |
| `memory_mb_end` | memoire du processus (approximative) | — |

Les deux metriques en gras forment un couple : elles se **contredisent**
quand on bouge `MIN_SCORE`. Monter le seuil ameliore la premiere et degrade la
seconde. Le bon reglage est celui qui obtient **1.0 et 0 simultanement**.

> **Note d'honnetete sur la memoire.** `memory_mb_end` utilise `psutil` s'il
> est installe (vraie memoire resident), sinon `tracemalloc`, qui ne compte
> que les allocations **Python** et **sous-estime** donc : il ignore la
> memoire de PyTorch. Le chiffre est indicatif, et le code le dit.

### 7. Resultat attendu

```
============================= 117 passed in 4.26s ==============================
```

**Mesure de reference** (corpus de 30 articles / 1 345 chunks, `tfidf`,
`extractive`, environnement d'annexe C) :

| Metrique | Windows natif | Docker Linux |
| --- | --- | --- |
| `n_chunks` | 1 345 | 1 345 |
| `index_build_s` | 24,43 | 14,33 |
| `hit_at_k` (k=4) | **1.0** | **1.0** |
| `mrr` | **0,875** | **0,875** |
| `keyword_coverage_mean` | 0,834 | 0,834 |
| `refusal_rate_out_of_corpus` | **1.0** | **1.0** |
| `false_refusals_in_corpus` | **0** | **0** |
| `mean_total_time_s` | 0,436 | 0,251 |
| `mean_retrieval_time_s` | 0,684 | 0,392 |
| `memory_mb_end` | 101,3 | 101,0 |

Les metriques de **qualite** sont rigoureusement identiques dans les deux
environnements ; seuls les **temps** changent (Windows natif est 1,3 a 2,2 fois
plus lent qu'un conteneur Linux, voir annexe C).

#### Les deux moteurs, mesures dans la meme execution

`python -m src.evaluate --compare` evalue `tfidf` **puis** `embeddings` sur le
meme corpus, dans le meme processus. Mesure du 16 octobre 2026, conteneur
`wikinlp:full` (environnement **C1** de l'annexe C) :

| Metrique | `tfidf` | `embeddings` | Qui gagne ? |
| --- | --- | --- | --- |
| `min_score` applique | 0,13 | 0,43 | seuil **par moteur** |
| `index_build_s` | 17,43 | 0,18 | embeddings (index relu du disque) |
| Construction initiale de l'index | incluse ci-dessus | **3 min 02 s** | tfidf |
| `hit_at_k` (k=4) | **1.0** | **1.0** | egalite |
| `mrr` | **0,875** | 0,583 | **tfidf** |
| `keyword_coverage_mean` | **0,834** | 0,768 | **tfidf** |
| `refusal_rate_out_of_corpus` | **1.0** | **1.0** | egalite |
| `false_refusals_in_corpus` | **0** | **0** | egalite |
| `first_query_s` (1re question, a froid) | 2,22 | **94,61** | **tfidf** |
| Recherche **a chaud**, par question | 0,015 | 0,19 | **tfidf**, ~13x |
| `mean_retrieval_time_s` (melange froid et chaud) | 0,456 | 13,47 | metrique trompeuse, voir ci-dessous |
| `mean_total_time_s` (idem) | 0,291 | 8,48 | metrique trompeuse |
| `memory_mb_end` (`tracemalloc`) | 103,0 | 165,8 | sous-estime les deux |
| Pic de RSS reel (`VmHWM`) * | 312 Mo | **1 495 Mo** | **tfidf** |
| Index sur disque | aucun | 5,2 Mo | tfidf |

\* Seule ligne mesuree dans deux executions **separees** (une par moteur) :
`--compare` evalue les deux moteurs dans le meme processus, ce qui rendrait
leurs pics de memoire indiscernables. Tout le reste du tableau vient d'une
seule execution.

**Lisez ce tableau avant de conclure que « le semantique, c'est mieux ».**
Sur ce corpus, le moteur lexical gagne sur presque toutes les metriques
chiffrees : meilleur `mrr`, environ **13 fois plus rapide a chaud** dans ce
conteneur, pres de 5 fois plus leger, et aucun modele a telecharger. C'est un
resultat **mesure**, pas une opinion, et il va a l'encontre de l'intuition.

Quatre precisions indispensables pour lire ces temps :

- **`mean_retrieval_time_s` est une metrique trompeuse, et il faut cesser de
  la citer seule.** Ses 13,47 s melangent deux regimes qui n'ont rien a voir :
  la premiere question, qui charge le modele (**94,61 s**), et les suivantes
  (**0,19 s** chacune) — un facteur **500**. Une moyenne entre ces deux-la ne
  decrit aucune situation reelle. `src/evaluate.py` publie donc desormais
  `first_query_s` et `warm_retrieval_time_s` separement ;
- **ce chargement paresseux etait un vrai defaut, pas une curiosite de
  mesure.** Comme le modele ne se charge qu'au premier `encode()`, l'index
  semblait pret en une seconde et c'est la premiere question de l'utilisateur
  qui payait la minute et demie. `EmbeddingRetriever.warmup()` charge donc le
  modele a la **construction** du retriever, dans le `spinner` de `app.py` :
  meme cout total, mais place la ou l'utilisateur l'attend. Mesure apres
  correction, en natif : premiere question **0,108 s** au lieu de 94,61 s ;
- une fois ce biais ote, l'ecart honnete est **beaucoup plus faible** que ne
  le suggerait la moyenne contaminee. En natif, le regime a chaud mesure est
  de **0,024 s** en lexical contre **0,117 s** en semantique, soit **4,9x** —
  et non les ~30x que donnait `mean_total_time_s`. Le semantique reste plus
  lent, mais a l'echelle d'une requete web la difference est imperceptible ;
- `memory_mb_end` vient de `tracemalloc` (psutil n'est pas une dependance) et
  **ignore la memoire de PyTorch** : il annonce 165,8 Mo quand le pic de RSS
  reel, lu dans `/proc/self/status`, vaut **1 495 Mo**. La note d'honnetete du
  point d) n'est pas une precaution de style : le chiffre est faux d'un
  facteur 9 pour le moteur semantique.

Alors pourquoi garder le moteur semantique ? Pour deux choses que ce tableau
ne montre pas, et que l'etape 5 et le reglage du seuil mesurent : il **separe
nettement** les questions legitimes des questions hors sujet, et il comprend
les **sigles et reformulations**. Sur « Qu'est-ce que le TAL ? », TF-IDF donne
0,1347 pour un seuil a 0,13 — il accepte de **0,0047**, alors qu'une question
hors sujet comme « Comment soigner une angine chez l'enfant ? » obtient
**0,1390**, soit davantage. Les embeddings donnent 0,4954 contre 0,2508.

Et le pire n'est pas la marge : a 0,1347, TF-IDF ne ramene **pas du tout** le
bon article (son premier passage vient d'*Exploration de donnees*) tout en
restant **au-dessus de son propre seuil**. Il ne se contente donc pas de mal
classer : il repond **avec aplomb a partir d'un passage hors sujet**, sans que
le garde-fou se declenche. Le detail est a l'etape 5, « Et sur une question
vraiment acronymique ? ».

#### Calibrer le seuil du moteur semantique

La valeur `MIN_SCORE_EMBEDDINGS=0.30` du premier jet etait une **estimation**,
jamais mesuree. Mesure faite :

| Population | `tfidf` | `embeddings` |
| --- | --- | --- |
| Questions **dans** le corpus (5) | 0,151 a 0,205 | 0,821 a 0,897 |
| Questions **hors** corpus (3) | 0,081 a 0,121 | 0,230 a 0,365 |

A 0,30, le seuil tombait **dans** la population hors corpus : la question
« Combien coute un billet de train Paris-Lyon aujourd'hui ? » obtenait 0,3652
et recevait une reponse. Resultat mesure :
`refusal_rate_out_of_corpus = 0,667` au lieu de 1.0 — exactement le defaut que
le garde-fou doit empecher.

Trois questions hors corpus font une base trop mince pour placer un seuil. Une
**sonde complementaire de 20 questions** (10 dans, 10 hors ; gardee en dehors
de `eval/questions.json` pour ne pas fausser les metriques publiees) precise
les deux populations :

| Population (28 questions au total) | `tfidf` | `embeddings` |
| --- | --- | --- |
| Dans le corpus | 0,105 a 0,234 | 0,495 a 0,897 |
| Hors corpus | 0,076 a 0,139 | 0,186 a 0,365 |
| Separation | **recouvrement** | **ecart net** : 0,365 -> 0,495 |

D'ou le seuil retenu : **0,43**, milieu exact de l'ecart mesure
(0,43 - 0,365 = 0,495 - 0,43 = 0,065). Verification apres correction :
`refusal_rate_out_of_corpus = 1.0` **et** `false_refusals_in_corpus = 0`.

Et un resultat qu'il faut savoir regarder en face : sur le jeu elargi, les
deux populations de **TF-IDF se recouvrent**. La question legitime « Comment
un ordinateur apprend-il a partir d'exemples ? » obtient 0,1054, soit **moins**
que la question hors sujet sur l'angine (0,1390). Le seuil 0,13 reste correct
sur le jeu officiel de 8 questions, mais **aucun seuil ne separe parfaitement
les deux populations en lexical**. C'est l'argument chiffre le plus solide en
faveur des embeddings — plus solide que la comparaison de rangs de l'etape 5.

**Vos chiffres seront differents** : corpus different, machine differente.
Ce qui doit rester vrai : `refusal_rate_out_of_corpus = 1.0` et
`false_refusals_in_corpus = 0`.

#### Ce que cette etape a reellement permis de corriger

Au premier passage, l'evaluation a donne ceci :

```
  hit_at_k                       : 1.0
  refusal_rate_out_of_corpus     : 1.0
  false_refusals_in_corpus       : 5        <-- PROBLEME
  keyword_coverage_mean          : 0.0
```

La recherche trouvait **toujours** le bon article (`hit@4 = 1.0`), mais le
systeme refusait de repondre aux **5** questions legitimes. Les scores
mesures expliquaient tout :

| | TF-IDF |
| --- | --- |
| Questions **dans** le corpus | 0,151 a 0,205 |
| Questions **hors** corpus | 0,081 a 0,121 |
| Seuil `MIN_SCORE` initial | **0,25** |

Le seuil etait **au-dessus des deux populations** : il refusait tout. Sur ces
8 questions, les deux populations sont bien separees et il existe un seuil
correct, entre 0,121 et 0,151 — d'ou `0.13`.

> **Attention a ne pas sur-interpreter cette separation.** Elle vaut pour ces
> 8 questions. La sonde de 20 questions du paragraphe precedent montre que les
> deux populations de TF-IDF **se recouvrent** des que les questions sont plus
> difficiles (0,1054 pour une question legitime contre 0,1390 pour une
> question hors sujet). `MIN_SCORE_TFIDF=0.13` est donc le **meilleur
> compromis disponible**, pas un separateur parfait.

Cause de fond : **un seuil unique pour deux moteurs dont les scores ne sont
pas sur la meme echelle.** Les similarites cosinus de TF-IDF sur des chunks
courts sont basses ; celles des embeddings sont bien plus hautes. Les deux
moteurs ayant ete mesures depuis, l'ecart est chiffre : pour les **memes**
5 questions dans le corpus, TF-IDF donne 0,151 a 0,205 quand les embeddings
donnent **0,821 a 0,897**, soit environ **4 fois plus haut**.

La correction introduit **un seuil par moteur** (`src/config.py`) :

```python
min_score_tfidf: float = _env_float("MIN_SCORE_TFIDF", 0.13)
min_score_embeddings: float = _env_float("MIN_SCORE_EMBEDDINGS", 0.43)
min_score_override: str = _env("MIN_SCORE", "")   # vide = automatique
```

> **Attention, piege a repetition.** La valeur de 0,43 ci-dessus est elle
> aussi le fruit d'une mesure. Le premier jet du TP ecrivait `0.30`, choisi
> « a vue » parce que les embeddings scorent plus haut : cette estimation
> s'est revelee **fausse** a la mesure (une question hors corpus passait a
> 0,3652). Le detail de la calibration est a l'etape 8.

et le seuil est resolu selon le moteur **reellement** utilise, car
l'interface permet d'en changer a chaud :

```python
if min_score is None:
    min_score = CFG.min_score_for(self.retriever.kind)
```

Resultat apres correction : `false_refusals_in_corpus` passe de **5 a 0** et
`keyword_coverage_mean` de **0,0 a 0,834**, sans aucune degradation du taux de
refus hors corpus (toujours 1.0).

**C'est exactement a cela que sert une evaluation chiffree.** A l'oeil nu, le
systeme « marchait » : il repondait correctement aux questions de
demonstration posees avec un seuil abaisse a la main. Seule la mesure
systematique a revele que la configuration par defaut etait inutilisable.

### 8. Test de validation

```powershell
python -m pytest
python -m src.evaluate --retriever tfidf
```

Criteres de reussite :

- [ ] tous les tests passent ;
- [ ] `refusal_rate_out_of_corpus` vaut **1.0** ;
- [ ] `false_refusals_in_corpus` vaut **0** ;
- [ ] `hit_at_k` est superieur a 0,6 ;
- [ ] un rapport est ecrit dans `eval/`.

> **Adaptez `eval/questions.json` a VOTRE corpus.** Si vous avez collecte des
> articles sur le climat, les `expected_titles` fournis ne correspondent a
> rien et `hit_at_k` sera faux. Le fichier est commente pour cela.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `ModuleNotFoundError: src` dans pytest | racine absente du chemin | `pythonpath = ["."]` dans `pyproject.toml` (fourni) | `pytest` collecte |
| Tests lents / echec reseau | un test appelle une vraie API | simuler avec `monkeypatch` | tests < 10 s |
| `false_refusals_in_corpus > 0` | seuil trop severe | baisser `MIN_SCORE_TFIDF` / `MIN_SCORE_EMBEDDINGS` | retombe a 0 |
| `refusal_rate_out_of_corpus < 1.0` | seuil trop permissif | **monter** le seuil | remonte a 1.0 |
| `hit_at_k` a 0 | `expected_titles` ne colle pas au corpus | adapter `eval/questions.json` | hit remonte |
| `keyword_coverage` a 0 | tout est refuse (voir ci-dessus) | corriger le seuil d'abord | la couverture remonte |
| `fixture 'chunks' not found` | `conftest.py` absent ou mal place | il doit etre dans `tests/` | les fixtures sont vues |

### 10. Livrable de l'etape

- [ ] `python -m pytest` : 100 % de reussite
- [ ] Rapport d'evaluation genere dans `eval/`
- [ ] Tableau de vos metriques dans le compte rendu
- [ ] `MIN_SCORE_*` regle, et le reglage **justifie par vos mesures**
- [ ] Vous savez expliquer l'arbitrage faux refus / hallucinations

---

## Etape 9 — Conteneurisation avec Docker

### 1. Objectif pedagogique

Emballer l'application dans une image reproductible, en gardant les secrets et
les donnees **a l'exterieur** de l'image.

### 2. Explication simple

**Image ou conteneur ?** C'est la question d'entretien classique :

| | Image | Conteneur |
| --- | --- | --- |
| Nature | modele **fige**, lecture seule | **instance en execution** |
| Creation | `docker build` | `docker run` |
| Analogie | une classe | un objet |
| Nombre | 1 | autant qu'on veut, depuis la meme image |

Une image est un empilement de **couches**. Chaque instruction du
`Dockerfile` en cree une, et Docker les met en cache : c'est ce qui rend les
reconstructions rapides — a condition d'ordonner correctement les
instructions.

### 3. Prerequis

- Docker Desktop installe et **demarre**.
- Un fichier `.env` a la racine (meme avec des cles vides).

### 4. Commandes exactes

```powershell
# Image legere (TF-IDF seulement)
docker build -t wikinlp:lite .

# Image complete (recherche semantique, telecharge PyTorch)
docker build --build-arg REQUIREMENTS=requirements.txt -t wikinlp:full .

# Lancer avec Compose (recommande)
docker compose up -d --build
docker compose ps
docker compose logs -f app
docker compose down            # arrete, GARDE les donnees
docker compose down -v         # arrete et SUPPRIME les volumes

# Lancer sans Compose
docker run --rm -p 8501:8501 --env-file .env -v "${PWD}\data:/app/data" wikinlp:lite

# Executer le pipeline DANS le conteneur
docker compose exec app python -m src.collect_wikipedia --max 30
docker compose exec app python -m src.preprocess
docker compose exec app python -m src.rag --build

# Diagnostic
docker compose exec app bash
docker images
docker system df
```

### 5. Code complet

| Fichier | Role |
| --- | --- |
| `Dockerfile` | construction de l'image, 2 variantes par `ARG` |
| `.dockerignore` | **ce qui n'entre pas** dans l'image |
| `compose.yaml` | ports, volumes, secrets, service Ollama optionnel |

### 6. Explication du code

**a) L'ordre des instructions determine l'efficacite du cache.**

```dockerfile
# Dependances SEULES et AVANT le code source
COPY requirements.txt requirements-lite.txt ./
RUN python -m pip install -r ${REQUIREMENTS}

# Code source ENSUITE
COPY src/ ./src/
COPY app.py app_gradio.py pyproject.toml ./
```

Inverser ces deux blocs ferait **reinstaller PyTorch a chaque modification de
`app.py`** : plusieurs minutes perdues a chaque edition. En l'etat, modifier
le code ne reconstruit que les deux dernieres couches.

**b) Ce qui n'entre jamais dans l'image** (`.dockerignore`) :

```
.env                 # les secrets
.venv/               # un venv Linux ne sert a rien ici
hf_cache/            # des centaines de Mo de modeles
data/raw/*           # les donnees : montees en volume
.git/                # tout l'historique
```

> **Securite, point essentiel :** une cle API copiee dans une image reste
> lisible dans l'**historique de ses couches**, meme si une couche ulterieure
> supprime le fichier. `docker history` suffit a la retrouver. Les secrets
> s'injectent **au lancement** (`--env-file` / `env_file:`), jamais a la
> construction.

**c) Deux variantes depuis un seul Dockerfile :**

```dockerfile
ARG REQUIREMENTS=requirements-lite.txt
RUN python -m pip install -r ${REQUIREMENTS}
```

Taille **mesuree** de l'image legere : **865 Mo**. C'est plus que les ~450 Mo
qu'on pourrait estimer de tete : `python:3.12-slim` pese deja ~150 Mo, et
scikit-learn, SciPy, NumPy, pandas et Streamlit sont volumineux. **Mesurez,
ne devinez pas** :

```powershell
docker images wikinlp --format "{{.Repository}}:{{.Tag}} {{.Size}}"
```

**d) Les ressources NLTK sont telechargees a la construction**, pas au premier
lancement :

```dockerfile
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt punkt_tab stopwords || true
```

Un conteneur doit pouvoir demarrer sans reseau. Le `|| true` evite de casser
la construction si le telechargement echoue : le code a ses replis.

**e) Un utilisateur non privilegie :**

```dockerfile
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
```

Par defaut un conteneur tourne en `root`. Si le processus est compromis,
mieux vaut qu'il ne soit pas administrateur. C'est une regle de securite de
base, et elle ne coute rien.

**f) `--server.address=0.0.0.0` est obligatoire.**

```dockerfile
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Avec la valeur par defaut (`localhost`), le serveur n'ecoute que la boucle
interne **du conteneur** : le port publie reste muet et l'on croit a un
probleme de port. C'est l'une des erreurs Docker les plus frequentes.

**g) Le cache des modeles est place dans `/app` :**

```dockerfile
ENV HF_HOME=/app/hf_cache
```

`/app/hf_cache` est monte en volume. Dans `/home/appuser`, le modele de
470 Mo serait retelecharge a **chaque** recreation du conteneur.

**h) La persistance, dans `compose.yaml` :**

```yaml
volumes:
  - ./data:/app/data        # corpus et index : survivent a `down`
  - hf_cache:/app/hf_cache  # volume nomme : evite les soucis de droits Windows
```

Sans le premier, tout le travail de collecte serait perdu a chaque
`docker compose down`.

**i) Le port se remappe sans toucher a l'image :**

```yaml
ports:
  - "8501:8501"       # "hote:conteneur"
```

Si 8501 est pris, changez **seulement** le nombre de gauche : `"8502:8501"`.

**j) Le LLM local est un profil optionnel :**

```yaml
ollama:
  profiles: ["local-llm"]
```

Il ne demarre **que** sur `docker compose --profile local-llm up -d`. Dans un
reseau Compose, on s'adresse a un service par son **nom** :
`OLLAMA_HOST=http://ollama:11434`. Mettre `localhost` designerait le
conteneur appelant — autre erreur tres fréquente.

### 7. Resultat attendu

```
[+] Building ... FINISHED
 => naming to docker.io/library/wikinlp:lite
```

```powershell
docker compose ps
```

```
NAME          IMAGE                      STATUS                   PORTS
wikinlp-app   wikinlp-assistant:local    Up 2 minutes (healthy)   0.0.0.0:8501->8501/tcp
```

Le `(healthy)` vient du `HEALTHCHECK` : Docker interroge
`/_stcore/health` et sait donc si l'application repond vraiment, et pas
seulement si le processus existe.

### 8. Test de validation

```powershell
# 1. L'image existe et sa taille est connue
docker images wikinlp --format "{{.Repository}}:{{.Tag}} {{.Size}}"

# 2. Les imports fonctionnent dans l'image
docker run --rm -e RETRIEVER=tfidf -e LLM_PROVIDER=extractive wikinlp:lite `
  python -c "from src.rag import RagPipeline; from src.config import CFG; print('OK', CFG.retriever)"

# 3. Les tests passent dans le conteneur
docker run --rm -v "${PWD}:/work" -w /work -e RETRIEVER=tfidf wikinlp:lite python -m pytest -q

# 4. Aucun secret dans l'image
docker run --rm wikinlp:lite sh -c "ls -la /app | grep -c '\.env' || echo 'OK : aucun .env dans l image'"

# 5. La persistance fonctionne
docker compose up -d
docker compose exec app python -m src.rag --info
docker compose down
docker compose up -d
docker compose exec app python -m src.rag --info    # les memes chiffres
```

Le point 5 est le vrai test de la persistance : apres un `down`/`up`, le
corpus et l'index doivent **toujours** etre la.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `Cannot connect to the Docker daemon` | Docker Desktop non demarre | le lancer et attendre l'icone verte | `docker info` |
| `port is already allocated` | 8501 deja pris | `"8502:8501"` dans `compose.yaml` | `docker compose ps` |
| Page inaccessible, conteneur « Up » | `--server.address` absent | ajouter `--server.address=0.0.0.0` | la page s'ouvre |
| `env file .env not found` | `.env` absent | `Copy-Item .env.example .env` | `docker compose config` |
| Image de plusieurs Go | `data/` ou `hf_cache/` copies | verifier `.dockerignore` | `docker images` |
| Donnees perdues apres `down` | pas de volume, ou `down -v` | monter `./data:/app/data` | point 5 ci-dessus |
| Modele retelecharge a chaque fois | `HF_HOME` hors volume | `HF_HOME=/app/hf_cache` | 2e demarrage rapide |
| `Permission denied` sur `/app/data` | droits du volume monte | `chown -R appuser` (deja fait), ou volume nomme | le conteneur ecrit |
| `OOMKilled` (code 137) | limite memoire atteinte | monter `memory: 4g` pour l'image complete | `docker stats` |
| Reconstruction tres lente | `COPY` du code avant les dependances | respecter l'ordre du Dockerfile | 2e build en secondes |

### 10. Livrable de l'etape

- [ ] `docker build` reussit
- [ ] `docker compose up -d` affiche `(healthy)`
- [ ] L'application repond sur `http://localhost:8501`
- [ ] Aucun `.env` ni cache de modele dans l'image
- [ ] Les donnees survivent a `down` puis `up`
- [ ] Taille de l'image **mesuree** et notee
- [ ] Vous savez expliquer image contre conteneur

---

# Seance 5 — Publier et deployer

## Etape 10 — Git et GitHub

### 1. Objectif pedagogique

Versionner proprement, et **garantir** qu'aucun secret ni fichier volumineux
ne part sur GitHub.

### 2. Explication simple

Git suit l'historique de vos fichiers **sources**. Il ne doit pas suivre ce
qui est **genere** (donnees, index, caches) ni ce qui est **secret** (`.env`).

Le piege majeur : un secret committe une fois reste dans l'historique
**pour toujours**, meme supprime dans un commit ulterieur. La seule reaction
correcte apres une fuite est de **revoquer la cle**.

### 3. Prerequis

- Git installe, un compte GitHub.
- `.gitignore` en place (fourni).

### 4. Commandes exactes

**a) Initialiser et verifier AVANT de committer**

```powershell
cd D:\TP_NLP\WikiNLP-Assistant
git init
git branch -M main

git config user.name "Votre Nom"
git config user.email "vous@example.com"

# VERIFICATION CRITIQUE : .env doit etre ignore
git check-ignore -v .env

# Voir ce qui SERAIT committe
git add -A
git status --short
```

**Relisez cette liste avant de continuer.** Vous ne devez voir ni `.env`, ni
`data/raw/*.jsonl`, ni `.venv/`, ni `hf_cache/`.

```powershell
# Verification chiffree : taille des fichiers suivis
git ls-files | ForEach-Object {
    [PSCustomObject]@{ Ko = [math]::Round((Get-Item $_).Length / 1KB, 1); Fichier = $_ }
} | Sort-Object Ko -Descending | Select-Object -First 10
```

**b) Premier commit**

```powershell
git commit -m "WikiNLP Assistant : pipeline NLP/RAG complet avec tests et Docker"
git log --oneline
```

**c) Depot distant**

Creez un depot **vide** sur <https://github.com/new> (sans README ni
`.gitignore`, pour eviter un conflit immediat), puis :

```powershell
git remote add origin https://github.com/<votre-compte>/WikiNLP-Assistant.git
git push -u origin main
```

**d) Verification apres publication**

```powershell
# Ce qui est reellement publie
git ls-remote --heads origin
# Rechercher une eventuelle fuite dans TOUT l'historique
git log --all --full-history -- .env
git log -p --all -S "gsk_" | Select-Object -First 20
```

Les deux dernieres commandes doivent ne **rien** renvoyer.

### 5. Code complet

| Fichier | Role |
| --- | --- |
| `.gitignore` | exclut secrets, donnees, caches, modeles |
| `.github/workflows/tests.yml` | integration continue |

### 6. Explication du code

**a) Le workflow de CI est volontairement leger :**

```yaml
env:
  RETRIEVER: tfidf
  LLM_PROVIDER: extractive
```

```yaml
- name: Installer les dependances legeres
  run: python -m pip install -r requirements-lite.txt
```

Sans PyTorch, les tests tournent en moins d'une minute au lieu de cinq, et
**sans aucune cle API** : la CI est donc deterministe. Une CI lente ou
instable finit toujours par etre ignoree.

**b) Deux versions de Python sont testees :**

```yaml
matrix:
  python-version: ["3.11", "3.12"]
```

Un projet qui ne fonctionne que sur une version mineure precise est fragile.

**c) La CI echoue volontairement si un secret est detecte :**

```yaml
- name: Verifier qu'aucun secret n'est commite
  run: |
    if git ls-files --error-unmatch .env 2>/dev/null; then
      echo "::error::Le fichier .env est versionne ! Retirez-le :"
      echo "  git rm --cached .env"
      exit 1
    fi
```

Le workflow cherche aussi les poids de modeles versionnes et les motifs de
cles (`gsk_`, `sk-`, `AIza`). **Mieux vaut une CI rouge qu'une cle publiee.**

**d) La CI construit aussi l'image Docker.** Beaucoup de projets ont un
`Dockerfile` qui ne compile plus depuis des mois ; cette etape l'empeche.

### 7. Resultat attendu

```
Enumerating objects: 45, done.
...
To https://github.com/<compte>/WikiNLP-Assistant.git
 * [new branch]      main -> main
```

Dans l'onglet **Actions** de GitHub, le workflow `tests` doit passer au vert
sur les deux versions de Python, plus le job `docker`.

Taille du depot attendue : **moins de 2 Mo** (le code, plus
`demo_chunks.jsonl`).

### 8. Test de validation

```powershell
# 1. Aucun secret publie
git log -p --all -S "gsk_" | Measure-Object -Line
# 2. Taille du depot
git count-objects -vH
# 3. LE test decisif : cloner ailleurs et tout relancer
cd ..
git clone https://github.com/<compte>/WikiNLP-Assistant.git test-clone
cd test-clone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lite.txt
Copy-Item .env.example .env
$env:RETRIEVER="tfidf"
python -m pytest
streamlit run app.py
```

Le point 3 est la seule preuve qui vaille : si le clone fonctionne sur une
machine vierge, votre depot est complet. C'est aussi ce que fera un
recruteur.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `.env` committe | `.gitignore` arrive trop tard | `git rm --cached .env`, committer, **et REVOQUER la cle** | `git log --all -- .env` ne renvoie rien |
| `remote: Repository not found` | URL erronee / droits | `git remote -v`, corriger avec `git remote set-url` | `git ls-remote origin` |
| `failed to push some refs` | le distant a des commits | `git pull --rebase origin main` puis repousser | `git push` passe |
| Depot de plusieurs centaines de Mo | `data/` versionne | `git rm -r --cached data/` | `git count-objects -vH` |
| `src refspec main does not match any` | aucun commit, ou autre branche | `git branch -M main` puis committer | `git log --oneline` |
| CI rouge : `ModuleNotFoundError` | dependance absente de `requirements-lite.txt` | l'ajouter | la CI repasse au vert |
| CI rouge sur le test de secrets | un secret est suivi | le retirer et revoquer la cle | la CI repasse au vert |
| `support for password authentication was removed` | mot de passe au lieu d'un token | utiliser un *Personal Access Token*, ou GitHub CLI (`gh auth login`) | le push passe |

### 10. Livrable de l'etape

- [ ] Depot GitHub public contenant tout le code
- [ ] `git check-ignore -v .env` confirme l'exclusion
- [ ] Aucune cle dans l'historique
- [ ] Depot de moins de 2 Mo
- [ ] CI verte sur les deux versions de Python
- [ ] Le clone de test fonctionne

---

## Etape 11 — Deployer gratuitement

### 1. Objectif pedagogique

Mettre l'application en ligne a cout nul, en comprenant les **contraintes
reelles** des offres gratuites — et en gardant une solution de secours.

### 2. Explication simple

Notre application a deux besoins difficiles a satisfaire gratuitement : de la
**RAM** (PyTorch plus un modele de 470 Mo demandent ~1 Go) et un **processus
qui tourne en continu**.

D'ou la strategie :

> **heberger une application LEGERE qui appelle une API LLM gratuite.**
> Le serveur ne fait que la recherche lexicale (quelques dizaines de Mo) ; la
> generation, couteuse, est deleguee a une API externe.

En pratique : `RETRIEVER=tfidf` et `LLM_PROVIDER=groq` (ou `extractive` pour
une demonstration sans cle).

### 3. Prerequis

- Etape 10 terminee (code sur GitHub).
- `data/processed/demo_chunks.jsonl` versionne.

### 4. Commandes exactes

Voir **`docs/DEPLOYMENT.md`**, qui contient la procedure complete. Resume :

```powershell
# 1. Creer le mini-corpus versionne (une seule fois)
python -m src.rag --build
python -c "import pathlib; src=pathlib.Path('data/processed/chunks.jsonl'); dst=pathlib.Path('data/processed/demo_chunks.jsonl'); lines=[l for l in src.read_text(encoding='utf-8').splitlines() if l.strip()]; dst.write_text('\n'.join(lines[:400])+'\n', encoding='utf-8'); print(len(lines[:400]), 'chunks ->', dst)"

# 2. Verifier son poids (doit rester sous 2 Mo)
(Get-Item data\processed\demo_chunks.jsonl).Length / 1KB

# 3. Le versionner (il est explicitement autorise par .gitignore)
git add data/processed/demo_chunks.jsonl
git commit -m "Ajout du mini-corpus de demonstration pour le deploiement"
git push
```

### 5. Code complet

| Fichier | Role |
| --- | --- |
| `app_gradio.py` | interface pour Hugging Face Spaces |
| `requirements-gradio.txt` | dependances sans PyTorch |
| `docs/DEPLOYMENT.md` | procedure detaillee et solutions de secours |

### 6. Explication du code

#### Pourquoi une seconde interface ? Une contrainte verifiee, pas supposee

Documentation officielle de Hugging Face (`docs/hub`, « Spaces Overview »,
consultee en **octobre 2026**) :

> « Static Spaces are free for everyone. Gradio and Docker Spaces run on
> compute and require a paid plan to create: PRO for personal accounts, Team
> or Enterprise for organizations. Free personal accounts in good standing
> can still host up to 2 Gradio Spaces running on ZeroGPU. »

Trois consequences pour un TP a 0 euro :

| Constat | Consequence |
| --- | --- |
| Les SDK proposes sont **Gradio, Docker et HTML statique** | **Streamlit n'est plus un SDK de Spaces** : `app.py` n'y est pas deployable |
| Un Space **Docker** exige un compte payant | la voie Docker est fermee |
| Un compte personnel gratuit peut avoir **2 Spaces Gradio sur ZeroGPU** | **c'est la seule voie gratuite** : d'ou `app_gradio.py` |

Le disque d'un Space n'etant **pas persistant**, le corpus doit etre
**versionne** dans le depot (d'ou `demo_chunks.jsonl`) et non genere au
demarrage.

> **Ces regles peuvent changer.** Verifiez la page officielle avant de
> deployer. C'est precisement pourquoi l'architecture du projet ne depend
> d'aucune plateforme : la logique est dans `src/`, les interfaces ne font que
> de l'affichage.

#### Le mode leger est force a l'execution

```python
FORCE_TFIDF = os.getenv("FORCE_TFIDF", "1") == "1"
RETRIEVER_KIND = "tfidf" if FORCE_TFIDF else CFG.retriever
```

Sur l'hebergement gratuit, la RAM disponible (de l'ordre du Go) ne suffit pas
a charger PyTorch. On force donc la recherche lexicale : la qualite baisse un
peu, mais l'application **demarre** et tient dans l'enveloppe memoire.

#### L'application doit demarrer meme en panne

```python
try:
    _chunks = load_chunks()
    ...
except Exception as exc:
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"
```

Si le corpus manque, l'interface se lance quand meme et **affiche la cause**.
Une application qui refuse de demarrer ne montre que « Build error » dans les
journaux : impossible a diagnostiquer pour un debutant.

#### Variable ou Secret ?

| | Variable | Secret |
| --- | --- | --- |
| Visibilite | **publique** | privee, illisible apres saisie |
| Duplication du Space | **recopiee** | non recopiee |
| Usage | `RETRIEVER`, `MIN_SCORE_TFIDF` | `GROQ_API_KEY` |

Une cle API va **toujours** dans un Secret. Hugging Face scanne les depots et
vous alerte si une cle est ecrite en dur.

#### Streamlit Community Cloud : l'alternative

Etat constate (octobre 2026) : gratuit, **1 application privee** et des
applications publiques illimitees, memoire limitee (les chiffres officiels,
dates de fevrier 2024, vont d'environ **690 Mo a 2,7 Go**, « may change at any
time without notice »), **mise en veille** apres inactivite.

Consequence : avec moins de 1 Go garanti, **ne tentez pas** la recherche
semantique. Forcez `RETRIEVER=tfidf` dans les secrets.

Les secrets y sont au format **TOML**, pas `.env` :

```toml
RETRIEVER = "tfidf"
LLM_PROVIDER = "groq"
GROQ_API_KEY = "gsk_votre_cle"
```

Aucune modification du code n'est necessaire : `src/config.py` lit
`os.getenv`, et Streamlit Cloud expose ces secrets comme variables
d'environnement.

### 7. Resultat attendu

Une URL publique du type `https://<user>-<nom>.hf.space` ou
`https://<nom>.streamlit.app`, affichant l'interface, repondant aux questions
du corpus et **refusant** les questions hors corpus.

> **Aucune promesse de disponibilite.** Une application gratuite peut etre
> mise en veille, bridee ou arretee sans preavis. Pour une soutenance,
> **prevoyez toujours** la demonstration locale en Docker comme solution de
> repli, et une capture d'ecran.

### 8. Test de validation — apres deploiement

| Verification | Resultat attendu |
| --- | --- |
| L'application s'affiche | interface complete, sans erreur |
| Nombre de passages | `Corpus : N passages` avec N > 0 |
| Question dans le corpus | reponse + sources cliquables |
| Question hors corpus | refus explicite, **aucune** source |
| Journaux | aucun `ModuleNotFoundError`, aucune alerte de secret |
| Cles | visibles nulle part dans l'interface ni les logs |
| Telephone | mise en page lisible |

**Le test qui previent 90 % des mauvaises surprises** : simulez ce que voit
l'hebergeur, c'est-a-dire **uniquement** ce qui est dans le depot.

```powershell
cd ..
git clone .\WikiNLP-Assistant test-clone
cd test-clone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lite.txt
$env:RETRIEVER="tfidf"
streamlit run app.py
```

Un deploiement echoue presque toujours parce qu'un fichier existe sur votre
machine mais **pas** dans le depot.

### 9. Erreurs frequentes et corrections

| Erreur | Cause | Correction | Verification |
| --- | --- | --- | --- |
| `ModuleNotFoundError: src` | `src/` non copie dans le Space | verifier l'arborescence distante | le build repart |
| `Aucun corpus trouve` | `demo_chunks.jsonl` ignore par git | `git add -f data/processed/demo_chunks.jsonl` | N > 0 |
| L'app redemarre en boucle | memoire depassee (PyTorch charge) | `RETRIEVER=tfidf` dans les secrets | l'app reste en ligne |
| Build du Space en echec | `requirements.txt` contient `torch` | utiliser `requirements-gradio.txt` | le build passe |
| `401` / `403` sur l'API | cle mise en **Variable** au lieu de **Secret**, ou mal copiee | recreer le Secret, redemarrer | la reponse revient |
| `429` | quota depasse | attendre, ou `LLM_PROVIDER=extractive` | la reponse revient |
| Space endormi | inactivite | ouvrir la page pour le reveiller : c'est le comportement **normal** du gratuit | l'app repart |
| Marche en local, pas en ligne | chemin absolu, ou fichier ignore | faire le test du clone propre | le clone fonctionne |
| Spaces refuse de creer le Space | Gradio/Docker demandent un compte payant | utiliser la voie ZeroGPU (2 Spaces Gradio), ou Streamlit Cloud | le Space se cree |

### 10. Livrable de l'etape

- [ ] URL publique fonctionnelle (ou demonstration Docker documentee)
- [ ] Les 7 verifications du tableau 8 passent
- [ ] Les cles sont dans des **Secrets**, jamais dans le code
- [ ] `docs/DEPLOYMENT.md` complete avec votre URL
- [ ] Solution de secours testee
- [ ] Capture d'ecran de l'application en ligne

---

# Annexe A — Tableau des erreurs courantes

## Etape 12 — Reference de depannage

Tableau recapitulatif, dans l'ordre ou les problemes surviennent. Les colonnes
sont : **erreur observee**, **cause probable**, **commande ou modification**,
**methode de verification**.

### A.1 Python et pip

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `python` ouvre le Microsoft Store | Python non installe (raccourci WindowsApps) | `winget install -e --id Python.Python.3.12` puis rouvrir PowerShell | `(Get-Command python).Source` ne contient plus `WindowsApps` |
| `Python est introuvable` | idem | idem | `python --version` |
| `Python est introuvable` **apres** installation | terminal deja ouvert avant l'installation : `PATH` perime | ouvrir un **nouveau** terminal | `(Get-Command python).Source` pointe vers `Programs\Python\Python312` |
| `'pip' is not recognized` | pip appele directement | toujours `python -m pip` | `python -m pip --version` |
| `error: Microsoft Visual C++ 14.0 required` | pas de wheel pour votre version de Python | utiliser Python 3.12, ou `requirements-lite.txt` | `pip install` sans compilation |
| `No matching distribution found` | version de Python trop recente | installer Python 3.12 | `pip install` reussit |
| `No space left on device` | cache pip/HF sur `C:` sature | `$env:PIP_CACHE_DIR="D:\pipcache"` ; `HF_HOME=./hf_cache` | `Get-PSDrive C` |

### A.2 Environnement virtuel (PowerShell)

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `Activate.ps1 cannot be loaded ... disabled` | politique d'execution | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` | l'invite affiche `(.venv)` |
| Les paquets s'installent hors du venv | venv non active | `.\.venv\Scripts\Activate.ps1` | `python -c "import sys; print(sys.prefix)"` |
| `(.venv)` disparait | nouveau terminal | reactiver le venv | l'invite affiche `(.venv)` |
| `ModuleNotFoundError` pour un paquet installe | deux Python differents | `Get-Command python -All` | un seul chemin, dans `.venv` |

### A.3 Dependances

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `ModuleNotFoundError: wikipediaapi` | paquet absent | `python -m pip install wikipedia-api` (installe avec tiret, importe **sans**) | `python -c "import wikipediaapi"` |
| `ModuleNotFoundError: sentence_transformers` | version legere installee | `pip install sentence-transformers` **ou** `RETRIEVER=tfidf` | `python -m src.embeddings` |
| `ModuleNotFoundError: No module named 'src'` | lance hors de la racine, ou sans `-m` | se placer a la racine, utiliser `python -m src.xxx` | `Get-Location` |
| `LookupError: Resource punkt not found` | ressource NLTK absente | `python -m nltk.downloader punkt punkt_tab stopwords` | plus d'avertissement |
| `chromadb` ne s'installe pas | compilation native sous Windows | rester sur `VECTOR_BACKEND=numpy` | `python -m src.vector_store` |
| Conflit de versions | installations successives melangees | recreer le venv : `Remove-Item -Recurse .venv` puis refaire l'etape 1 | `pip check` |

### A.4 API gratuite indisponible, quota, cles

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `Quota depasse (HTTP 429)` | limite par minute ou par jour atteinte | attendre ; ou `LLM_PROVIDER=ollama` ; ou `extractive` | `python -m src.llm` |
| `Cle API refusee (HTTP 401/403)` | cle vide, mal copiee, revoquee | recoller la cle dans `.env`, **sans** guillemets ni espace | `python -m src.llm` |
| `Modele introuvable (HTTP 404)` | nom de modele retire par le fournisseur | relever un nom valide dans sa documentation, corriger `GROQ_MODEL` | la reponse revient |
| `Panne cote fournisseur (HTTP 5xx)` | incident du service | reessayer plus tard, ou `extractive` | la reponse revient |
| `Gemini n'a renvoye aucun texte` | filtre de securite, ou budget de tokens epuise | augmenter `LLM_MAX_TOKENS`, reformuler | la reponse revient |
| `Connexion impossible` vers Ollama | serveur local eteint | `ollama serve` puis `ollama pull qwen2.5:1.5b-instruct` | `ollama list` |
| `Delai depasse` | reseau lent, ou LLM local sur CPU | augmenter `LLM_TIMEOUT`, baisser `LLM_MAX_TOKENS` | la reponse arrive |
| L'offre gratuite a disparu | conditions modifiees par le fournisseur | `LLM_PROVIDER=ollama` (local) ou `extractive` | l'app fonctionne |

### A.5 Memoire et performances

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `MemoryError` a l'encodage | lot trop gros | baisser `BATCH_SIZE` (16 → 8) dans `src/embeddings.py` | l'encodage finit |
| Machine qui rame a l'indexation | PyTorch plus modele plus IDE | fermer les autres applications, ou `RETRIEVER=tfidf` | `docker stats` / Gestionnaire des taches |
| `OOMKilled` (code 137) en Docker | limite memoire du conteneur | `memory: 4g` dans `compose.yaml` | `docker compose ps` |
| Index reconstruit a chaque lancement | corpus modifie, ou cache absent | normal apres une modification ; sinon verifier `data/index/emb_*.npy` | 2e lancement rapide |
| Premiere reponse tres lente | chargement du modele | desormais paye pendant l'ecran de chargement (`warmup()`), plus a la 1re question | `first_query_s` ~= `warm_retrieval_time_s` dans `src.evaluate` |
| `mean_retrieval_time_s` enorme (plusieurs secondes) | la moyenne inclut la requete a froid | lire `warm_retrieval_time_s`, publie separement | 0,024 s en lexical, 0,117 s en semantique |

### A.6 Docker

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `Cannot connect to the Docker daemon` | Docker Desktop non demarre | le lancer, attendre l'icone verte | `docker info` |
| `port is already allocated` | port 8501 occupe | `"8502:8501"` dans `compose.yaml` | `docker compose ps` |
| `failed to solve: ... requirements.txt not found` | mauvais contexte de build | lancer `docker build` **depuis la racine** du projet | le build repart |
| `env file .env not found` | `.env` absent | `Copy-Item .env.example .env` | `docker compose config` |
| Conteneur « Up » mais page inaccessible | `--server.address=0.0.0.0` manquant | le remettre dans `CMD` | la page s'ouvre |
| Image de plusieurs Go | `data/` ou `hf_cache/` embarques | verifier `.dockerignore` | `docker images` |
| `Permission denied` sur `/app/data` | droits du volume | utiliser un volume nomme, ou `chown` | le conteneur ecrit |
| Build tres lent a chaque fois | `COPY` du code avant les dependances | respecter l'ordre du `Dockerfile` | 2e build en secondes |
| Donnees perdues apres `down` | volume absent, ou `down -v` | monter `./data:/app/data` | `rag --info` apres redemarrage |

### A.7 Index vectoriel et RAG

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `[X] Aucun chunk disponible` | index non construit | `python -m src.rag --build` | `python -m src.rag --info` |
| `Index illisible / incoherent` | fichier corrompu (arret brutal) | `python -m src.rag --rebuild` | l'index se recharge |
| `Cache incoherent (taille differente)` | corpus modifie | normal : le recalcul se declenche seul | les embeddings se recalculent |
| Repond a **tout**, meme hors sujet | seuil trop bas | monter `MIN_SCORE_TFIDF` / `MIN_SCORE_EMBEDDINGS` | la tarte tatin est refusee |
| Refuse **toutes** les questions | seuil trop haut (**cas vecu, cf. etape 8**) | baisser le seuil du moteur concerne | `false_refusals_in_corpus` a 0 |
| Resultats classes a l'envers (Chroma) | distance non convertie en similarite | verifier `1.0 - distance` | scores dans [0,1] |
| Passages tronques en milieu de mot | chevauchement par caracteres | chevauchement **par phrases** (corrige) | les chunks commencent par un mot entier |
| Bibliographies et ISBN dans les resultats | sections finales non coupees | verifier `_heading_of()` et `TAIL_SECTIONS` | plus d'ISBN dans les extraits |

### A.8 Fonctionne en local, echoue en ligne

| Erreur observee | Cause probable | Correction | Verification |
| --- | --- | --- | --- |
| `ModuleNotFoundError` en ligne seulement | fichier present localement mais non versionne | `git status`, ajouter le fichier | le clone de test fonctionne |
| `Aucun corpus trouve` en ligne | `data/` ignore par git | versionner `demo_chunks.jsonl` (`git add -f`) | N > 0 en ligne |
| Redemarrages en boucle | memoire depassee | `RETRIEVER=tfidf` | l'app reste en ligne |
| `401` en ligne, OK en local | `.env` n'existe pas chez l'hebergeur | definir un **Secret** sur la plateforme | la reponse revient |
| Chemins introuvables | chemins absolus `D:\...` | utiliser les chemins de `src/config.py` | fonctionne sur Linux |
| L'app s'endort | mise en veille du gratuit | comportement normal ; ouvrir la page | l'app repart |

---

# Annexe B — Checklist de validation finale

## B.1 Environnement

- [ ] `python --version` affiche 3.10, 3.11 ou 3.12
- [ ] L'invite affiche `(.venv)`
- [ ] `python -m src.config` s'execute sans erreur
- [ ] `.env` existe et `WIKI_USER_AGENT` est personnalise
- [ ] `git check-ignore -v .env` confirme l'exclusion

## B.2 Donnees

- [ ] 20 a 50 articles dans `data/raw/articles.jsonl`
- [ ] Articles et titres uniques en nombre egal
- [ ] Les titres collectes sont coherents avec le theme
- [ ] Une relance de la collecte n'ajoute rien
- [ ] `articles_clean.jsonl` contient `text_clean` **et** `tokens_norm`
- [ ] Reduction du nettoyage entre 5 et 20 %
- [ ] Plus aucun ISBN/DOI dans les extraits

## B.3 Modeles et RAG

- [ ] Experience A executee, frequences relevees
- [ ] Experience B executee, similarites relevees
- [ ] Au moins un fournisseur LLM teste
- [ ] `chunks.jsonl` existe, taille max <= `CHUNK_SIZE + CHUNK_OVERLAP`
- [ ] `--compare` execute, les trois approches observees
- [ ] Une question hors corpus est refusee **sans** source

## B.4 Application

- [ ] L'application repond avec ses sources
- [ ] Les 10 tests manuels de l'etape 6 passent
- [ ] Le message « aucun passage pertinent » s'affiche
- [ ] L'historique fonctionne
- [ ] Le mode demonstration marche sans cle API
- [ ] Mise en page lisible en largeur telephone

## B.5 Qualite

- [ ] `python -m pytest` : 100 % de reussite
- [ ] `refusal_rate_out_of_corpus` = 1.0
- [ ] `false_refusals_in_corpus` = 0
- [ ] `hit_at_k` > 0,6
- [ ] Rapport d'evaluation dans `eval/`
- [ ] Metriques reportees dans le compte rendu

## B.6 Docker

- [ ] `docker build` reussit
- [ ] `docker compose up -d` affiche `(healthy)`
- [ ] Aucun `.env` ni cache de modele dans l'image
- [ ] Les donnees survivent a `down` puis `up`
- [ ] Taille de l'image mesuree

## B.7 Git, GitHub, deploiement

- [ ] Depot GitHub contenant tout le code
- [ ] Aucune cle dans l'historique
- [ ] Depot < 2 Mo
- [ ] CI verte sur 3.11 et 3.12
- [ ] Le clone de test fonctionne
- [ ] URL publique **ou** demonstration Docker documentee
- [ ] Cles dans des Secrets

## B.8 Documentation

- [ ] `README.md` a jour
- [ ] `docs/DEPLOYMENT.md` complete avec votre URL
- [ ] Schema d'architecture present
- [ ] Limites connues documentees
- [ ] Attribution Wikipedia (CC BY-SA 4.0) visible dans l'interface

---

# Annexe C — Environnement des mesures de reference

Les chiffres presentes comme « mesure de reference » dans ce document ont ete
obtenus en executant reellement le TP. **Le TP a ete execute DEUX fois**, dans
deux environnements differents, ce qui permet de distinguer ce qui est stable
de ce qui depend de la plateforme.

| Element | Environnement A (reference) | Environnement B |
| --- | --- | --- |
| Execution | **Windows natif** (venv) | conteneur Docker `python:3.12-slim` |
| Python | **3.12.10** | 3.12.15 |
| Plateforme | `Windows-11-10.0.26200-SP0` | `Linux-...-WSL2-x86_64-glibc2.41` |
| Installation | `winget install Python.Python.3.12` | image Docker |
| Dependances | `requirements-lite.txt` | `requirements-lite.txt` |
| Moteur | `tfidf` | `tfidf` |
| LLM | `extractive` (aucun appel reseau) | `extractive` |

Hote commun : Windows 11 Pro N 26200, AMD64, 15,8 Go de RAM, **sans GPU**.
Corpus commun : 30 articles de Wikipedia **francais**, themes NLP/IA,
1 137 769 caracteres bruts. Mesures du 16 octobre 2026.

### Resultats compares

| Metrique | Windows natif | Docker Linux | Stable ? |
| --- | --- | --- | --- |
| Caracteres apres nettoyage | 993 396 | 993 576 | a 0,02 % pres |
| Chunks produits | **1 345** | **1 345** | identique |
| Taille des chunks (min/moy/max) | 145 / 774 / 900 | 145 / 774 / 900 | identique |
| `hit_at_k` (k=4) | **1.0** | **1.0** | identique |
| `mrr` | **0,875** | **0,875** | identique |
| `keyword_coverage_mean` | **0,834** | **0,834** | identique |
| `refusal_rate_out_of_corpus` | **1.0** | **1.0** | identique |
| `false_refusals_in_corpus` | **0** | **0** | identique |
| `memory_mb_end` | 103,2 | 101,0 | identique |
| `index_build_s` | 26,76 | 12,13 | **2,2x plus lent** |
| `warm_retrieval_time_s` (regime normal) | 0,024 | 0,013 | **1,9x plus lent** |
| `first_query_s` (a froid) | 2,99 | 1,68 | 1,8x plus lent |
| `pytest` (117 tests, a chaud) | 6,2 s | 4,9 s | **1,3x plus lent** |

> `mean_retrieval_time_s` ne figure volontairement **pas** dans ce tableau :
> cette moyenne melange la premiere requete (a froid) et les suivantes, et
> vaut ici 0,616 s en natif pour un regime normal a 0,024 s — elle
> sous-estime la vitesse reelle d'un facteur 25. C'est
> `warm_retrieval_time_s` qu'il faut lire, et c'est pour cela que les deux
> sont desormais publies separement par `src/evaluate.py`.

**Trois enseignements, et ils sont utiles :**

1. **La qualite ne depend pas de la plateforme.** Toutes les metriques de
   pertinence sont rigoureusement identiques. C'est normal : ce sont des
   calculs deterministes sur les memes donnees.
2. **Les temps, si.** Sur Windows natif, tout est 1,3 a 2,2 fois plus lent
   qu'en conteneur Linux. La cause principale est le systeme de fichiers de
   Windows, tres penalise par les milliers de petits acces que font Python a
   l'import et pytest a la collecte. **Ne comparez donc jamais vos temps a
   ceux d'une autre machine** ; comparez-les entre eux, avant et apres une
   modification.
3. **Ne mesurez jamais un temps sur la premiere execution.** Cette ligne
   `pytest` a d'abord ete relevee a **19,3 s**, ce qui donnait un ecart
   fantaisiste de « 5x ». C'etait la toute premiere execution apres
   l'installation : Python compilait ses `.pyc`, et l'antivirus de Windows
   inspectait 55 paquets fraichement ecrits. A chaud, c'est **6,2 s**, et
   l'ecart reel tombe a 1,3x. Le conteneur subit le meme effet (11,4 s au
   premier lancement, 4,9 s ensuite).
   **Regle : jetez la premiere mesure, gardez la mediane de trois autres.**
   Sans cela, on documente le temps de son antivirus.

L'ecart de 180 caracteres sur le nettoyage (0,02 %) vient tres probablement de
la version de la base Unicode embarquee dans chaque build de Python, utilisee
par `unicodedata.normalize("NFKC", ...)`. Il n'a aucun effet : le nombre de
chunks et toutes les metriques sont identiques.

### Environnement C — le moteur semantique

Les deux environnements ci-dessus mesurent le chemin **lexical**. Le chemin
**semantique** a ete mesure dans **deux** environnements a son tour : en
conteneur, puis en **natif sur Windows** dans le venv du TP.

| Element | C1 — conteneur | C2 — Windows natif |
| --- | --- | --- |
| Execution | image **`wikinlp:full`** | venv `.venv` |
| Dependances | `requirements.txt` | `requirements.txt` |
| Poids installe | image **1,89 Go** | venv **1,28 Go** (dont torch **502 Mo**) |
| Variante torch | `2.14.1+cpu` (roue 196,3 Mo) | `2.14.1+cpu` (roue **124,1 Mo**) |
| Moteur | `embeddings`, backend `numpy` | idem |
| Modele | `paraphrase-multilingual-MiniLM-L12-v2`, **384 dimensions** | idem |
| Cache du modele | volume temporaire | `./hf_cache` sur D:, **458 Mo** |
| Corpus | le meme : 30 articles, **1 345 chunks** (non recollecte) | idem |

**Les deux installations donnent la variante CPU**, mais pas par le meme
chemin : sous Windows `pip install torch` suffit, alors que sous Linux il
faut forcer l'index CPU (voir juste apres). La roue Windows est aussi plus
petite (124,1 Mo contre 196,3 Mo).

**Le cache d'embeddings est portable d'une plateforme a l'autre.** Le fichier
`data/index/emb_cfb19e7ddda19cc0.npy` (1,97 Mo), calcule **dans le conteneur
Linux**, a ete relu tel quel par l'execution **native Windows** :

```
  Embeddings lus depuis le cache : emb_cfb19e7ddda19cc0.npy (1345, 384)
```

La construction est ainsi passee de 3 min 02 s a **16,3 s**. C'est la
validation concrete de la cle de cache (modele + corpus) decrite a
l'etape 4 : elle ne depend ni de la machine ni du systeme.

**Un piege a l'installation, mesure et corrige.** Sur Linux — donc dans
l'image — `pip install torch` installe par defaut la variante **CUDA** :
environ 2,4 Go de bibliotheques NVIDIA parfaitement inutiles sans GPU. C'est
tres probablement la cause de l'echec de la premiere tentative. Le
`Dockerfile` installe donc explicitement la roue **CPU** avant
`requirements.txt` :

```dockerfile
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Mesure : `torch-2.14.1+cpu`, **196,3 Mo** telecharges en ~30 s. Sous Windows,
`pip install torch` donne deja la variante CPU : l'etape ne concerne que
l'image.

**Le modele capte bien le sens** (`python -m src.embeddings`) :

| Paire de phrases | Cosinus mesure |
| --- | --- |
| « Le chat dort sur le canape. » / « Un felin se repose sur le sofa. » | **0,761** |
| « Le chat dort sur le canape. » / « La voiture roule tres vite sur l'autoroute. » | **-0,035** |

Les deux premieres phrases ne partagent **aucun mot** et obtiennent pourtant
0,761 : c'est exactement ce que TF-IDF ne sait pas faire.

**Construction de l'index vectoriel** (`python -m src.rag --build`) :

| Element | Mesure |
| --- | --- |
| Encodage de 1 345 chunks sur CPU | **3 min 02 s** (85 lots de 16) |
| `data/index/` apres construction | **5,2 Mo** |
| dont `wikinlp_vectors.npy` | 2,07 Mo (1 345 x 384 x 4 octets) |
| dont `emb_<cle>.npy` (cache d'encodage) | 2,07 Mo |
| dont `wikinlp_records.jsonl` | 1,24 Mo |
| Rechargement de l'index depuis le disque | **0,17 s** |

Le cache d'encodage double le volume des vecteurs : c'est voulu, il evite de
re-encoder 3 minutes de corpus a chaque reconstruction de l'index.

Les metriques de pertinence des deux moteurs, mesurees dans la **meme**
execution de `src.evaluate --compare`, sont rassemblees a l'etape 8
(« Les deux moteurs, mesures dans la meme execution »), avec la calibration
de `MIN_SCORE_EMBEDDINGS` qui en decoule. Resume : `hit_at_k` 1.0 pour les
deux, `mrr` 0,875 en lexical contre 0,583 en semantique, refus hors corpus
1.0 et faux refus 0 **pour les deux** apres calibration, et un pic de memoire
de 312 Mo contre 1 495 Mo.

#### C2 — le moteur semantique en natif, et un defaut qu'il a revele

Le moteur semantique a ensuite ete mesure **en natif sous Windows**. La
qualite est identique au conteneur — mais la mesure a mis au jour un vrai
defaut d'experience utilisateur.

**Le symptome.** En detaillant les temps question par question :

| Question | Recherche (embeddings) |
| --- | --- |
| q1 (la premiere) | **94,61 s** |
| q2 | 0,192 s |
| q3 | 0,190 s |
| q4 | 0,181 s |
| q5 | 0,193 s |

Un facteur **500** entre la premiere question et les suivantes. Le meme
effet existe en lexical, en plus discret : 3,33 s puis 0,022 s (facteur 152).

**La cause.** `SentenceEmbedder` charge son modele **paresseusement**, au
premier `encode()`. Quand l'index vectoriel est relu depuis le disque, on ne
touche donc au modele qu'a la premiere question — et le charger coute 94,6 s,
meme avec un cache chaud. Dans l'interface web, l'utilisateur voyait l'index
se charger en quelques secondes, puis **sa premiere question restait bloquee
une minute et demie**, sans aucune explication.

**Deux consequences, et deux corrections.**

1. **La metrique etait trompeuse.** `mean_retrieval_time_s` moyennait la
   requete a froid avec les requetes a chaud : 0,616 s annonces pour un
   regime reel a 0,024 s, soit une sous-estimation de la vitesse d'un facteur
   25. `src/evaluate.py` publie desormais `first_query_s` et
   `warm_retrieval_time_s` separement.

2. **Le cout a ete deplace, pas supprime.** `EmbeddingRetriever.warmup()`
   charge le modele pendant la construction du retriever — donc, dans
   l'application Streamlit, **a l'interieur de l'ecran de chargement** qui
   existait deja. Apres correction :

   | | Avant | Apres |
   | --- | --- | --- |
   | `index_build_s` | 0,20 | **130,09** |
   | `first_query_s` | **94,61** | **0,108** |
   | `warm_retrieval_time_s` | 0,117 | 0,117 |

   Le cout total est du meme ordre, mais il change de place : l'attente se
   produit desormais pendant le chargement annonce, avec un indicateur de
   progression, au lieu de frapper la premiere question.

   (Les 130 s depassent les 94,6 s mesures isolement : le chargement du
   modele varie d'une execution a l'autre selon le cache de fichiers du
   systeme. C'est une mesure unique, pas une moyenne — relevez la votre.)

**Les deux moteurs en natif, apres correction :**

| Metrique | `tfidf` | `embeddings` | Rapport |
| --- | --- | --- | --- |
| `min_score` (calibre) | 0,13 | 0,43 | — |
| `index_build_s` | 26,76 | 130,09 | 4,9x |
| `first_query_s` | 2,99 | 0,108 | — |
| **`warm_retrieval_time_s`** | **0,024** | **0,117** | **4,9x** |
| `hit_at_k` | **1.0** | **1.0** | identique |
| `mrr` | **0,875** | 0,583 | lexical meilleur |
| `keyword_coverage_mean` | **0,834** | 0,768 | lexical meilleur |
| `refusal_rate_out_of_corpus` | **1.0** | **1.0** | identique |
| `false_refusals_in_corpus` | **0** | **0** | identique |
| `memory_mb_end` | 103,2 | 682,5 | 6,6x |

**A retenir sur le cout reel du semantique.** En regime normal, une requete
semantique prend 0,117 s contre 0,024 s en lexical : **4,9 fois plus**, et
non les 30 fois que suggerait la moyenne contaminee. Les deux sont largement
assez rapides pour un chatbot. Le vrai prix du semantique n'est pas la
requete, c'est le **demarrage** (130 s contre 27 s) et la **memoire**
(683 Mo contre 103 Mo en allocations Python, 1 495 Mo de pic reel).

**Ce qui a ete reellement execute et verifie :**

| Element | Etat |
| --- | --- |
| Installation de Python 3.12 par `winget` | **executee sur une machine sans Python** |
| Creation du venv et activation | executees, `sys.prefix` verifie |
| `pip install -r requirements-lite.txt` | execute, **55 paquets**, plages de versions validees |
| `pytest` | **117/117** en natif ET en conteneur (le 117e test verifie la calibration du seuil semantique) |
| Collecte Wikipedia | executee sur l'API reelle, en natif et en conteneur |
| Pretraitement, chunking, index TF-IDF | executes dans les deux environnements |
| `src.rag --ask` et `--compare` | executes, scores identiques dans les deux |
| `src.evaluate --retriever tfidf` | execute dans les deux, chiffres ci-dessus |
| Interface Streamlit | **chargee dans un navigateur** (Docker), question posee, reponse et sources affichees ; demarrage verifie en natif (`/_stcore/health` repond `ok`) |
| `docker build` (variante legere) | reussi, image de **865 Mo** |
| Conteneur Docker | demarre, etat `(healthy)` |
| `src.vector_store` (backend NumPy) | execute, auto-similarite a 1,0000 |
| Compilation de tous les modules | `compileall` sans erreur, y compris `embeddings.py` |
| `requirements.lock.txt` | **genere** par `pip freeze` sur l'installation reelle |
| L'image Docker **complete** | construite, **1,89 Go** (apres correction de la roue torch CPU) |
| `sentence-transformers` + PyTorch | installes et executes (`torch-2.14.1+cpu`) |
| `src.embeddings` | execute : 0,761 contre -0,035, **le modele capte le sens** |
| Index vectoriel sur le corpus reel | construit, 1 345 vecteurs, 3 min 02 s, 5,2 Mo |
| `src.evaluate --retriever embeddings` | execute, **toutes les metriques semantiques mesurees** |
| `src.evaluate --compare` | execute, les deux moteurs dans la meme execution |
| `MIN_SCORE_EMBEDDINGS` | **calibre sur mesures** : 0,30 estime -> 0,43 mesure |

**Ce qui n'a PAS ete execute, et doit donc etre verifie par vous :**

| Element | Pourquoi | Comment le verifier |
| --- | --- | --- |
| Le backend `chroma` | non installe | `pip install chromadb`, puis `VECTOR_BACKEND=chroma` et `python -m src.vector_store` |
| Les appels a un LLM distant (Groq, Gemini) | aucune cle API utilisee ; seuls les 4 fournisseurs **simules** sont testes dans `tests/test_llm.py` | creer une cle gratuite, puis `python -m src.llm` |
| Ollama | non installe | `ollama pull qwen2.5:1.5b-instruct` puis `python -m src.llm` |
| L'experience B2 (sentiment) | telechargement de 270 Mo non effectue | `python -m src.nlp_models --sentiment` |
| Le moteur semantique **en natif sous Windows** | mesure en conteneur uniquement ; les temps y seront plus lents (voir A contre B) | `pip install -r requirements.txt` puis `python -m src.evaluate --compare` |
| Le deploiement en ligne | depend de votre compte et des conditions de l'hebergeur | etape 11 |

Le projet est concu pour que l'absence de `sentence-transformers` ne bloque
rien : `get_retriever()` bascule alors **automatiquement** sur TF-IDF avec un
avertissement, et l'application continue de fonctionner. C'est ce qui s'est
passe avant l'installation de PyTorch, et la sortie de `--compare` le
montrait :

```
--- 2. RECHERCHE SEMANTIQUE (embeddings) ---
    [!] Indisponible : La bibliotheque 'sentence-transformers' n'est pas installee.
```

Ce repli reste le comportement attendu avec `requirements-lite.txt` : c'est ce
qui permet a l'image legere et a l'hebergement gratuit de fonctionner.

Pourquoi vos chiffres differeront : Wikipedia evolue en permanence (les
articles grossissent, les liens changent), et les temps dependent du
processeur, du disque et de la charge de la machine.

---

# Annexe D — Livrables finaux

| Livrable | Fichier / emplacement | Etat |
| --- | --- | --- |
| Code source complet | `src/` (9 modules) | fourni |
| Interface web | `app.py`, `app_gradio.py` | fourni |
| Dependances | `requirements.txt`, `-lite`, `-gradio` | fourni |
| Verrou de versions | `requirements.lock.txt` | fourni : 55 paquets, genere par `pip freeze` sur Windows / Python 3.12.10 |
| Conteneurisation | `Dockerfile`, `.dockerignore`, `compose.yaml` | fourni |
| Configuration | `.env.example` | fourni |
| Protection du depot | `.gitignore` | fourni |
| Tests automatises | `tests/` (117 tests) | fourni |
| Jeu d'evaluation | `eval/questions.json` | fourni, **a adapter a votre corpus** |
| Harnais de mesure | `src/evaluate.py` | fourni |
| Rapport d'evaluation | `eval/results_*.json` et `.md` | **genere par vos mesures** |
| CI | `.github/workflows/tests.yml` | fourni |
| Documentation | `README.md` | fourni |
| Deploiement | `docs/DEPLOYMENT.md` | fourni, **a completer avec votre URL** |
| Schema d'architecture | `README.md` et seance 0 | fourni |
| Checklist finale | annexe B | fourni |
| Grille d'evaluation | annexe E | fourni |
| Script d'orchestration | `run_all.ps1` | fourni |
| Captures d'ecran | votre compte rendu | **a produire** |

---

# Annexe E — Grille d'evaluation du TP

**Total : 100 points.** Les points marques (M) exigent une **mesure**, pas une
affirmation.

| # | Critere | Points | Preuve attendue |
| --- | --- | --- | --- |
| 1 | **Environnement** | **8** | |
| 1a | venv fonctionnel, dependances installees | 4 | `pip list`, invite `(.venv)` |
| 1b | `.env` cree et **exclu de git** | 4 | `git check-ignore -v .env` |
| 2 | **Collecte** | **12** | |
| 2a | 20 a 50 articles, corpus thematiquement coherent | 5 | sortie de `--stats` (M) |
| 2b | Deduplication effective | 3 | articles = titres uniques (M) |
| 2c | Script relancable | 2 | 2e execution n'ajoute rien |
| 2d | Attribution conservee (URL, licence) | 2 | champs du JSONL |
| 3 | **Pretraitement** | **12** | |
| 3a | Nettoyage effectif, reduction 5-20 % | 4 | rapport de `preprocess` (M) |
| 3b | **Les deux representations** et leur justification | 6 | code + explication orale |
| 3c | Documents trop courts ecartes | 2 | compteur du rapport |
| 4 | **Experiences NLP** | **14** | |
| 4a | Experience A : frequences, TF-IDF, cosinus | 5 | sorties relevees (M) |
| 4b | Experience B : embeddings, comparaison | 5 | matrice + recouvrement (M) |
| 4c | Experience C : un LLM teste | 2 | sortie de `src.llm` |
| 4d | Tableau avantages / limites par approche | 2 | compte rendu |
| 5 | **Systeme RAG** | **18** | |
| 5a | Chunking avec chevauchement correct | 4 | pas de debut tronque |
| 5b | Deux moteurs fonctionnels et compares | 5 | sortie de `--compare` (M) |
| 5c | Base vectorielle persistante | 3 | `data/index/` apres redemarrage |
| 5d | **Garde-fou anti-hallucination** | 6 | refus sans source |
| 6 | **Interface web** | **12** | |
| 6a | Interface fonctionnelle et responsive | 4 | capture d'ecran |
| 6b | **Sources affichees** avec licence | 4 | capture d'ecran |
| 6c | Message explicite si aucun passage | 2 | capture d'ecran |
| 6d | Erreurs d'API gerees, mode demo sans cle | 2 | demonstration |
| 7 | **Tests et evaluation** | **12** | |
| 7a | Tests passent | 4 | sortie de `pytest` |
| 7b | `refusal_rate_out_of_corpus` = 1.0 | 4 | rapport (M) |
| 7c | `false_refusals_in_corpus` = 0, seuil justifie | 2 | rapport (M) |
| 7d | Temps et memoire releves | 2 | rapport (M) |
| 8 | **Docker** | **6** | |
| 8a | Image construite, conteneur `(healthy)` | 3 | `docker compose ps` |
| 8b | Secrets hors image, donnees persistantes | 3 | `down` puis `up` |
| 9 | **Git / GitHub / deploiement** | **6** | |
| 9a | Depot propre, **aucun secret**, CI verte | 4 | onglet Actions |
| 9b | Deploiement en ligne **ou** demo Docker documentee | 2 | URL ou capture |

### Bareme indicatif

| Note | Niveau atteint |
| --- | --- |
| 90-100 | Tout fonctionne, mesures completes, choix techniques justifies et limites comprises |
| 75-89 | Pipeline complet et garde-fou operationnel ; mesures partielles |
| 60-74 | RAG fonctionnel, mais evaluation ou conteneurisation incomplete |
| 45-59 | Collecte et recherche fonctionnelles, RAG fragile |
| < 45 | Pipeline incomplet |

### Points qui distinguent un bon TP d'un excellent TP

Un excellent rendu ne se contente pas de faire tourner le code. Il **montre
qu'il a mesure, compris et corrige** :

1. Il explique pourquoi il y a **deux representations** du texte.
2. Il sait dire pourquoi un **seuil unique** ne peut pas convenir a deux
   moteurs dont les scores n'ont pas la meme echelle, chiffres a l'appui.
3. Il montre un cas ou **TF-IDF se trompe** et ou le semantique corrige — ou
   l'inverse.
4. Il assume ses **limites** au lieu de les cacher.
5. Il a verifie les conditions **reelles** des offres gratuites, plutot que de
   recopier un tutoriel perime.

---

## Mot de la fin

Ce TP vous a fait construire une application complete, mais surtout il vous a
fait rencontrer de vrais problemes :

- un corpus incoherent parce que le tri etait alphabetique ;
- des bibliographies polluant la recherche parce que le format des donnees
  n'etait pas celui qu'on supposait ;
- des chunks tronques en milieu de mot parce que le chevauchement comptait des
  caracteres au lieu de phrases ;
- un systeme qui refusait **toutes** les questions parce qu'un seuil unique
  servait deux echelles de scores differentes.

Aucun de ces problemes n'etait visible « a l'oeil ». Tous ont ete trouves en
**regardant les donnees reelles** et en **mesurant**. C'est la competence
centrale d'un ingenieur NLP, bien plus que la connaissance d'une bibliotheque
particuliere.

Bon courage, et surtout : **mesurez vos propres chiffres.**


