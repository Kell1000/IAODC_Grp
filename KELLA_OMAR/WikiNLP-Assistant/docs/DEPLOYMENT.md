# Deploiement gratuit de WikiNLP Assistant

> **A lire d'abord.** Les offres gratuites decrites ici ont ete verifiees en
> **octobre 2026** sur les pages officielles des plateformes. Elles changent
> regulierement, parfois du jour au lendemain. **Reverifiez-les avant de
> deployer.** Aucun hebergeur gratuit ne garantit une disponibilite
> permanente : une application gratuite peut etre mise en veille, bridee ou
> arretee sans preavis.

---

## 1. Le probleme a resoudre

Notre application a deux besoins difficiles a satisfaire gratuitement :

1. **De la RAM.** Avec `RETRIEVER=embeddings`, il faut charger PyTorch et un
   modele de 470 Mo : comptez 700 Mo a 1 Go de memoire vive. La plupart des
   offres gratuites plafonnent bien en dessous.
2. **Un processus qui tourne en continu** (serveur web), et non un simple
   site statique.

La strategie retenue est donc :

> **heberger une application LEGERE qui appelle une API LLM gratuite.**
> Le serveur ne fait que la recherche lexicale (quelques dizaines de Mo) ;
> la generation de texte, couteuse, est deleguee a une API externe.

Concretement, en production gratuite : `RETRIEVER=tfidf` et
`LLM_PROVIDER=groq` (ou `extractive` pour une demonstration sans cle).

---

## 2. Etat verifie des plateformes (octobre 2026)

### Hugging Face Spaces

La documentation officielle (`huggingface.co/docs/hub/spaces-overview`)
indique :

> « Static Spaces are free for everyone. Gradio and Docker Spaces run on
> compute and require a paid plan to create: PRO for personal accounts, Team
> or Enterprise for organizations. Free personal accounts in good standing
> can still host up to 2 Gradio Spaces running on ZeroGPU. »

Ce qu'il faut en retenir pour ce TP :

| Point | Consequence |
| --- | --- |
| Les SDK proposes sont Gradio, Docker et HTML statique | **Streamlit n'est plus un SDK de Spaces** : `app.py` n'est pas deployable tel quel |
| Un Space Docker exige un compte payant | la voie Docker est fermee a budget zero |
| Un compte personnel gratuit peut avoir **2 Spaces Gradio sur ZeroGPU** | **c'est la seule voie gratuite** : on deploie `app_gradio.py` |
| Materiel CPU Basic | 2 vCPU, 16 Go RAM, 50 Go de disque **non persistant** |
| Mise en veille | « On free hardware, your Space will "go to sleep" and stop executing after a period of time if unused. » |

Le disque n'etant **pas persistant**, le corpus doit etre **versionne dans le
depot du Space** (voir section 3) et non genere au premier demarrage.

### Streamlit Community Cloud

C'est l'option la plus simple si l'on veut garder l'interface Streamlit.

| Point | Valeur constatee |
| --- | --- |
| Prix | gratuit |
| Nombre d'applications | 1 privee, publiques illimitees |
| Memoire | les limites officielles publiees (datees de fevrier 2024) vont d'environ **690 Mo a 2,7 Go**, « may change at any time without notice » |
| Mise en veille | oui, apres une periode d'inactivite (environ 12 h d'apres plusieurs sources non officielles) |
| Source | depot GitHub |

**Consequence directe :** avec moins de 1 Go garanti, ne tentez pas la
recherche semantique. Forcez `RETRIEVER=tfidf`.

### Autres pistes

| Plateforme | Remarque |
| --- | --- |
| Railway, Render, Fly.io | offres d'essai ou credits limites, conditions tres changeantes. A verifier au cas par cas. |
| Oracle Cloud / Google Cloud | paliers gratuits existants mais demandent une carte bancaire et de l'administration systeme. Hors perimetre d'un TP. |
| **Votre machine + Docker** | gratuit, sans limite, totalement maitrise. **C'est la solution de secours a privilegier** si tout le reste echoue. |

---

## 3. Solution principale : Space Gradio (gratuit)

### Etape 3.1 - Preparer un corpus leger et versionne

Le disque d'un Space n'est pas persistant : il faut embarquer le corpus.

```powershell
# Depuis le projet, avec un corpus deja construit
python -m src.rag --build

# Creer un mini-corpus versionnable (quelques centaines de chunks)
python -c "import json, itertools, pathlib; src=pathlib.Path('data/processed/chunks.jsonl'); dst=pathlib.Path('data/processed/demo_chunks.jsonl'); lines=src.read_text(encoding='utf-8').splitlines(); dst.write_text('\n'.join(lines[:400])+'\n', encoding='utf-8'); print(len(lines[:400]), 'chunks ->', dst)"
```

`demo_chunks.jsonl` est explicitement **autorise** par `.gitignore` (voir la
ligne `!data/processed/demo_chunks.jsonl`) : c'est le seul fichier de donnees
versionne. `src/rag.py: load_chunks()` l'utilise automatiquement si
`chunks.jsonl` est absent.

Verifiez son poids : il doit rester petit (moins de 2 Mo).

```powershell
(Get-Item data\processed\demo_chunks.jsonl).Length / 1KB
```

### Etape 3.2 - Creer le Space

1. Compte sur <https://huggingface.co> (gratuit).
2. **New Space** : SDK **Gradio**, visibilite **Public**, materiel **CPU Basic**.
3. Notez l'URL du depot : `https://huggingface.co/spaces/<user>/<nom>`.

### Etape 3.3 - Preparer les fichiers attendus par Spaces

Spaces impose deux conventions : le point d'entree s'appelle `app.py`, et les
dependances sont lues dans `requirements.txt`.

```powershell
# Dans un dossier de travail SEPARE, clone du Space
git clone https://huggingface.co/spaces/<user>/<nom> hf-space
cd hf-space

# Copier le strict necessaire
Copy-Item -Recurse ..\WikiNLP-Assistant\src .\src
Copy-Item ..\WikiNLP-Assistant\app_gradio.py .\app.py
Copy-Item ..\WikiNLP-Assistant\requirements-gradio.txt .\requirements.txt
New-Item -ItemType Directory -Force .\data\processed | Out-Null
Copy-Item ..\WikiNLP-Assistant\data\processed\demo_chunks.jsonl .\data\processed\
```

Creez l'en-tete `README.md` que Spaces exige :

```markdown
---
title: WikiNLP Assistant
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

Assistant RAG sur un corpus Wikipedia. Contenus sous licence CC BY-SA 4.0.
```

### Etape 3.4 - Variables d'environnement et secrets

Dans le Space : **Settings → Variables and secrets**.

| Nom | Type | Valeur | Role |
| --- | --- | --- | --- |
| `RETRIEVER` | Variable | `tfidf` | pas de torch : economise la RAM |
| `FORCE_TFIDF` | Variable | `1` | force le mode leger dans `app_gradio.py` |
| `LLM_PROVIDER` | Variable | `groq` ou `extractive` | `extractive` = demo sans cle |
| `MIN_SCORE` | Variable | `0.25` | seuil du garde-fou |
| `WIKI_LANG` | Variable | `fr` | |
| `GROQ_API_KEY` | **Secret** | `gsk_...` | **jamais une Variable** |

> **Variable ou Secret ?** Une *Variable* est publique et visible de tous, et
> elle est recopiee dans les Spaces dupliques. Un *Secret* est prive et
> illisible apres enregistrement. Une cle API va **toujours** dans un Secret.
> Hugging Face scanne les depots et vous alerte si une cle est ecrite en dur.

### Etape 3.5 - Publier

```powershell
git add .
git commit -m "Deploiement initial de WikiNLP Assistant"
git push
```

Le Space se reconstruit automatiquement. Suivez l'onglet **Logs**.

### Etape 3.6 - Verifications apres deploiement

| Verification | Resultat attendu |
| --- | --- |
| L'application s'affiche | interface Gradio, sans trace d'erreur |
| Le bandeau indique un nombre de passages | `Corpus : N passages` avec N > 0 |
| Question dans le corpus | reponse + sources cliquables |
| Question hors corpus (tarte tatin) | refus explicite, aucune source |
| Logs | aucun `ModuleNotFoundError`, aucune alerte de secret |
| Aucune cle visible | `Settings` : la cle est bien un Secret |

---

## 4. Variante : Streamlit Community Cloud

A preferer si vous voulez conserver l'interface `app.py`.

1. Poussez le projet sur GitHub (depot **public** de preference).
2. Verifiez que `data/processed/demo_chunks.jsonl` est bien versionne.
3. <https://share.streamlit.io> → **New app** → choisissez le depot,
   la branche, et `app.py` comme fichier principal.
4. **Advanced settings → Secrets** : Streamlit utilise le format TOML, et non
   un fichier `.env` :

```toml
RETRIEVER = "tfidf"
LLM_PROVIDER = "groq"
GROQ_API_KEY = "gsk_votre_cle"
MIN_SCORE = "0.25"
WIKI_LANG = "fr"
```

> `src/config.py` lit ces valeurs via `os.getenv`, et Streamlit Cloud expose
> les secrets comme variables d'environnement : aucune modification du code
> n'est necessaire.

5. **Deploy**, puis refaites les verifications du tableau 3.6.

**Piege le plus frequent :** l'application depasse la memoire autorisee et
redemarre en boucle. Cause quasi systematique : `RETRIEVER=embeddings`, qui
charge torch. Corrigez le secret en `tfidf`.

---

## 5. Solution de secours : Docker en local

Si aucune offre gratuite ne convient (conditions modifiees, quota epuise,
compte refuse), le projet reste integralement demontrable en local. C'est
suffisant pour une soutenance ou un entretien technique.

```powershell
docker compose up -d --build
Start-Process http://localhost:8501
```

Pour montrer l'application a quelqu'un sur le meme reseau local :

```powershell
# Trouver votre adresse IP locale
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" } | Select-Object IPAddress, InterfaceAlias
# L'autre poste ouvre http://<votre-ip>:8501
```

> N'exposez pas ce port sur Internet : l'application n'a ni authentification
> ni limitation de debit.

---

## 6. Obtenir une cle LLM gratuite

### Groq (recommande)

1. <https://console.groq.com> - inscription sans carte bancaire.
2. **API Keys → Create API Key**, copiez la cle (`gsk_...`).
3. **Consultez vos quotas reels** : `console.groq.com/settings/limits`.
   Cette page est la seule source fiable ; les limites varient selon le
   modele et le compte, et changent dans le temps.
4. Si un modele renvoie `404`, son nom a change : relevez un nom valide dans
   la documentation du fournisseur et corrigez `GROQ_MODEL`.

### Google Gemini

1. <https://aistudio.google.com/apikey> → **Create API key**.
2. Les limites ne sont plus publiees sous forme de tableau fige : consultez
   AI Studio pour votre projet et le modele appele.
3. **Avertissement sur la confidentialite** : sur l'offre gratuite, Google
   indique pouvoir utiliser les requetes et les reponses pour ameliorer ses
   modeles, avec relecture humaine possible. **N'y envoyez aucune donnee
   personnelle ou confidentielle.**

### Ollama (local, aucune cle, hors ligne)

```powershell
# Installer depuis https://ollama.com, puis :
ollama pull qwen2.5:1.5b-instruct     # ~1 Go sur disque
ollama list
# .env :  LLM_PROVIDER=ollama
```

Un modele 1,5 B quantifie occupe environ 1,5 Go de RAM : confortable sur
16 Go. Sur CPU, comptez plusieurs secondes par reponse.

---

## 7. Erreurs de deploiement les plus courantes

| Erreur observee | Cause probable | Correction |
| --- | --- | --- |
| `ModuleNotFoundError: src` | le dossier `src/` n'a pas ete copie | verifier l'arborescence du depot distant |
| `Aucun corpus trouve` | `demo_chunks.jsonl` absent ou exclu par `.gitignore` | `git add -f data/processed/demo_chunks.jsonl` |
| L'app redemarre en boucle | memoire depassee (torch charge) | `RETRIEVER=tfidf` dans les secrets |
| `401` / `403` sur l'API | cle absente, mal copiee, ou mise en Variable au lieu de Secret | recreer le Secret, redemarrer l'app |
| `429` | quota depasse | attendre, ou `LLM_PROVIDER=extractive` |
| Build Space en echec | `requirements.txt` contient `torch` | utiliser `requirements-gradio.txt` |
| Fonctionne en local, pas en ligne | un chemin absolu, ou un fichier ignore par git | tester dans un clone propre du depot |
| Space endormi | inactivite | le reveiller en ouvrant la page ; c'est le comportement normal du gratuit |

### Le test qui previent 90 % des mauvaises surprises

Un deploiement echoue presque toujours parce qu'un fichier existe sur votre
machine mais n'est pas dans le depot. Simulez ce que voit l'hebergeur :

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

Si cela fonctionne dans ce clone neuf, le deploiement a de bonnes chances de
reussir. Sinon, vous venez d'identifier le fichier manquant.
