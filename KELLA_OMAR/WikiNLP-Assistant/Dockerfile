# =====================================================================
#  WikiNLP Assistant - image Docker
# =====================================================================
#  IMAGE vs CONTENEUR (a savoir expliquer en entretien)
#    image     = modele fige, en lecture seule, construit par `docker build`
#    conteneur = instance en cours d'execution de cette image, jetable
#  On peut lancer 10 conteneurs depuis 1 image, comme 10 objets depuis
#  1 classe.
#
#  DEUX VARIANTES, choisies a la construction :
#
#    # Image LEGERE (865 Mo mesure) : TF-IDF seul, demarre en 2 s
#    docker build -t wikinlp:lite .
#
#    # Image COMPLETE : recherche semantique (torch inclus, nettement plus lourde)
#    docker build --build-arg REQUIREMENTS=requirements.txt -t wikinlp:full .
#
#  Ce qui n'entre JAMAIS dans l'image (voir .dockerignore) :
#    - le fichier .env et les cles API  -> injectes au lancement
#    - les donnees collectees            -> montees en volume
#    - les caches de modeles             -> montes en volume
#  Une cle API copiee dans une image reste lisible dans son historique de
#  couches, meme apres suppression du fichier dans une couche suivante.
# =====================================================================

FROM python:3.12-slim

# ---------------------------------------------------------------------
# Variables d'environnement de l'interpreteur
# ---------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
#   PYTHONUNBUFFERED      : les logs sortent immediatement (sinon
#                           `docker logs` reste vide et le debogage est aveugle)
#   PYTHONDONTWRITEBYTECODE : pas de .pyc, inutiles dans un conteneur jetable
#   PIP_NO_CACHE_DIR      : evite ~200 Mo de cache pip dans l'image

WORKDIR /app

# ---------------------------------------------------------------------
# Dependances : copiees SEULES et AVANT le code source.
# Docker met chaque etape en cache : tant que les requirements ne changent
# pas, modifier app.py ne relance pas l'installation des paquets.
# Inverser ces deux COPY ferait reinstaller torch a chaque edition de code.
# ---------------------------------------------------------------------
ARG REQUIREMENTS=requirements-lite.txt
COPY requirements.txt requirements-lite.txt ./
# Sur Linux (donc dans cette image), `pip install torch` tire par defaut la
# variante CUDA : ~2,4 Go de bibliotheques NVIDIA inutiles sans GPU, et un
# telechargement qui echoue souvent. On installe donc EXPLICITEMENT la roue
# CPU (~200 Mo) depuis l'index officiel PyTorch, AVANT requirements.txt :
# sentence-transformers trouve alors torch deja satisfait et ne retelecharge
# rien. Sous Windows, `pip install torch` donne deja la variante CPU : cette
# etape ne concerne que l'image Docker.
RUN python -m pip install --upgrade pip && \
    if [ "${REQUIREMENTS}" = "requirements.txt" ]; then \
        python -m pip install --index-url https://download.pytorch.org/whl/cpu torch ; \
    fi && \
    python -m pip install -r ${REQUIREMENTS}

# ---------------------------------------------------------------------
# Ressources NLTK, telechargees a la CONSTRUCTION et non au premier
# lancement : un conteneur doit pouvoir demarrer sans acces reseau.
# `|| true` : si le telechargement echoue, le code bascule sur ses
# solutions de secours (regex + liste de stop words interne).
# ---------------------------------------------------------------------
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt punkt_tab stopwords || true

# ---------------------------------------------------------------------
# Code source
# ---------------------------------------------------------------------
COPY src/ ./src/
COPY app.py app_gradio.py pyproject.toml ./
COPY eval/questions.json ./eval/questions.json

# Dossiers de travail. Ils seront normalement recouverts par des volumes,
# mais doivent exister pour que l'application demarre meme sans volume.
RUN mkdir -p data/raw data/processed data/index hf_cache

# ---------------------------------------------------------------------
# Utilisateur non privilegie : un processus compromis dans le conteneur
# ne doit pas etre root. C'est une regle de securite de base.
# ---------------------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Cache des modeles DANS /app (monte en volume) et non dans /home :
# sinon le modele serait retelecharge a chaque recreation du conteneur.
ENV HF_HOME=/app/hf_cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Verification de sante : Streamlit expose /_stcore/health.
# `docker ps` affichera (healthy) ou (unhealthy).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

# --server.address=0.0.0.0 est OBLIGATOIRE dans un conteneur : avec la
# valeur par defaut (localhost), le serveur n'ecoute que sur la boucle
# interne du conteneur et le port publie reste muet.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
