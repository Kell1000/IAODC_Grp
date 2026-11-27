<#
.SYNOPSIS
    Lance tout le pipeline WikiNLP Assistant d'un seul coup.

.DESCRIPTION
    Enchaine collecte -> pretraitement -> index -> tests, en s'arretant a la
    premiere erreur avec un message clair. Pratique pour reprendre le TP
    apres une pause, ou pour verifier que tout fonctionne encore.

.PARAMETER MaxArticles
    Nombre d'articles a collecter (defaut : 30).

.PARAMETER Retriever
    'tfidf' (rapide, aucun telechargement) ou 'embeddings' (semantique).

.PARAMETER SkipCollect
    Ne pas relancer la collecte (utile si le corpus est deja la).

.PARAMETER Launch
    Lancer l'interface Streamlit a la fin.

.EXAMPLE
    .\run_all.ps1
    .\run_all.ps1 -MaxArticles 40 -Retriever embeddings -Launch
    .\run_all.ps1 -SkipCollect -Retriever tfidf
#>

[CmdletBinding()]
param(
    [int]$MaxArticles = 30,
    [ValidateSet('tfidf', 'embeddings')]
    [string]$Retriever = 'tfidf',
    [switch]$SkipCollect,
    [switch]$Launch
)

$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)
    Write-Step $Label
    # & lance python avec ses arguments ; $LASTEXITCODE porte le code retour.
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ECHEC] $Label (code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Corrigez l'erreur ci-dessus avant de continuer." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ---------------------------------------------------------------------
# 0. Verifications prealables
# ---------------------------------------------------------------------
Write-Step "0. Verification de l'environnement"

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Host "[ECHEC] python est introuvable dans le PATH." -ForegroundColor Red
    Write-Host "Installez Python 3.12 :  winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}
if ($pythonCommand.Source -like "*WindowsApps*") {
    Write-Host "[ECHEC] 'python' pointe sur le raccourci du Microsoft Store." -ForegroundColor Red
    Write-Host "Python n'est pas reellement installe." -ForegroundColor Yellow
    Write-Host "  winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}
Write-Host "  python : $($pythonCommand.Source)"
& python --version

if (-not $env:VIRTUAL_ENV) {
    Write-Host ""
    Write-Host "[!] Aucun environnement virtuel actif." -ForegroundColor Yellow
    Write-Host "    Recommande :  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}
else {
    Write-Host "  venv   : $env:VIRTUAL_ENV"
}

if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "[!] .env absent : creation depuis .env.example" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "    Pensez a renseigner WIKI_USER_AGENT avec un contact reel." -ForegroundColor Yellow
}

# Impose le moteur demande pour toute la duree du script (et seulement elle).
$env:RETRIEVER = $Retriever
Write-Host "  RETRIEVER = $Retriever"

# ---------------------------------------------------------------------
# 1 a 4. Pipeline
# ---------------------------------------------------------------------
if (-not $SkipCollect) {
    Invoke-Step "1. Collecte Wikipedia ($MaxArticles articles max)" `
        @('-m', 'src.collect_wikipedia', '--max', "$MaxArticles")
}
else {
    Write-Step "1. Collecte ignoree (-SkipCollect)"
}

Invoke-Step "2. Pretraitement NLP" @('-m', 'src.preprocess')
Invoke-Step "3. Construction de l'index" @('-m', 'src.rag', '--build')
Invoke-Step "4. Tests automatises" @('-m', 'pytest', '-q')

Write-Step "5. Etat final du pipeline"
& python -m src.rag --info

Write-Host ""
Write-Host "[OK] Pipeline complet termine." -ForegroundColor Green
Write-Host ""
Write-Host "Etapes suivantes possibles :" -ForegroundColor Green
# Dans une chaine entre apostrophes simples, '' represente une apostrophe.
# (Enchainer '...'"'"'...' ferait passer plusieurs ARGUMENTS a Write-Host,
#  qui les joindrait avec des espaces : "Qu ' est-ce".)
Write-Host '  python -m src.rag --ask "Qu''est-ce que l''apprentissage profond ?"'
Write-Host '  python -m src.rag --compare "Qu''est-ce qu''un reseau de neurones ?"'
Write-Host '  python -m src.evaluate --compare'
Write-Host '  streamlit run app.py'

if ($Launch) {
    Write-Step "Lancement de l'interface Streamlit"
    Write-Host "  Ouvrez http://localhost:8501  (Ctrl+C pour arreter)"
    & streamlit run app.py
}
