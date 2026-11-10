# Ziyad Food - RAG Chatbot

## Description
Ziyad Food is an AI-powered restaurant chatbot using
Retrieval-Augmented Generation (RAG).

It retrieves relevant information from restaurant documents
and uses a local language model to generate answers.

## Technologies
- Python
- Streamlit
- Ollama
- Qwen3
- Nomic Embed Text
- NumPy
- PyPDF

## Requirements
- Python
- Ollama
- nomic-embed-text
- qwen3:4b-instruct

## Installation
Install dependencies:

uv sync

## Run
uv run streamlit run app.py

## Project Structure
- app.py: Chatbot interface
- rag.py: RAG implementation
- docs/: Restaurant knowledge base