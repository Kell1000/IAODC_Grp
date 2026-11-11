import requests
import numpy as np
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


def embed(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def ask_llm(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def load_documents():
    documents = []

    for file in DOCS_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif file.suffix.lower() == ".md":
            text = file.read_text(encoding="utf-8")
        else:
            continue

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        documents.extend(paragraphs)

    return documents


def build_index():
    chunks = load_documents()

    if not chunks:
        raise ValueError("No documents found in the docs folder.")

    embeddings = embed(chunks)
    return chunks, np.array(embeddings)


def retrieve(question, chunks, embeddings, top_k=3):
    question_embedding = np.array(embed([question])[0])

    scores = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(question_embedding)
        + 1e-10
    )

    indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in indices]


def answer_question(question, chunks, embeddings):
    relevant_chunks = retrieve(question, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are Ziyad Food, a helpful restaurant assistant.

Answer the user's question using ONLY the information in the context.
If the context does not contain the answer, say:
"I don't know based on the available restaurant information."

Context:
{context}

Question:
{question}
"""

    return ask_llm(prompt)


if __name__ == "__main__":
    chunks, embeddings = build_index()

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        print("\nZiyad Food:", answer_question(question, chunks, embeddings))