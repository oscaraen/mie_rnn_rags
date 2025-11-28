import faiss
import numpy as np
import requests
import json
import os


# Modelos
LLM_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
embeddings_folder = "embeddings"
# --- Embeddings ---
def ollama_embed(text):
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    data = r.json()
    return np.array(data["embedding"], dtype="float32")

def load_faiss(folder):
    # Cargar índice
    index_path = os.path.join(folder, "faiss.index")
    metadata_path = os.path.join(folder, "metadata.json")
    index = faiss.read_index(index_path)
    # Cargar metadata desde JSON
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

index, metadata = load_faiss(embeddings_folder)

def search_index(query, k=3):
    print(">> Buscando documentos relevantes...")
    query_emb = ollama_embed(query).reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    docs = [" ".join(metadata[i]) for i in indices[0]]
    return docs



def generate_with_ollama(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt},
        stream=True
    )

    full_text = ""

    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        piece = data.get("response", "")
        print(piece, end="", flush=True)
        full_text += piece

    print()  # salto de línea final
    return full_text


def build_prompt(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    prompt = None
    return prompt


def rag(query, k=3):
    docs = search_index(query, k=k)
    prompt = build_prompt(query, docs)
    print("\n--- Respuesta del modelo ---\n")
    return generate_with_ollama(prompt)


if __name__ == "__main__":
    while True:
        q = input("\nPregunta > ")
        if q.lower() in ("salir", "exit"):
            break

        answer = rag(q, k=3)


