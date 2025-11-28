import os
import numpy as np
import faiss
import requests
import json
from pathlib import Path

EMBED_MODEL = "nomic-embed-text"

def ollama_embed(text):
    """Llama a Ollama para obtener embeddings."""
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    data = r.json()
    return np.array(data["embedding"], dtype="float32")


def load_txt_files(folder_path):
    """Carga todos los .txt de un directorio."""
    texts = []
    paths = list(Path(folder_path).glob("*.txt"))
    for path in paths:
        print("Leyendo:", path.name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        texts.append((path.name, text))
    return texts


def build_faiss_index(texts):
    """Construye un índice FAISS usando embeddings."""
    # Obtener dimensión de embedding probando con un texto
    dim = len(ollama_embed("test text"))

    index = faiss.IndexFlatL2(dim)

    metadata = []  # Aquí guardamos (filename, texto original)
    vectors = []

    for filename, content in texts:
        print(f"Embed → {filename}")
        emb = ollama_embed(content)
        vectors.append(emb)
        metadata.append((filename, content))

    vectors = np.vstack(vectors).astype("float32")
    index.add(vectors)

    print("\nFAISS Index creado:")
    print("  - Dimensión:", dim)
    print("  - Cantidad de documentos:", index.ntotal)

    return index, metadata



def main():
    folder = "context_files"
    texts = load_txt_files(folder)
    output_folder_name = "embeddings"
    os.makedirs(output_folder_name, exist_ok=True)
    index, metadata = build_faiss_index(texts)
    faiss.write_index(index, os.path.join(output_folder_name, "faiss.index"))
    with open(os.path.join(output_folder_name, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print("\nArchivos guardados:")
    print(" - faiss.index")
    print(" - metadata.json")


if __name__ == "__main__":
    main()