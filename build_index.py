"""BM25 + FAISS indekslarini qurish va index/ papkasiga saqlash."""
from rag_system import UzbekWikiRAG
import config


def main():
    print("Indeks qurilmoqda (BM25 + Dense embeddings)...")
    rag = UzbekWikiRAG(rebuild=True)
    print(f"Tayyor. Chunk soni: {len(rag.chunks)}")
    print(f"Indeks saqlandi: {config.INDEX_DIR}")


if __name__ == '__main__':
    main()
