"""Umumiy sozlamalar — Hybrid RAG pipeline uchun."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
WIKI_OUTPUT_DIR = BASE_DIR / "wiki_output"
ARTICLES_PATH = BASE_DIR / "articles.jsonl"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"
INDEX_DIR = BASE_DIR / "index"
QA_PATH = BASE_DIR / "qa_pairs.json"
RESULTS_PATH = BASE_DIR / "results.md"

# Resurs cheklovi tufayli (CPU-only, ~5GB bo'sh RAM) butun ~180k maqola o'rniga
# tasodifiy, seed bilan takrorlanuvchi N_ARTICLES ta maqola tanlanadi.
# To'liq dump bilan ishlash uchun None qiling.
N_ARTICLES = 15000
RANDOM_SEED = 42
MIN_WORDS = 50  # spec: stub maqolalarni o'tkazib yuborish

CHUNK_MAX_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_TOP_N = 20
DENSE_TOP_N = 20
RRF_K = 60
