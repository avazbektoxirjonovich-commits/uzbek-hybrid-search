"""4.5 — Hybrid RAG: BM25 + Dense (FAISS) + Reciprocal Rank Fusion."""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

import config


class UzbekWikiRAG:
    def __init__(self, chunks_path=config.CHUNKS_PATH, model_name=config.EMBEDDING_MODEL,
                 index_dir=config.INDEX_DIR, rebuild=False):
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        self.texts = [c['text'] for c in self.chunks]

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name)
        self._build_or_load_bm25(rebuild)
        self._build_or_load_dense(rebuild)

        self._reranker = None  # lazy-load (bonus, section 4.9)

    # ---------- BM25 ----------
    def _build_or_load_bm25(self, rebuild):
        bm25_path = self.index_dir / 'bm25.pkl'
        if bm25_path.exists() and not rebuild:
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            return
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)

    # ---------- Dense / FAISS ----------
    def _build_or_load_dense(self, rebuild):
        faiss_path = self.index_dir / 'faiss.index'
        emb_path = self.index_dir / 'embeddings.npy'
        if faiss_path.exists() and emb_path.exists() and not rebuild:
            self.faiss_index = faiss.read_index(str(faiss_path))
            return
        embs = self.model.encode(
            self.texts, batch_size=64, show_progress_bar=True
        ).astype('float32')
        faiss.normalize_L2(embs)
        self.faiss_index = faiss.IndexFlatIP(embs.shape[1])
        self.faiss_index.add(embs)
        np.save(emb_path, embs)
        faiss.write_index(self.faiss_index, str(faiss_path))

    # ---------- Individual retrievers (baholash uchun) ----------
    def bm25_search(self, query: str, top_k: int = 5):
        scores = self.bm25.get_scores(query.lower().split())
        top_ids = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_ids]

    def dense_search(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        _, top_ids = self.faiss_index.search(q_emb, top_k)
        return [self.chunks[i] for i in top_ids[0]]

    # ---------- Hybrid (RRF) ----------
    def search(self, query: str, top_k: int = 5):
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = bm25_scores.argsort()[-config.BM25_TOP_N:][::-1]

        q_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        _, dense_top = self.faiss_index.search(q_emb, config.DENSE_TOP_N)
        dense_top = dense_top[0]

        rrf = defaultdict(float)
        k = config.RRF_K
        for r, i in enumerate(bm25_top):
            rrf[i] += 1 / (k + r + 1)
        for r, i in enumerate(dense_top):
            rrf[i] += 1 / (k + r + 1)

        top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return [self.chunks[i] for i in top_ids]

    # ---------- 4.9 Bonus: Cross-Encoder reranking ----------
    def search_reranked(self, query: str, top_k: int = 3, candidate_k: int = 10):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANKER_MODEL)

        candidates = self.search(query, top_k=candidate_k)
        pairs = [[query, c['text']] for c in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]
