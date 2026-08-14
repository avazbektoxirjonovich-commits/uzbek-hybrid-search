# Uzbek Hybrid Search

**O'zbek Wikipediasi ustidagi hybrid information-retrieval dvigateli — BM25 + dense (FAISS) qidiruv Reciprocal Rank Fusion bilan birlashtirilgan, ixtiyoriy cross-encoder qayta tartiblash bilan. Faqat qidiruv; javob generatsiyasi yo'q.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-dense%20retrieval-005571)
![BM25](https://img.shields.io/badge/BM25-sparse%20retrieval-8E44AD)
![Litsenziya](https://img.shields.io/badge/litsenziya-MIT-blue)

[English](README.md) | [O'zbek](README.uz.md)

> **Kelib chiqishi, halol aytilgan:** bu loyiha strukturaviy ML-kurs moduli sifatida qurilgan (ichki log/izohlarda "4.5 — Hybrid RAG", "4.7 — Baholash", "4.9 — Reranking bonus" havolalari bor), mustaqil ravishda o'ylab topilgan yon loyiha sifatida emas. Implementatsiyaning o'zi esa haqiqiy, darslikka mos hybrid IR — bu bevosita kodni o'qish orqali tasdiqlangan, faqat fayl nomlari orqali emas.

## Tavsif

Bu chatbot emas, qidiruv tizimi: so'rov berilganda, u eng tegishli o'zbek Wikipedia matn bo'laklarini, leksik (BM25) va semantik (dense embedding / FAISS) retrieverlarni birlashtirib, tartiblab qaytaradi. Bu yerda LLM ishtirok etmaydi — `evaluate.py` retrieval sifatini to'g'ridan-to'g'ri o'lchash uchun mavjud (Hit Rate@5, QA to'plamiga nisbatan F1), bu portfolioning boshqa joylaridagi RAG *chatbotlar*idan farqli, to'ldiruvchi ko'nikma.

## Xususiyatlar

- BM25 sparse retrieval (`rank_bm25.BM25Okapi`)
- `sentence-transformers` embeddinglari orqali FAISS `IndexFlatIP`da dense retrieval (L2-normallashtirilgan inner product orqali cosine o'xshashlik)
- Ikkala tartiblashni birlashtiruvchi Reciprocal Rank Fusion
- Ixtiyoriy, lazy-load qilinadigan cross-encoder qayta tartiblash bosqichi
- BM25-only, Dense-only va Hybrid retrievalni solishtiruvchi mustaqil baholash mexanizmi

## Arxitektura

[docs/architecture/pipeline.svg](docs/architecture/pipeline.svg) ga qarang.

```
uzwiki Wikipedia dump (manba, kiritilmagan — quyidagi Ma'lumotlarga qarang)
    ↓
run_extractor.py → clean_data.py → chunking.py
    ↓
build_index.py  →  BM25 indeks (rank_bm25)  +  FAISS indeks (dense embeddinglar)
    ↓
rag_system.py: UzbekWikiRAG.search()
    ↓
Reciprocal Rank Fusion  →  (ixtiyoriy) Cross-Encoder qayta tartiblash
    ↓
Tartiblangan matn bo'laklari
```

## AI/ML Quvuri

- **Embedding modeli:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Reranker modeli:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (faqat `search_reranked()` chaqirilganda lazy-load qilinadi)
- **Fusion:** `RRF(d) = Σ 1/(k + rank)`, `k=60`, har bir retrieverning top-20 nomzodini birlashtiradi
- **Korpus:** o'zbek Wikipedia dumpidan tasodifiy, seed bilan tanlangan 15,000 ta maqola (`N_ARTICLES = 15000`, `RANDOM_SEED = 42`) — to'liq ~180k maqolali dump emas, manba izohlarida qayd etilgan mahalliy hisoblash/RAM cheklovi tufayli

## Baholash

`evaluate.py` haqiqiy solishtirish mexanizmini amalga oshiradi (Hit Rate@5 va F1, BM25-only / Dense-only / Hybrid bo'yicha), lekin **hech qanday baholash haqiqatan ishga tushirilmagan va saqlanmagan** — `qa_pairs.json` (zarur savol-javob benchmark to'plami) hozircha mavjud emas, `results.md` hech qachon yaratilmagan. Bu shu yerda o'ylab topilgan raqamlarni ko'rsatish o'rniga ochiqchasiga aytilmoqda: **bu repo'da baholash grafigi yo'q**, va quyidagi Roadmap buni ochiq ish sifatida ko'rsatadi.

## Ma'lumotlar

Manba Wikipedia dumpi (`uzwiki-*.xml`, ~1.3GB) va barcha generatsiya qilingan artefaktlar (`wiki_output/`, `articles.jsonl`, `chunks.pkl`, `index/`) **bu repo'ga kiritilmagan** — ular quyidagi pipeline skriptlarini ishga tushirish orqali mahalliy qayta yaratiladi. `run_extractor.py`'ni ishga tushirishdan oldin joriy o'zbek Wikipedia dumpini [dumps.wikimedia.org](https://dumps.wikimedia.org/uzwiki/) dan yuklab oling.

## O'rnatish

```bash
git clone <this-repository>
cd uzbek-hybrid-search
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. uzwiki-*.xml dumpini shu papkaga yuklab oling, so'ng:
python run_extractor.py
python clean_data.py
python chunking.py
python build_index.py

# 2. Sinab ko'ring
python demo.py
```

## Roadmap

- [ ] `qa_pairs.json` benchmark to'plamini yozish va `evaluate.py`ni haqiqatan ishga tushirish, so'ng real natijalarni nashr qilish
- [ ] Fusion/reranking logikasi uchun avtomatlashtirilgan testlar qo'shish
- [ ] Yetarli hisoblash resursi bo'lsa, 15,000 maqolali namuna o'rniga to'liq ~180k maqolali dumpni ko'rib chiqish

## Litsenziya

MIT — qarang [LICENSE](LICENSE).
