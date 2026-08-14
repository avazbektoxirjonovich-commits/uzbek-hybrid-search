"""4.7 — Tizimni baholash: BM25 vs Dense vs Hybrid (Hit Rate@5, F1)."""
import json

import config
from rag_system import UzbekWikiRAG


def evaluate_retriever(search_fn, qa_pairs: list, top_k: int = 5) -> dict:
    retrieval_hits = 0
    f1_scores = []

    for qa in qa_pairs:
        results = search_fn(qa['query'], top_k)
        combined = ' '.join([r['text'] for r in results]).lower()

        if qa['answer'].lower() in combined:
            retrieval_hits += 1

        pred_words = set(combined.split())
        true_words = set(qa['answer'].lower().split())
        if true_words:
            precision = len(pred_words & true_words) / len(pred_words) if pred_words else 0
            recall = len(pred_words & true_words) / len(true_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

    return {
        'Hit Rate@5': retrieval_hits / len(qa_pairs),
        'Avg F1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'Total QA': len(qa_pairs),
    }


def main():
    with open(config.QA_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    print(f"QA juftliklari: {len(qa_pairs)}")

    rag = UzbekWikiRAG()

    systems = {
        'BM25': rag.bm25_search,
        'Dense (FAISS)': rag.dense_search,
        'Hybrid (RRF)': rag.search,
    }

    rows = []
    for name, fn in systems.items():
        print(f"Baholanmoqda: {name} ...")
        res = evaluate_retriever(fn, qa_pairs, top_k=5)
        rows.append((name, res))
        print(f"  {res}")

    lines = [
        "| Tizim | Hit Rate@5 | Avg F1 |",
        "|---|---|---|",
    ]
    for name, res in rows:
        lines.append(f"| {name} | {res['Hit Rate@5']:.3f} | {res['Avg F1']:.3f} |")
    table = '\n'.join(lines)
    print('\n' + table)

    with open(config.RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# Baholash natijalari (QA soni: {len(qa_pairs)})\n\n{table}\n")
    print(f"\nSaqlandi: {config.RESULTS_PATH}")


if __name__ == '__main__':
    main()
