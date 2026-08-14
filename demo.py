"""4.10 — Interaktiv CLI demo."""
import argparse

from rag_system import UzbekWikiRAG


def print_results(results, title):
    print(f"\n--- {title} ---")
    for i, r in enumerate(results, 1):
        snippet = r['text'][:220].replace('\n', ' ')
        print(f"[{i}] {r['article_title']}  (chunk_id={r['chunk_id']})")
        print(f"    {snippet}...")


def main():
    parser = argparse.ArgumentParser(description="O'zbek Vikipediyasi Hybrid RAG — CLI demo")
    parser.add_argument('--rerank', action='store_true', help="Cross-encoder bilan qayta tartiblash (bonus)")
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    print("Indeks yuklanmoqda...")
    rag = UzbekWikiRAG()
    print(f"Tayyor. Jami chunk: {len(rag.chunks)}. Chiqish uchun 'exit' yozing.\n")

    while True:
        try:
            query = input("Savol: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ('exit', 'quit', 'chiqish'):
            break

        hybrid = rag.search(query, top_k=args.top_k)
        print_results(hybrid, "Hybrid (BM25 + Dense, RRF)")

        if args.rerank:
            reranked = rag.search_reranked(query, top_k=min(3, args.top_k))
            print_results(reranked, "Reranked (Cross-Encoder, bonus)")


if __name__ == '__main__':
    main()
