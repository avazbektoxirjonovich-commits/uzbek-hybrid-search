"""4.4 — Semantic chunking (sliding window, so'z darajasida)."""
import json
import pickle

import config


def chunk_article(article: dict, max_words: int = 150, overlap_words: int = 30):
    """Bir maqolani so'z soni asosida chunklarga bo'lish (metadata bilan)."""
    words = article['text'].split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = ' '.join(words[start:end])
        chunks.append({
            'article_id': article['id'],
            'article_title': article['title'],
            'text': chunk_text,
            'chunk_id': f"{article['id']}_{len(chunks)}",
        })
        if end == len(words):
            break
        start += max_words - overlap_words
    return chunks


def main():
    articles = []
    with open(config.ARTICLES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"Yuklandi: {len(articles)} maqola")

    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(
            article,
            max_words=config.CHUNK_MAX_WORDS,
            overlap_words=config.CHUNK_OVERLAP_WORDS,
        ))

    print(f"Jami chunk: {len(all_chunks)}")

    with open(config.CHUNKS_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"Saqlandi: {config.CHUNKS_PATH}")


if __name__ == '__main__':
    main()
