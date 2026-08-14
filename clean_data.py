"""4.3 — WikiExtractor chiqishini yuklash, tozalash va namuna tanlash."""
import json
import random
import re
from pathlib import Path

import config


def clean_text(text: str) -> str:
    """Vikipediya maxsus belgilarini tozalash (qoldiq wiki-markup, HTML)."""
    text = re.sub(r'\[\[([^|\]]+)\|?[^\]]*\]\]', r'\1', text)  # [[Havola|Matn]]
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)  # {{Shablon}}
    text = re.sub(r'<[^>]+>', '', text)  # HTML teglar
    text = re.sub(r'\s+', ' ', text).strip()  # ortiqcha bo'sh joylar
    return text


def load_wiki_articles(wiki_dir: Path):
    """WikiExtractor --json chiqishidan maqolalar generatori."""
    for sub in sorted(Path(wiki_dir).rglob('wiki_*')):
        with open(sub, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = clean_text(doc.get('text', ''))
                if len(text.split()) <= config.MIN_WORDS:
                    continue  # stub maqolalarni o'tkazish
                yield {
                    'id': doc['id'],
                    'title': doc['title'],
                    'text': text,
                }


def main():
    random.seed(config.RANDOM_SEED)
    print(f"WikiExtractor chiqishi o'qilmoqda: {config.WIKI_OUTPUT_DIR}")

    # Reservoir sampling (Algorithm R): butun 545k+ maqolani xotiraga
    # yuklamasdan, N_ARTICLES ta tasodifiy, takrorlanuvchi (seed bilan)
    # namuna tanlanadi — RAM cheklangan mashinada xavfsiz.
    reservoir = []
    total_qualifying = 0
    for article in load_wiki_articles(config.WIKI_OUTPUT_DIR):
        total_qualifying += 1
        if config.N_ARTICLES is None:
            reservoir.append(article)
            continue
        if len(reservoir) < config.N_ARTICLES:
            reservoir.append(article)
        else:
            j = random.randint(0, total_qualifying - 1)
            if j < config.N_ARTICLES:
                reservoir[j] = article
        if total_qualifying % 50000 == 0:
            print(f"  ... {total_qualifying} maqola ko'rib chiqildi")

    print(f"MIN_WORDS={config.MIN_WORDS} dan katta maqolalar: {total_qualifying}")
    print(f"Tanlangan namuna: {len(reservoir)} (seed={config.RANDOM_SEED})")

    with open(config.ARTICLES_PATH, 'w', encoding='utf-8') as f:
        for a in reservoir:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')

    print(f"Saqlandi: {config.ARTICLES_PATH} ({len(reservoir)} maqola)")


if __name__ == '__main__':
    main()
