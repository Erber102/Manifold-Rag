#!/usr/bin/env python3
"""
Hard QA pair generator for the wiki dataset.

Generates two types of harder QA pairs on top of the existing corpus:

  1. hierarchy_disambiguation (programmatic, free)
     Finds section titles that appear in multiple hierarchy branches.
     The question embeds the full hierarchy path, making same-titled
     chunks from other branches natural hard negatives during retrieval.

  2. paraphrase (Claude API, optional)
     Asks Claude to write a question using DIFFERENT vocabulary than
     the source chunk, so keyword-matching baselines cannot cheat.

Strategy for BEIR output:
  - train qrels : easy QA pairs  (model learns basic retrieval signal)
  - test  qrels : hard QA pairs  (meaningful evaluation)

Usage:
    python generate_hard_qa.py

Set GENERATE_PARAPHRASE = True and export ANTHROPIC_API_KEY to enable
paraphrase generation.
"""

import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────

_WIKI_API     = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "WikiRAGMiner/1.0 (educational)"}
_CAT_API_DELAY    = 0.4   # seconds between category API calls
MIN_CAT_TEXT_LEN  = 80    # skip categories with too little description
MAX_CAT_TEXT_LEN  = 1500  # truncate very long category descriptions

CORPUS_PATH  = "wiki_dataset/corpus.jsonl"
EASY_QA_PATH = "wiki_dataset/qa_pairs.jsonl"
HARD_QA_OUT  = "wiki_dataset/qa_pairs_hard.jsonl"
BEIR_OUT_DIR = "data/wiki"

# Set True + export ANTHROPIC_API_KEY to generate paraphrase QA via Claude
GENERATE_PARAPHRASE = False
PARAPHRASE_MODEL    = "claude-haiku-4-5-20251001"
API_DELAY_S         = 0.2     # seconds between API calls

# A section title must appear in at least this many different top-level
# hierarchy branches to qualify for disambiguation QA
MIN_BRANCHES = 2

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Fraction of easy QA to mix into train for training stability (0 = none)
EASY_QA_MIX_RATIO = 0.2

# Per-type test fraction.  Heavily weight test toward hierarchy-navigation QA
# (generalize, sibling, navigation) — these are tasks where Poincaré geometry
# is structurally advantaged.  Disambiguation goes mostly to train.
TYPE_TEST_RATIOS = {
    "generalize":               0.70,   # main test: section content → category doc
    "sibling":                  0.00,   # noise — random sibling has no signal in question; train-only
    "hierarchy_navigation":     0.50,   # article → category (existing)
    "hierarchy_disambiguation": 0.10,   # mostly train signal
    "paraphrase":               0.20,
}
# Cap test size per type to keep evaluation tractable.
MAX_TEST_PER_TYPE = {
    "generalize":               600,
    "sibling":                  400,
    "hierarchy_navigation":     200,
    "hierarchy_disambiguation": 200,
    "paraphrase":               300,
}

# Caps for generation (avoid runaway QA counts on common categories)
GEN_MAX_PER_CATEGORY     = 400   # generalize QA per category
SIB_MAX_PER_SECTION      = 2     # sibling QA per source section

# ─────────────────────────────────────────────────────────────────────────────


def make_id(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:12]


def first_sentences(text: str, n: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:n]).strip()


# ── Part 1: Hierarchy disambiguation (no API needed) ─────────────────────────

def generate_disambiguation_qa(corpus: list[dict]) -> list[dict]:
    """
    Example:
      corpus has "History" sections under both "Artificial intelligence"
      and "World War II".  For each, we generate:
        Q: "In the context of 'AI > Machine learning > Neural network',
            what does the 'History' section cover?"
        A: first sentences of that specific chunk

    The other "History" chunks are natural hard negatives — same section
    title, wrong hierarchy branch.
    """
    # Group section chunks by normalised section title
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in corpus:
        if doc.get("type") == "section" and doc.get("section"):
            key = doc["section"].lower().strip()
            groups[key].append(doc)

    qa_pairs = []
    for section_title, chunks in groups.items():
        # Only keep groups that span multiple top-level categories
        top_cats = {
            c["hierarchy"][0]
            for c in chunks
            if c.get("hierarchy")
        }
        if len(top_cats) < MIN_BRANCHES:
            continue

        for chunk in chunks:
            hier = chunk.get("hierarchy", [])
            # Exclude the article title (last hierarchy element) from the path.
            # Including it would let SBERT keyword-match the corpus doc title
            # ("Article - Section") and trivially retrieve it — defeating the
            # purpose of disambiguation.
            parent_hier = hier[:-1] if len(hier) > 1 else hier
            path = " > ".join(parent_hier[-2:]) if parent_hier else "general"
            answer = first_sentences(chunk["content"])
            if not answer:
                continue

            question = (
                f"In the context of '{path}', "
                f"what does the '{chunk['section']}' section cover?"
            )
            qa_pairs.append({
                "id":              make_id("qa-disambig", chunk["id"]),
                "question":        question,
                "answer":          answer,
                "source_id":       chunk["id"],
                "hierarchy":       hier,
                "hierarchy_level": len(hier),
                "qa_type":         "hierarchy_disambiguation",
            })

    return qa_pairs


# ── Part 2: Paraphrase QA via Claude API ──────────────────────────────────────

def generate_paraphrase_qa(corpus: list[dict]) -> list[dict]:
    try:
        import anthropic
    except ImportError:
        print("  anthropic not installed — run: pip install anthropic")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — skipping paraphrase QA.")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Generate only for section chunks (richer content)
    targets = [d for d in corpus if d.get("type") == "section"]
    print(f"  Calling API for {len(targets)} section chunks …")

    qa_pairs = []
    for i, chunk in enumerate(targets):
        hierarchy_str = " > ".join(chunk.get("hierarchy", []))
        content_preview = chunk["content"][:600]

        prompt = (
            "You are creating retrieval evaluation questions for a Wikipedia dataset.\n\n"
            f"Chunk hierarchy: {hierarchy_str}\n"
            f"Article: {chunk['title']}\n"
            f"Section: {chunk.get('section', '')}\n\n"
            f"Content:\n\"\"\"{content_preview}\"\"\"\n\n"
            "Write ONE question that:\n"
            "1. Can be answered using only the content above\n"
            "2. Uses DIFFERENT words than the content (paraphrase — do not copy phrases)\n"
            f"3. Naturally reflects the topic context: {hierarchy_str}\n"
            "4. Sounds like a genuine user question\n\n"
            "Output only the question, nothing else."
        )

        try:
            time.sleep(API_DELAY_S)
            msg = client.messages.create(
                model=PARAPHRASE_MODEL,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            question = msg.content[0].text.strip()
            answer   = first_sentences(chunk["content"])

            if question and answer:
                qa_pairs.append({
                    "id":              make_id("qa-para", chunk["id"]),
                    "question":        question,
                    "answer":          answer,
                    "source_id":       chunk["id"],
                    "hierarchy":       chunk.get("hierarchy", []),
                    "hierarchy_level": chunk.get("level", 0),
                    "qa_type":         "paraphrase",
                })

        except Exception as e:
            print(f"  API error ({chunk['id']}): {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(targets)} done")

    return qa_pairs


# ── Part 3: Category documents + hierarchy navigation QA ─────────────────────

def fetch_category_text(category: str) -> str:
    """
    Return the intro text of the Wikipedia *article* with the same name as
    this category (NOT the Category: page, which has no prose).
    The article's lead paragraph is exactly the abstract description we want
    to use as a level-1 'category doc' in the hierarchy.
    """
    time.sleep(_CAT_API_DELAY)
    try:
        resp = requests.get(
            _WIKI_API,
            params={
                "action":      "query",
                "prop":        "extracts",
                "exintro":     "1",
                "explaintext": "1",
                "exsentences": 8,
                "titles":      category,   # plain article title, no "Category:"
                "redirects":   "1",
                "format":      "json",
            },
            headers=_WIKI_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        if not pages:
            return ""
        page = next(iter(pages.values()))
        # Wikipedia returns pageid=-1 (or "missing" key) for nonexistent pages
        if page.get("missing") is not None or "extract" not in page:
            return ""
        text = page["extract"].strip()
        return text[:MAX_CAT_TEXT_LEN]
    except Exception as e:
        print(f"  fetch_category_text({category!r}): {e}")
        return ""


def generate_category_docs_and_nav_qa(
    corpus: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    1. Collect every unique top-level category that appears in corpus hierarchy.
    2. Fetch a short Wikipedia description for each Category page.
    3. Create a corpus document of type='category' for each (level=1, near
       Poincaré ball origin ↔ most abstract).
    4. For each article chunk, create a 'hierarchy_navigation' QA pair:
         Q: "What broader Wikipedia category does the article '{title}' belong to?"
         A: first sentences of the category document
       The positive doc is the category doc; nearby article sections are
       natural hard negatives (same words, wrong abstraction level).
    """
    # Collect unique top-level categories and the article titles under each
    cat_to_articles: dict[str, set[str]] = defaultdict(set)
    for doc in corpus:
        hier = doc.get("hierarchy", [])
        if hier:
            cat_to_articles[hier[0]].add(doc.get("title", ""))

    print(f"  Found {len(cat_to_articles)} unique top-level categories")

    cat_docs: list[dict] = []
    nav_qa: list[dict] = []

    for cat_name, article_titles in cat_to_articles.items():
        text = fetch_category_text(cat_name)
        if len(text) < MIN_CAT_TEXT_LEN:
            print(f"  Skipping '{cat_name}' (text too short or missing)")
            continue

        cat_id = make_id("cat", cat_name)
        cat_docs.append({
            "id":       cat_id,
            "title":    cat_name,
            "section":  "",
            "content":  text,
            "type":     "category",
            "level":    1,
            "hierarchy": [cat_name],
            "url":      f"https://en.wikipedia.org/wiki/Category:{cat_name.replace(' ', '_')}",
        })

        answer = first_sentences(text)
        for article in sorted(article_titles):
            question = (
                f"What broader Wikipedia category does the article '{article}' belong to, "
                f"and what does that category cover?"
            )
            nav_qa.append({
                "id":              make_id("qa-nav", cat_name, article),
                "question":        question,
                "answer":          answer,
                "source_id":       cat_id,
                "hierarchy":       [cat_name],
                "hierarchy_level": 1,
                "qa_type":         "hierarchy_navigation",
            })

        print(f"  '{cat_name}': {len(article_titles)} articles → {len(article_titles)} nav QA")

    return cat_docs, nav_qa


# ── Part 4: Hierarchy-navigation QA (Path A — Poincaré-favorable) ────────────

def _clean_snippet(text: str, taboo: list[str], max_len: int = 220) -> str:
    """
    Take 2nd–3rd sentences of `text` (skip the first, which often re-states
    the article title) and replace any token in `taboo` with '[topic]'.
    This prevents SBERT from finding the target doc by keyword-matching the
    article / category / section title.
    """
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sents) >= 3:
        snippet = " ".join(sents[1:4])
    elif len(sents) == 2:
        snippet = sents[1]
    else:
        snippet = sents[0] if sents else ""
    for word in taboo:
        if not word or len(word) < 3:
            continue
        snippet = re.sub(rf"\b{re.escape(word)}\b", "[topic]",
                         snippet, flags=re.IGNORECASE)
    return snippet[:max_len].strip()


def generate_generalize_qa(corpus: list[dict]) -> list[dict]:
    """
    Q: "Which broad subject area covers the following discussion: <content>?"
    A: the level-1 category document for this section's branch.

    Tests upward navigation (specific → abstract).  Poincaré geometry is
    structurally advantaged: the model can place category docs near the
    origin and sections near the boundary, making this query a pure
    norm-difference + angular-match problem.
    """
    cat_by_name = {d["title"]: d for d in corpus if d.get("type") == "category"}
    if not cat_by_name:
        print("  No category docs in corpus — skipping generalize QA.")
        return []

    sections_by_cat: dict[str, list[dict]] = defaultdict(list)
    for doc in corpus:
        if doc.get("type") != "section":
            continue
        hier = doc.get("hierarchy", [])
        if hier and hier[0] in cat_by_name:
            sections_by_cat[hier[0]].append(doc)

    qa_pairs: list[dict] = []
    rng = random.Random(RANDOM_SEED)
    for cat_name, sections in sections_by_cat.items():
        cat_doc = cat_by_name[cat_name]
        rng.shuffle(sections)
        for sec in sections[:GEN_MAX_PER_CATEGORY]:
            hier = sec.get("hierarchy", [])
            taboo = [cat_name, sec.get("section", ""), sec.get("title", "")] + hier
            snippet = _clean_snippet(sec["content"], taboo, max_len=220)
            if len(snippet) < 40:
                continue

            question = (
                f"Which broad subject area covers the following discussion: "
                f"\"{snippet}\"?"
            )
            qa_pairs.append({
                "id":              make_id("qa-gen", sec["id"]),
                "question":        question,
                "answer":          first_sentences(cat_doc["content"]),
                "source_id":       cat_doc["id"],
                "hierarchy":       cat_doc.get("hierarchy", [cat_name]),
                "hierarchy_level": 1,
                "qa_type":         "generalize",
            })
    return qa_pairs


def generate_sibling_qa(corpus: list[dict]) -> list[dict]:
    """
    Q: "In the broad area of '<category>', what other topic relates to: <content>?"
    A: a different section in the same top-level category but a DIFFERENT article.

    Tests lateral (same-level) navigation.  The query's ground-truth doc is
    intentionally a sibling, not the source — the model must learn that
    'same category, different sub-tree' is closer than 'different category,
    similar wording'.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for doc in corpus:
        if doc.get("type") != "section":
            continue
        hier = doc.get("hierarchy", [])
        if hier:
            by_cat[hier[0]].append(doc)

    qa_pairs: list[dict] = []
    rng = random.Random(RANDOM_SEED + 1)
    for cat_name, sections in by_cat.items():
        if len(sections) < 4:
            continue
        for sec in sections:
            article = sec.get("title", "")
            siblings = [s for s in sections if s.get("title", "") != article]
            if not siblings:
                continue
            chosen = rng.sample(siblings, min(SIB_MAX_PER_SECTION, len(siblings)))

            hier = sec.get("hierarchy", [])
            taboo = [cat_name, sec.get("section", ""), article] + hier
            snippet = _clean_snippet(sec["content"], taboo, max_len=200)
            if len(snippet) < 40:
                continue

            for sib in chosen:
                question = (
                    f"In the broad area of '{cat_name}', what other topic is "
                    f"studied that relates to: \"{snippet}\"?"
                )
                qa_pairs.append({
                    "id":              make_id("qa-sib", sec["id"], sib["id"]),
                    "question":        question,
                    "answer":          first_sentences(sib["content"]),
                    "source_id":       sib["id"],
                    "hierarchy":       sib.get("hierarchy", []),
                    "hierarchy_level": len(sib.get("hierarchy", [])),
                    "qa_type":         "sibling",
                })
    return qa_pairs


# ── BEIR conversion ───────────────────────────────────────────────────────────

def write_beir(
    corpus:     list[dict],
    easy_qa:    list[dict],
    hard_qa:    list[dict],
    out_dir:    str,
    extra_docs: list[dict] | None = None,
) -> None:
    """Write BEIR files.  extra_docs (e.g. category docs) are appended to the
    corpus file but NOT used to filter QA pairs (their IDs are added
    separately via nav_qa source_ids already pointing to them)."""
    out = Path(out_dir)
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    all_corpus = corpus + (extra_docs or [])
    corpus_ids = {d["id"] for d in all_corpus}

    # corpus.jsonl — original chunks + category docs
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in all_corpus:
            section = doc.get("section", "")
            title = (
                doc["title"]
                if (not section or section == "Introduction")
                else f"{doc['title']} - {section}"
            )
            f.write(json.dumps({
                "_id":      doc["id"],
                "title":    title,
                "text":     doc["content"],
                "metadata": {
                    "hierarchy": doc.get("hierarchy", []),
                    "level":     doc.get("level", 0),
                    "type":      doc.get("type", ""),
                    "url":       doc.get("url", ""),
                },
            }, ensure_ascii=False) + "\n")

    rng = random.Random(RANDOM_SEED)

    # Split hard QA per-type (Path A): heavily weight test toward
    # hierarchy-navigation tasks where Poincaré geometry is advantaged.
    valid_hard = [q for q in hard_qa if q.get("source_id") in corpus_ids]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in valid_hard:
        by_type[q.get("qa_type", "unknown")].append(q)

    train_qas: list[dict] = []
    test_qas:  list[dict] = []
    split_breakdown: dict[str, tuple[int, int]] = {}
    for qa_type, items in by_type.items():
        rng.shuffle(items)
        test_ratio = TYPE_TEST_RATIOS.get(qa_type, 0.20)
        n_test = int(len(items) * test_ratio)
        # Cap test count per type
        n_test = min(n_test, MAX_TEST_PER_TYPE.get(qa_type, 300))
        test_qas.extend(items[:n_test])
        train_qas.extend(items[n_test:])
        split_breakdown[qa_type] = (len(items) - n_test, n_test)

    # Optionally mix a fraction of easy QA into train for stability
    if EASY_QA_MIX_RATIO > 0:
        valid_easy = [q for q in easy_qa if q.get("source_id") in corpus_ids]
        rng.shuffle(valid_easy)
        n_easy = int(len(valid_easy) * EASY_QA_MIX_RATIO)
        train_qas = train_qas + valid_easy[:n_easy]

    rng.shuffle(train_qas)
    rng.shuffle(test_qas)

    for qas, name in [(train_qas, "train"), (test_qas, "test")]:
        with open(out / "qrels" / f"{name}.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for qa in qas:
                f.write(f"{qa['id']}\t{qa['source_id']}\t1\n")

    # queries.jsonl: train + test combined
    with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
        for qa in train_qas + test_qas:
            f.write(json.dumps({
                "_id":      qa["id"],
                "text":     qa["question"],
                "metadata": {"qa_type": qa.get("qa_type", "")},
            }, ensure_ascii=False) + "\n")

    # Summary
    type_counts: dict[str, int] = {}
    for qa in test_qas:
        t = qa.get("qa_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nBEIR output  →  {out_dir}")
    print(f"  corpus     :  {len(all_corpus)} chunks ({len(corpus)} article + {len(extra_docs or [])} category)")
    print(f"  train qrels:  {len(train_qas)} QA")
    print(f"  test  qrels:  {len(test_qas)} QA")
    print("  per-type split (train / test):")
    for t, (n_tr, n_te) in sorted(split_breakdown.items(), key=lambda x: -x[1][1]):
        print(f"    {t:<30}:  {n_tr:>5} / {n_te:>4}")
    print("  test QA by type:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<30}:  {n}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Load corpus
    corpus: list[dict] = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"Corpus loaded: {len(corpus)} chunks")

    # Load easy QA
    easy_qa: list[dict] = []
    with open(EASY_QA_PATH, encoding="utf-8") as f:
        for line in f:
            easy_qa.append(json.loads(line))
    print(f"Easy QA loaded: {len(easy_qa)} pairs")

    # ── Generate hard QA ──
    print("\n── Hierarchy disambiguation QA (programmatic) ──")
    disambig_qa = generate_disambiguation_qa(corpus)
    print(f"Generated: {len(disambig_qa)} pairs")

    paraphrase_qa: list[dict] = []
    if GENERATE_PARAPHRASE:
        print("\n── Paraphrase QA (Claude API) ──")
        paraphrase_qa = generate_paraphrase_qa(corpus)
        print(f"Generated: {len(paraphrase_qa)} pairs")

    print("\n── Category documents + hierarchy navigation QA (Wikipedia API) ──")
    cat_docs, nav_qa = generate_category_docs_and_nav_qa(corpus)
    print(f"Generated: {len(cat_docs)} category docs, {len(nav_qa)} navigation QA pairs")

    # Build the merged corpus once so generalize/sibling QA can reference
    # the freshly-fetched category docs.
    extended_corpus = corpus + cat_docs

    print("\n── Generalize QA (section content → category doc) ──")
    generalize_qa = generate_generalize_qa(extended_corpus)
    print(f"Generated: {len(generalize_qa)} pairs")

    print("\n── Sibling QA (section → cross-article section in same category) ──")
    sibling_qa = generate_sibling_qa(extended_corpus)
    print(f"Generated: {len(sibling_qa)} pairs")

    hard_qa = disambig_qa + paraphrase_qa + nav_qa + generalize_qa + sibling_qa

    # Save hard QA
    with open(HARD_QA_OUT, "w", encoding="utf-8") as f:
        for qa in hard_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"\nHard QA saved → {HARD_QA_OUT}  ({len(hard_qa)} pairs total)")

    # Update BEIR files (append category docs to corpus)
    write_beir(corpus, easy_qa, hard_qa, BEIR_OUT_DIR, extra_docs=cat_docs)


if __name__ == "__main__":
    main()
