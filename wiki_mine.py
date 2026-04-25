#!/usr/bin/env python3
"""
Wikipedia hierarchical dataset miner for RAG projects.

Mines Wikipedia via its public API, preserving the category→subcategory→
article→section hierarchy.  Optionally generates simple QA pairs from
each harvested chunk.

Usage:
    python wiki_mine.py

All tunable knobs are in the HYPERPARAMETERS block below.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────

# Top-level Wikipedia category names to seed the crawl from.
# Eight well-separated domains chosen for breadth: gives the resulting
# corpus a real 8-way category structure (path B), which is the minimum
# needed for non-Euclidean geometry to demonstrate any norm/angle advantage.
SEED_CATEGORIES: list[str] = [
    "Biology",
    "Physics",
    "Mathematics",
    "Computer science",
    "History",
    "Philosophy",
    "Chemistry",
    "Economics",
]

# How many levels of subcategory to descend.
# 0 = only harvest articles directly in the seed categories.
# 1 = seed cats + their direct subcategories, etc.
MAX_DEPTH: int = 4

# Subcategories to follow per category node (breadth limit).
MAX_SUBCATS_PER_CAT: int = 5

# Articles to harvest per category node.
MAX_ARTICLES_PER_CAT: int = 6

# Per-seed cap on harvested articles.  Without this, a single rich seed
# (e.g. "Biology") can exhaust MAX_TOTAL_ARTICLES before later seeds run,
# producing a corpus dominated by 1-2 categories.
MAX_ARTICLES_PER_SEED: int = 100

# Hard cap on unique articles regardless of category structure.
MAX_TOTAL_ARTICLES: int = 1000

# How many named sections (after the intro) to keep per article.
MAX_SECTIONS_PER_ARTICLE: int = 6

# Sections / intros shorter than this (in characters) are discarded.
MIN_CHUNK_LENGTH: int = 150

# Maximum characters to store per chunk (keeps file size manageable).
MAX_CHUNK_LENGTH: int = 2000

# Where to write corpus.jsonl (and qa_pairs.jsonl if GENERATE_QA=True).
OUTPUT_DIR: str = "wiki_dataset"

# Set to True to also emit qa_pairs.jsonl.
GENERATE_QA: bool = True

# Wikipedia language edition.
LANGUAGE: str = "en"

# Seconds to sleep between API requests (be polite to Wikipedia).
REQUEST_DELAY_S: float = 0.2

# ─────────────────────────────────────────────────────────────────────────────

API_URL = f"https://{LANGUAGE}.wikipedia.org/w/api.php"
_HEADERS = {
    "User-Agent": (
        "WikiRAGMiner/1.0 (educational RAG dataset; "
        "https://github.com/wikimedia/mediawiki)"
    )
}


# ─── Wikipedia API helpers ────────────────────────────────────────────────────

def _api(params: dict) -> dict:
    params.setdefault("format", "json")
    time.sleep(REQUEST_DELAY_S)
    resp = requests.get(API_URL, params=params, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_subcategories(category: str, limit: int) -> list[str]:
    data = _api({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "subcat",
        "cmlimit": limit,
    })
    members = data.get("query", {}).get("categorymembers", [])
    return [m["title"].removeprefix("Category:") for m in members]


def get_articles_in_category(category: str, limit: int) -> list[str]:
    data = _api({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "page",
        "cmlimit": limit,
        "cmnamespace": 0,
    })
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def get_intro_text(title: str) -> str:
    """Return the lead section of an article as plain text."""
    data = _api({
        "action": "query",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": 8,
        "titles": title,
    })
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    return page.get("extract", "").strip()


def get_sections_metadata(title: str) -> list[dict]:
    """Return section list [{index, line, level, anchor}, ...]."""
    data = _api({
        "action": "parse",
        "page": title,
        "prop": "sections",
        "disabletoc": "1",
    })
    return data.get("parse", {}).get("sections", [])


def get_section_wikitext(title: str, section_index: int) -> str:
    data = _api({
        "action": "parse",
        "page": title,
        "section": section_index,
        "prop": "wikitext",
        "disablelimitreport": "1",
    })
    return data.get("parse", {}).get("wikitext", {}).get("*", "")


# ─── Text cleaning ────────────────────────────────────────────────────────────

_TEMPLATE_RE   = re.compile(r"\{\{[^{}]*\}\}")
_FILE_RE       = re.compile(r"\[\[(File|Image):[^\]]*\]\]", re.IGNORECASE)
_WIKILINK_RE   = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_HTML_RE       = re.compile(r"<[^>]+>")
_REF_BLOCK_RE  = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF_RE   = re.compile(r"<ref[^/]*/?>", re.IGNORECASE)
_BOLD_ITAL_RE  = re.compile(r"'{2,5}")
_HEADER_RE     = re.compile(r"^={2,6}.+?={2,6}\s*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\n{3,}")


def clean_wikitext(text: str) -> str:
    for _ in range(3):                      # nested templates (up to 3 deep)
        text = _TEMPLATE_RE.sub("", text)
    text = _REF_BLOCK_RE.sub("", text)
    text = _REF_SELF_RE.sub("", text)
    text = _FILE_RE.sub("", text)
    text = _WIKILINK_RE.sub(r"\1", text)
    text = _HTML_RE.sub("", text)
    text = _BOLD_ITAL_RE.sub("", text)
    text = _HEADER_RE.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


# ─── QA generation (heuristic, no LLM required) ───────────────────────────────

_SECTION_QUESTION_TEMPLATES: dict[str, str] = {
    "history":        "What is the history of {article}?",
    "background":     "What is the background of {article}?",
    "overview":       "What is an overview of {article}?",
    "applications":   "What are the applications of {article}?",
    "uses":           "What are the uses of {article}?",
    "causes":         "What are the causes of {article}?",
    "effects":        "What are the effects of {article}?",
    "impact":         "What is the impact of {article}?",
    "consequences":   "What are the consequences of {article}?",
    "definition":     "How is {article} defined?",
    "types":          "What are the types of {article}?",
    "classification": "How is {article} classified?",
    "structure":      "What is the structure of {article}?",
    "mechanism":      "What is the mechanism of {article}?",
    "treatment":      "How is {article} treated?",
    "criticism":      "What are the criticisms of {article}?",
    "controversy":    "What controversies surround {article}?",
}


def make_question(article_title: str, section_title: str) -> str:
    key = section_title.lower().strip()
    template = _SECTION_QUESTION_TEMPLATES.get(key)
    if template:
        return template.format(article=article_title)
    return f"What does Wikipedia say about the {section_title} of {article_title}?"


def make_hierarchy_context_question(
    hierarchy: list[str], article_title: str, section_title: str
) -> str:
    """Embed the hierarchy path into the question so it explicitly requires
    hierarchical context to retrieve the right answer."""
    if len(hierarchy) >= 2:
        path = " > ".join(hierarchy[-2:])
    elif len(hierarchy) == 1:
        path = hierarchy[0]
    else:
        return make_question(article_title, section_title)
    return f"Under the topic hierarchy '{path}', what does '{article_title}' cover in its '{section_title}' section?"


def first_sentences(text: str, n: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:n]).strip()


def _find_grounding_sentence(text: str, keyword: str) -> str:
    """Return the first sentence(s) in text that mention keyword.
    Falls back to the first sentence if none match."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kw_lower = keyword.lower()
    matching = [s for s in sentences if kw_lower in s.lower()]
    if matching:
        return " ".join(matching[:2]).strip()
    return sentences[0].strip() if sentences else ""


# ─── ID utilities ─────────────────────────────────────────────────────────────

def make_id(*parts: str) -> str:
    key = ":".join(parts)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def wiki_url(title: str, anchor: str = "") -> str:
    slug = title.replace(" ", "_")
    base = f"https://{LANGUAGE}.wikipedia.org/wiki/{slug}"
    return f"{base}#{anchor}" if anchor else base


# ─── Core crawler ─────────────────────────────────────────────────────────────

def harvest_article(
    title: str,
    hierarchy: list[str],
    corpus: list[dict],
    qa_pairs: list[dict],
) -> None:
    art_hierarchy = hierarchy + [title]

    # ── Lead / intro section ──
    try:
        intro = get_intro_text(title)
    except Exception as e:
        print(f"    ! intro error for '{title}': {e}")
        intro = ""

    art_id = make_id("intro", title)
    if intro and len(intro) >= MIN_CHUNK_LENGTH:
        corpus.append({
            "id": art_id,
            "type": "article_intro",
            "title": title,
            "section": "Introduction",
            "section_level": 0,
            "content": intro[:MAX_CHUNK_LENGTH],
            "hierarchy": art_hierarchy,
            "level": len(art_hierarchy),
            "url": wiki_url(title),
        })
        if GENERATE_QA:
            answer = first_sentences(intro)
            if answer:
                qa_pairs.append({
                    "id": make_id("qa", title, "intro"),
                    "question": f"What is {title}?",
                    "answer": answer,
                    "source_id": art_id,
                    "hierarchy": art_hierarchy,
                    "hierarchy_level": len(art_hierarchy),
                    "qa_type": "content",
                })

            # hierarchy QA: find sentence(s) in intro that mention the parent
            # category, so the answer is grounded in actual chunk content
            if hierarchy:
                parent = hierarchy[-1]
                grounding = _find_grounding_sentence(intro, parent)
                if grounding:
                    qa_pairs.append({
                        "id": make_id("qa-hier", title, "parent"),
                        "question": f"What field or category does '{title}' belong to, and how is it related?",
                        "answer": grounding,
                        "source_id": art_id,
                        "hierarchy": art_hierarchy,
                        "hierarchy_level": len(art_hierarchy),
                        "qa_type": "parent_category",
                    })

    # ── Named sections ──
    try:
        sections_meta = get_sections_metadata(title)
    except Exception as e:
        print(f"    ! sections error for '{title}': {e}")
        sections_meta = []

    kept = 0
    for sec in sections_meta:
        if kept >= MAX_SECTIONS_PER_ARTICLE:
            break
        if not sec.get("index", "").strip():
            continue
        sec_idx   = int(sec["index"])
        sec_title = sec["line"]
        sec_level = int(sec["level"])
        anchor    = sec.get("anchor", "")

        # Skip deep sub-sub-sections to keep hierarchy clean
        if sec_level > 3:
            continue

        try:
            raw  = get_section_wikitext(title, sec_idx)
            text = clean_wikitext(raw)
        except Exception as e:
            print(f"    ! section error '{sec_title}': {e}")
            continue

        if len(text) < MIN_CHUNK_LENGTH:
            continue

        sec_hierarchy = art_hierarchy + [sec_title]
        sec_id = make_id("section", title, sec_title)
        corpus.append({
            "id": sec_id,
            "type": "section",
            "title": title,
            "section": sec_title,
            "section_level": sec_level,
            "content": text[:MAX_CHUNK_LENGTH],
            "hierarchy": sec_hierarchy,
            "level": len(sec_hierarchy),
            "url": wiki_url(title, anchor),
        })
        if GENERATE_QA:
            answer = first_sentences(text)
            if answer:
                # content QA (original)
                qa_pairs.append({
                    "id": make_id("qa", title, sec_title),
                    "question": make_question(title, sec_title),
                    "answer": answer,
                    "source_id": sec_id,
                    "hierarchy": sec_hierarchy,
                    "hierarchy_level": len(sec_hierarchy),
                    "qa_type": "content",
                })
                # hierarchy-context QA: question embeds the category path
                qa_pairs.append({
                    "id": make_id("qa-ctx", title, sec_title),
                    "question": make_hierarchy_context_question(hierarchy, title, sec_title),
                    "answer": answer,
                    "source_id": sec_id,
                    "hierarchy": sec_hierarchy,
                    "hierarchy_level": len(sec_hierarchy),
                    "qa_type": "hierarchy_context",
                })
        kept += 1


def crawl(
    category: str,
    depth: int,
    hierarchy: list[str],
    corpus: list[dict],
    qa_pairs: list[dict],
    seen_articles: set[str],
    seed_cap: int,
) -> None:
    """seed_cap is the |seen_articles| value at which this crawl must stop —
    enforces both the global MAX_TOTAL_ARTICLES and the per-seed budget."""
    if len(seen_articles) >= seed_cap:
        return

    cat_hierarchy = hierarchy + [category]
    indent = "  " * depth

    # ── Articles in this category ──
    try:
        article_titles = get_articles_in_category(category, MAX_ARTICLES_PER_CAT)
    except Exception as e:
        print(f"{indent}! category error '{category}': {e}")
        article_titles = []

    for title in article_titles:
        if len(seen_articles) >= seed_cap:
            break
        if title in seen_articles:
            continue
        seen_articles.add(title)
        print(f"{indent}  article: {title}  (total={len(seen_articles)})")
        harvest_article(title, cat_hierarchy, corpus, qa_pairs)


    # ── Recurse into subcategories ──
    if depth < MAX_DEPTH:
        try:
            subcats = get_subcategories(category, MAX_SUBCATS_PER_CAT)
        except Exception as e:
            print(f"{indent}! subcat error '{category}': {e}")
            subcats = []

        for subcat in subcats:
            if len(seen_articles) >= seed_cap:
                return
            print(f"{indent}→ subcat: {subcat}")
            crawl(subcat, depth + 1, cat_hierarchy, corpus, qa_pairs,
                  seen_articles, seed_cap)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    corpus: list[dict]    = []
    qa_pairs: list[dict]  = []
    seen_articles: set[str] = set()

    for seed in SEED_CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Seeding: {seed}  (already harvested: {len(seen_articles)})")
        print(f"{'='*60}")
        # Per-seed budget: stop this seed once it has added MAX_ARTICLES_PER_SEED
        # new articles (or hits the global cap, whichever comes first).
        seed_cap = min(
            len(seen_articles) + MAX_ARTICLES_PER_SEED,
            MAX_TOTAL_ARTICLES,
        )
        crawl(
            category=seed,
            depth=0,
            hierarchy=[],
            corpus=corpus,
            qa_pairs=qa_pairs,
            seen_articles=seen_articles,
            seed_cap=seed_cap,
        )

    # ── Write outputs ──
    corpus_path = out / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"\nCorpus  : {len(corpus):,} chunks  →  {corpus_path}")

    if GENERATE_QA:
        qa_path = out / "qa_pairs.jsonl"
        with open(qa_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"QA pairs: {len(qa_pairs):,} pairs   →  {qa_path}")

        type_counts: dict[str, int] = {}
        for qa in qa_pairs:
            t = qa.get("qa_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print("QA pairs by type:")
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<22}: {n:,}")

    # ── Summary by hierarchy level ──
    level_counts: dict[int, int] = {}
    for doc in corpus:
        lvl = doc["level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    print("\nCorpus breakdown by hierarchy level:")
    for lvl in sorted(level_counts):
        bar = "█" * min(level_counts[lvl], 40)
        print(f"  Level {lvl:2d}: {level_counts[lvl]:4d}  {bar}")

    print(f"\nUnique articles crawled: {len(seen_articles)}")


if __name__ == "__main__":
    main()
