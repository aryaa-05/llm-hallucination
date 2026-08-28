"""
Stage 2a: Web Evidence + NLI
Primary: Wikipedia API (reliable, entity-focused)
Fallback: DuckDuckGo (broad web search)
Scoring: Bidirectional DeBERTa-v3 NLI + structured contradiction detection
"""

import time
import torch
import re
import requests
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from pipeline.claim_decomposer import call_llm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Models ──────────────────────────────────────────────────
NLI_MODEL_REPO = "cross-encoder/nli-deberta-v3-base"
EMB_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_nli_tokenizer = None
_nli_model     = None
_NLI_LABEL_MAP = {}
_emb_model     = None

def _load_models_internal():
    global _nli_tokenizer, _nli_model, _NLI_LABEL_MAP, _emb_model
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_REPO)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_REPO
        ).to(device)
        _nli_model.eval()
        _NLI_LABEL_MAP = {
            v.lower(): int(k)
            for k, v in _nli_model.config.id2label.items()
        }
    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_REPO, device=device)

def load_models():
    _load_models_internal()


# ── Entity Extraction (regex-based, no spaCy needed) ─────────
def _extract_entities(text: str) -> dict:
    """Extract named entities, dates, numbers, and locations from text."""
    # Proper nouns: sequences of capitalized words
    proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    # Filter out sentence starters (rough heuristic: keep multi-word or mid-sentence)
    proper_nouns = [pn for pn in proper_nouns if len(pn.split()) >= 2 or
                    not text.startswith(pn)]

    # Years (4-digit numbers)
    years = re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', text)

    # Numbers (excluding years)
    numbers = [n for n in re.findall(r'\b\d+\b', text) if len(n) != 4]

    # Main entity: longest proper noun (usually the subject)
    main_entity = max(proper_nouns, key=len) if proper_nouns else ""

    return {
        "proper_nouns": proper_nouns,
        "main_entity": main_entity,
        "years": years,
        "numbers": numbers,
    }


# ── Wikipedia Retrieval (primary) ────────────────────────────
def _search_wikipedia(query: str) -> list[str]:
    """Fetch Wikipedia summary sentences for a query."""
    if not query:
        return []

    sentences = []
    try:
        # Try the Wikipedia REST API (works best if query is exactly an entity)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "HalluciDetect/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            if extract:
                # Split into sentences for focused NLI
                sents = re.split(r'(?<=[.!?])\s+', extract)
                sentences.extend([s.strip() for s in sents if len(s.strip()) > 20])
    except Exception as e:
        print(f"  Wikipedia API error: {e}")

    # Also try search endpoint if direct lookup failed
    if not sentences:
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
            }
            resp = requests.get(search_url, params=params, timeout=8,
                              headers={"User-Agent": "HalluciDetect/1.0"})
            if resp.status_code == 200:
                results = resp.json().get("query", {}).get("search", [])
                for r in results:
                    # Clean HTML from snippets
                    snippet = re.sub(r'<[^>]+>', '', r.get("snippet", ""))
                    if snippet and len(snippet) > 20:
                        sentences.append(snippet)
                    # Also fetch the full summary for the top result
                    if not sentences:
                        title = r.get("title", "")
                        if title:
                            sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
                            sum_resp = requests.get(sum_url, timeout=8,
                                                   headers={"User-Agent": "HalluciDetect/1.0"})
                            if sum_resp.status_code == 200:
                                extract = sum_resp.json().get("extract", "")
                                sents = re.split(r'(?<=[.!?])\s+', extract)
                                sentences.extend([s.strip() for s in sents if len(s.strip()) > 20])
        except Exception as e:
            print(f"  Wikipedia search error: {e}")

    return sentences[:8]  # Cap at 8 evidence sentences


# ── DuckDuckGo Fallback ──────────────────────────────────────
def _search_ddg(query: str, max_results: int = 5) -> list[str]:
    """DuckDuckGo text search with retry — used as fallback."""
    from duckduckgo_search import DDGS
    snippets = []
    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            for r in results:
                body = r.get("body", "").strip()
                if body:
                    snippets.append(body)
            if snippets:
                return snippets
        except Exception as e:
            print(f"  DDG attempt {attempt+1} error: {e}")
        time.sleep(2)
    return snippets


# ── Relevance Filter (Semantic) ──────────────────────────────
def _relevance_filter(claim: str, snippets: list[str], top_k: int = 5) -> list[str]:
    """Semantic relevance filter using Sentence Transformers."""
    if not snippets:
        return []
    
    _load_models_internal()
    
    # Encode claim and snippets
    claim_emb = _emb_model.encode(claim, convert_to_tensor=True)
    snip_embs = _emb_model.encode(snippets, convert_to_tensor=True)
    
    # Compute cosine similarities
    cos_scores = util.cos_sim(claim_emb, snip_embs)[0]
    
    # Sort and filter
    scored = []
    for i, score in enumerate(cos_scores):
        if score > 0.35:  # Threshold for semantic relevance
            scored.append((float(score), snippets[i]))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]


# ── Bidirectional NLI ────────────────────────────────────────
def _nli_score(text_a: str, text_b: str) -> dict:
    """Single-direction NLI: P(text_a entails text_b)."""
    _load_models_internal()
    enc = _nli_tokenizer(
        text_a, text_b,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)

    with torch.no_grad():
        logits = _nli_model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)[0]

    return {
        "entailment"    : float(probs[_NLI_LABEL_MAP.get("entailment", 1)]),
        "neutral"       : float(probs[_NLI_LABEL_MAP.get("neutral", 2)]),
        "contradiction" : float(probs[_NLI_LABEL_MAP.get("contradiction", 0)]),
    }

def _bidirectional_nli(evidence: str, claim: str) -> dict:
    """
    Compute NLI in BOTH directions and take the max.
    This fixes asymmetric entailment failures where extra info causes neutral.
    """
    forward  = _nli_score(evidence, claim)   # evidence → claim
    backward = _nli_score(claim, evidence)   # claim → evidence

    return {
        "entailment"    : round(max(forward["entailment"],    backward["entailment"]),    3),
        "neutral"       : round(min(forward["neutral"],       backward["neutral"]),       3),
        "contradiction" : round(max(forward["contradiction"], backward["contradiction"]), 3),
    }


# ── Structured Contradiction Detection ───────────────────────
def _structured_contradiction(claim: str, evidence: str) -> float:
    """
    Detect contradictions via date/number slot comparison.
    Returns a contradiction boost score [0, 1].
    NOTE: Only fires when the evidence is clearly about the SAME subject
    but gives a different year — not when evidence simply discusses other dates.
    """
    boost = 0.0

    claim_ents    = _extract_entities(claim)
    evidence_ents = _extract_entities(evidence)

    # Year mismatch: only fire if BOTH sides have years AND they share a
    # common named entity (same subject). Otherwise a Wikipedia article
    # that mentions Marie Curie's death year (1934) would falsely contradict
    # a claim about her 1903 Nobel Prize.
    claim_years    = set(claim_ents["years"])
    evidence_years = set(evidence_ents["years"])

    if claim_years and evidence_years and claim_years.isdisjoint(evidence_years):
        # Only boost if they share at least one named entity (same topic)
        claim_nouns    = set(n.lower() for n in claim_ents["proper_nouns"])
        evidence_nouns = set(n.lower() for n in evidence_ents["proper_nouns"])
        shared_topic   = claim_nouns & evidence_nouns
        if shared_topic:
            boost += 0.15   # Mild boost — evidence is on-topic but mentions a different year

    return min(boost, 0.4)


# ── Main entry point ─────────────────────────────────────────
def _generate_search_query(claim: str) -> str:
    """Use LLM to generate a concise search query."""
    prompt = f"Generate a 2-4 word search query to verify this claim. Return ONLY the search query, no quotes. Claim: {claim}"
    try:
        query = call_llm(prompt).strip('\"\'\n ')
        return query if len(query) > 3 else claim[:60]
    except:
        return claim[:60]

def run_web_nli(claim: str) -> dict:
    """Full Stage 2a pipeline for one atomic claim."""
    entities = _extract_entities(claim) # Kept for structured contradiction
    search_query = _generate_search_query(claim)

    # Step 1: Targeted retrieval — Wikipedia first, DDG fallback
    evidence_sentences = []
    retrieval_source = "none"

    if search_query:
        evidence_sentences = _search_wikipedia(search_query)
        if evidence_sentences:
            retrieval_source = "wikipedia"

    if not evidence_sentences:
        # Fallback to DDG
        ddg_results = _search_ddg(search_query, max_results=5)
        if ddg_results:
            evidence_sentences = ddg_results
            retrieval_source = "duckduckgo"

    if not evidence_sentences:
        return _empty_result(search_query)

    # Step 2: Semantic Relevance filter
    relevant = _relevance_filter(claim, evidence_sentences, top_k=3)
    if not relevant:
        relevant = evidence_sentences[:3]

    # Step 3: Bidirectional NLI + structured contradiction on each snippet
    snippet_scores = []
    for snippet in relevant:
        nli = _bidirectional_nli(snippet, claim)
        struct_boost = _structured_contradiction(claim, snippet)
        nli["structured_contradiction"] = round(struct_boost, 3)
        nli["snippet"] = snippet[:250]
        snippet_scores.append(nli)

    # Step 4: Aggregate
    entailment_score     = float(np.max([s["entailment"]               for s in snippet_scores]))
    # Use MEAN for contradiction: a single bad/historical snippet must not dominate.
    # np.max was causing 1 out-of-context snippet (e.g. old Celsius scale) to give 0.997
    # and override multiple supportive snippets. Mean requires majority agreement.
    contradiction_score  = float(np.mean([s["contradiction"]           for s in snippet_scores]))
    neutral_score        = float(np.mean([s["neutral"]                  for s in snippet_scores]))
    struct_contradiction = float(np.max([s["structured_contradiction"]  for s in snippet_scores]))

    # Keep struct_contradiction SEPARATE — the meta-classifier uses it as its own signal
    # Only apply a mild additive boost to raw NLI contradiction (not a full stack)
    final_contradiction = min(1.0, contradiction_score + 0.5 * struct_contradiction)

    # Retrieval confidence: how much we trust the evidence
    retrieval_confidence = min(1.0, len(snippet_scores) * 0.2 + (0.3 if retrieval_source == "wikipedia" else 0.0))

    return {
        "entailment_score"       : round(entailment_score,    3),
        "contradiction_score"    : round(final_contradiction, 3),
        "neutral_score"          : round(neutral_score,       3),
        "num_snippets"           : len(snippet_scores),
        "search_query"           : search_query,
        "snippet_scores"         : snippet_scores,
        "retrieval_source"       : retrieval_source,
        "retrieval_confidence"   : round(retrieval_confidence, 3),
        "structured_contradiction": round(struct_contradiction, 3),
    }

def _empty_result(query: str) -> dict:
    return {
        "entailment_score"       : 0.0,
        "contradiction_score"    : 0.0,
        "neutral_score"          : 0.0,
        "num_snippets"           : 0,
        "search_query"           : query,
        "snippet_scores"         : [],
        "retrieval_source"       : "none",
        "retrieval_confidence"   : 0.0,
        "structured_contradiction": 0.0,
    }
