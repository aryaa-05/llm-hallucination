# HalluciDetect: Technical Documentation

## Overview

HalluciDetect is a multi-stage hallucination detection system designed to verify factual claims made by Large Language Models (LLMs). It uses a "defense-in-depth" approach, combining Wikipedia-based evidence retrieval, bidirectional NLI, structured contradiction detection, entropy-based self-consistency checking, and hierarchical decision logic with SHAP explainability.

---

## 1. System Architecture

```
Input: LLM Text Output
         │
    Stage 1: Claim Decomposer
    (Gemini 1.5 Flash primary / LLaMA 3.1 8B via Groq fallback)
    Prompt: temperature=0.0, max_tokens=512, rate-limit=2.0s
         │
    ┌────┴────┐
    │         │
Stage 2a                    Stage 2b
Web Evidence + NLI          QA Consistency
──────────────────          ─────────────
Wikipedia REST API          LLM-based QGen (primary)
→ DuckDuckGo fallback       T5-base-qg-hl (fallback)
all-MiniLM-L6-v2            LLM re-query × 3 (temp=0.7)
(cosine filter ≥ 0.35)      DeBERTa-v3-small (44M)
DeBERTa-v3-base (184M)      BERTScore distilbert (66M)
Bidirectional NLI           Entropy variance penalty
Struct. contradiction        Struct. contradiction QA
    │         │
    └────┬────┘
         │
    Stage 3: Meta-Classifier
    8 features → XGBoost (trained) OR Heuristic ensemble
    LLM Judge for 0.35 ≤ risk ≤ 0.90 (with web evidence)
    SHAP TreeExplainer / approximate SHAP
         │
    Hallucination Risk [0–1] + Verdict + Confidence
```

---

## 2. Stage 1: Claim Decomposition

**File**: `pipeline/claim_decomposer.py`  
**Purpose**: Break multi-sentence text into atomic, independently verifiable claims.

### LLM Router

| Provider | Model | Role | Trigger |
|----------|-------|------|---------|
| Gemini | `gemini-1.5-flash` | Primary | Default |
| Groq | `llama-3.1-8b-instant` | Fallback | Gemini 429 / quota error |

**Rate limiter**: `MIN_CALL_INTERVAL = 2.0s` between any LLM call.  
**Cooldown recovery**: On Gemini 429, the retry delay is parsed from the error message. Groq takes over during cooldown and Gemini auto-recovers.

### Prompt Design

Key enforced rules:
1. **Faithfulness Rule**: Never correct factual errors — extract verbatim
2. **Pronoun Resolution**: Every claim must name the subject explicitly (no "She", "He", "They")
3. **No opinions/filler**: Only verifiable factual assertions
4. **Hard cap**: Maximum `max_claims` (default: 5) per input

### Parameters
```
temperature     = 0.0   (deterministic output)
max_output_tokens = 512
max_claims      = 5     (configurable at runtime)
min_claim_length = 10 chars
```

---

## 3. Stage 2a: Web Evidence + NLI

**File**: `pipeline/web_nli.py`

### 3.1 Search Query Generation
LLM generates a concise 2–4 word search query from the claim. Fallback: first 60 chars of the claim.

### 3.2 Tiered Retrieval

**Primary — Wikipedia REST API**
```
GET https://en.wikipedia.org/api/rest_v1/page/summary/{entity}
```
- Returns clean authoritative article summaries
- Splits into individual sentences (better NLI granularity than full paragraphs)
- Falls back to Wikipedia Search API (`?action=query&list=search`) if direct lookup fails
- Cap: 8 sentences maximum per article

**Fallback — DuckDuckGo**
- Only used when Wikipedia returns nothing
- `max_results=5`, 2 retry attempts, 2-second sleep between

### 3.3 Semantic Relevance Filter
- Model: `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim embeddings)
- Cosine similarity computed between claim embedding and each snippet embedding
- **Threshold: 0.35** — snippets below this are discarded as irrelevant
- Top-3 most relevant snippets retained (`top_k=3`)

### 3.4 Bidirectional NLI
- Model: `cross-encoder/nli-deberta-v3-base` (184M params)
- Trained on SNLI + MultiNLI (3-class: contradiction=0, entailment=1, neutral=2)
- Standard NLI is asymmetric — runs in **both directions** to fix false neutrals

```
forward  = NLI(evidence → claim)
backward = NLI(claim → evidence)

entailment    = max(forward_ent,  backward_ent)   # any direction confirming is enough
neutral       = min(forward_neu,  backward_neu)
contradiction = max(forward_con,  backward_con)
```

**Why bidirectional**: When evidence has extra info ("Marie Curie won in 1903 *along with Pierre Curie and Henri Becquerel*"), the forward direction predicts `neutral`. The backward direction correctly captures the `entailment`. Taking max fixes this.

### 3.5 Structured Contradiction Detection
Detects single-token factual errors that NLI embeddings blur:

| Check | Logic | Boost |
|-------|-------|-------|
| Year mismatch | Claim years ≠ evidence years AND shared named entity | +0.15 |
| Max cap | — | +0.40 |

Guard: only fires when the sides share at least one named entity (prevents spurious fires).

### 3.6 Aggregation
```python
entailment_score    = max(snippet entailments)     # best-case support
contradiction_score = mean(snippet contradictions) # requires majority agreement
neutral_score       = mean(snippet neutrals)
final_contradiction = min(1.0, contradiction_score + 0.5 × struct_contradiction)
retrieval_confidence = min(1.0, num_snippets × 0.2 + (0.3 if Wikipedia else 0))
```

> **Design note**: `max` for entailment (any support counts), `mean` for contradiction (prevents one noisy snippet from condemning a claim).

---

## 4. Stage 2b: QA Consistency

**File**: `pipeline/qa_checker.py`  
**Inspiration**: SelfCheckGPT (sampling-based consistency checking)

### 4.1 Question Generation (3-tier fallback)

| Priority | Method | Trigger |
|----------|--------|---------|
| 1st | LLM-generated (Gemini / LLaMA) | Primary |
| 2nd | T5-base-qg-hl with claim context | Missing slots |
| 3rd | Hard-coded template (`"Is it true that {claim}?"`) | Emergency |

**LLM QGen rules**: Questions must name the specific subject, must be Wh-questions (no yes/no), validated with `_subject_is_present()` regex guard.

**T5-QG** (`valhalla/t5-base-qg-hl`, 223M):
```
input  = "generate question: context: {claim} question about: <hl> {span} <hl>"
output = beam search (num_beams=4, no_repeat_ngram_size=2, max_length=64)
```

### 4.2 LLM Re-query
Each question asked **3 times** at `temperature=0.7` to induce variance.

### 4.3 Scoring

For each sampled answer:
1. **Bidirectional NLI** (`cross-encoder/nli-deberta-v3-small`, 44M)
2. **BERTScore F1** (`distilbert-base-uncased`, 66M)
3. **Structured contradiction** (year/number mismatch): boost +0.4
4. **Explicit contradiction phrases** ("was not", "is incorrect", "is wrong"): penalty +0.5–0.8

**Entropy penalty** (variance across 3 samples):
```python
ent_variance = variance(entailment_scores_across_3_samples)
if ent_variance > 0.15:
    entropy_penalty += 0.30
```

**Final score formula**:
```python
base_score    = max(avg_entailment, avg_bertscore)
nli_con_pen   = max(0, max_contradiction - avg_bertscore)  # excess beyond BERTScore
total_penalty = max(nli_con_pen, max_struct_boost, explicit_phrase_penalty)
final_score   = clamp(base_score - total_penalty - entropy_penalty, 0, 1)
```

### 4.4 Thresholds
```
Normal threshold        = 0.40
With contradiction fired = 0.50  (stricter)
contradiction_fired = (max_penalty > 0.3) OR (max_entropy > 0.2)
label = "FACTUAL" if final_score >= threshold else "HALLUCINATED"
```

### 4.5 Why QA is Auxiliary
LLMs can consistently repeat the same hallucination (e.g., always say "Einstein won for General Relativity"). Consistency ≠ accuracy. QA is a supporting signal, never the sole verdict driver.

---

## 5. Stage 3: Meta-Classifier

**File**: `pipeline/meta_classifier.py`

### 5.1 Input Features (8 total)

| Feature | Source | Description |
|---------|--------|-------------|
| `entailment_score` | Stage 2a | Max bidirectional entailment across snippets |
| `contradiction_score` | Stage 2a | Mean contradiction + 0.5×struct boost |
| `neutral_score` | Stage 2a | Mean neutral across snippets |
| `num_snippets` | Stage 2a | Number of retrieved & filtered snippets |
| `consistency_score` | Stage 2b | QA final score (clamped 0–1) |
| `electra_avg` | Stage 2b | Average NLI entailment support (QA) |
| `bertscore_avg` | Stage 2b | Average BERTScore F1 (QA) |
| `max_contradiction_qa` | Stage 2b | Max contradiction or struct boost (QA) |

### 5.2 Scoring Mode

**Primary**: XGBoost classifier (`models/meta_classifier.pkl`)
- Trained on HaluEval + FactCC datasets
- Outputs `predict_proba(X)[0][1]` as hallucination risk
- SHAP: `shap.TreeExplainer`

**Fallback** (cold-start, no .pkl file): Heuristic hierarchical ensemble

### 5.3 Heuristic Decision Pipeline
```
A: Retrieval Quality Gate
   └── No web AND QA < 0.4 → risk = 0.5 (INSUFFICIENT EVIDENCE)

B: Evidence Agreement
   ├── B0: BERTScore override — high semantic QA overlap despite confusing web → factual
   │       Fires when: BERTScore > 0.80 AND entailment < 0.30 AND contradiction < 0.65
   ├── B1: Web entailment > 0.65 → FACTUAL  (0.75×ent + 0.25×qa)
   ├── B2: Raw NLI contradiction > 0.65 → HALLUCINATION  (0.15×qa)
   ├── B3: No web, QA > 0.5 AND NLI > 0.3 → weak factual
   └── B4: No web, QA < 0.25 → hallucination signal

C: Structured Contradiction Override
   └── struct_con > 0.35 → boost risk (0.3×qa - 0.3×struct)

Fallback blend:
   With web:    0.40×ent + 0.35×qa + 0.25×qa_nli
   Without web: 0.60×qa  + 0.40×qa_nli
```

### 5.4 LLM-as-a-Judge

Activated when: `0.35 ≤ risk_score ≤ 0.90` AND web snippets exist AND claim is non-empty.

```
Input:  claim + top-3 web snippets
Prompt: Chain-of-Thought → "Think step-by-step. End with [SUPPORTED], [CONTRADICTED], or [UNVERIFIABLE]"
Output: [SUPPORTED]→0.15, [CONTRADICTED]→0.85, [UNVERIFIABLE]→0.50
```

Purpose: Prevents false positives from noisy/out-of-context web snippets.

### 5.5 Branch Weights
```
With web evidence:    web=0.45,  qa=0.55
Without web evidence: web=0.10,  qa=0.90
```

### 5.6 Verdict Thresholds

| Risk Score | Verdict |
|------------|---------|
| `< 0.35` | ✅ LIKELY FACTUAL |
| `0.35 – 0.70` | ⚠️ UNCERTAIN — NEEDS REVIEW |
| `≥ 0.70` | ❌ HALLUCINATION DETECTED |

Special: `INSUFFICIENT EVIDENCE` — no web AND QA < 0.4

**Confidence score**:
```python
confidence = min(abs(risk_score - 0.5) * 2, 1.0)
# 0.0 at perfectly ambiguous (0.5), 1.0 at extreme (0.0 or 1.0)
```

### 5.7 SHAP Explanations
- XGBoost path: `shap.TreeExplainer` — exact SHAP values
- Heuristic path: approximate contributions based on branch weights
- Every verdict in the UI includes a per-feature SHAP bar chart

---

## 6. Models Summary

| Component | Model | Params | Fine-tuned? | Task |
|-----------|-------|--------|-------------|------|
| Web NLI | `cross-encoder/nli-deberta-v3-base` | 184M | No (SNLI+MNLI) | 3-class NLI on web evidence |
| QA NLI | `cross-encoder/nli-deberta-v3-small` | 44M | No (SNLI+MNLI) | 3-class NLI on QA pairs |
| QGen | `valhalla/t5-base-qg-hl` | 223M | No (SQuAD) | Question generation |
| Semantic Filter | `all-MiniLM-L6-v2` | 22M | No (MS MARCO) | Sentence embeddings |
| BERTScore | `distilbert-base-uncased` | 66M | No | Token-level semantic F1 |
| Claim Decomposer | Gemini 1.5 Flash | API | No (prompting) | Atomic claim extraction |
| Claim Decomposer (fallback) | LLaMA 3.1 8B Instant | API | No (prompting) | Atomic claim extraction |
| Meta-Classifier | XGBoost | ~100 trees | **Yes** (HaluEval+FactCC) | Signal fusion |

---

## 7. Directory Structure

```
hallucination_detector/
├── streamlit_app.py          # Interactive dashboard (Inter + JetBrains Mono UI)
├── pipeline/
│   ├── __init__.py
│   ├── claim_decomposer.py   # Stage 1: LLM claim extraction with smart router
│   ├── web_nli.py            # Stage 2a: Wikipedia + DDG + bidirectional DeBERTa NLI
│   ├── qa_checker.py         # Stage 2b: T5-QG + SelfCheckGPT + BERTScore
│   ├── meta_classifier.py    # Stage 3: XGBoost/heuristic + LLM Judge + SHAP
│   └── orchestrator.py       # Pipeline wiring, progress callbacks
├── models/
│   ├── README.md
│   └── meta_classifier.pkl   # Optional trained XGBoost (falls back to heuristic)
├── requirements.txt          # All dependencies (17 packages)
├── documentation.md          # This file
└── README.md
```

---

## 8. How to Run

### Local Development
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
streamlit run streamlit_app.py
```

### Hugging Face Spaces
Add `GEMINI_API_KEY` and `GROQ_API_KEY` as repository secrets in **Settings → Repository secrets**.

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Wikipedia first, DDG fallback | Wikipedia is free, authoritative, no API key, reliable on HF Spaces free tier |
| `max` for entailment, `mean` for contradiction | One confirming snippet is sufficient; one noisy snippet shouldn't condemn a claim |
| Bidirectional NLI | Extra information in evidence causes false neutral in forward-only NLI |
| QA is auxiliary | LLMs repeat hallucinations consistently — consistency ≠ accuracy |
| BERTScore tie-breaker | NLI confused by paraphrasing/negation; BERTScore gives raw semantic overlap |
| Faithfulness rule | Decomposer must never correct errors — would verify corrected (true) claims |
| LLM Judge for 0.35–0.90 | Middle-ground risk needs contextual reasoning, not just numerical fusion |
| XGBoost with heuristic fallback | Works with 8 features, SHAP-compatible, cold-start safe |

---

## 10. Known Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|------------|--------|
| Wikipedia coverage | Obscure/recent topics may lack Wikipedia articles |
| CPU inference latency | ~25–40s per claim on CPU |
| QA self-consistency flaw | LLMs can consistently repeat hallucinations |
| English only | Non-English claims fail or produce poor results |
| XGBoost cold-start | Heuristic is less accurate than trained XGBoost |

### Planned Improvements
1. **Train XGBoost on labeled data** — HaluEval + FactCC labeling pipeline
2. **Wikidata integration** — Structured SPARQL slot-level verification
3. **Multilingual support** — mDeBERTa for non-English claims
4. **Batch NLI inference** — Reduce per-claim latency
5. **Confidence calibration** — Platt scaling on XGBoost outputs
