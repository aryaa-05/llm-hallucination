---
title: HalluciDetect
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: 1.57.0
app_file: streamlit_app.py
pinned: false
---

# 🔍 HalluciDetect — LLM Hallucination Detector

Multi-source, explainable hallucination detection pipeline for LLM-generated text.  
Every claim is verified independently using real-world evidence and internal consistency checks.

---

## Architecture

```
Input: LLM-Generated Text
         │
    Stage 1: Claim Decomposer
    (Gemini 1.5 Flash primary / LLaMA 3.1 8B fallback)
    Output: Atomic, pronoun-resolved claims
         │
    ┌────┴────┐
    │         │
Stage 2a                    Stage 2b
Web Evidence + NLI          QA Consistency (SelfCheckGPT-style)
──────────────────          ──────────────────────────────────
Wikipedia REST API          T5-base-qg-hl (Question Gen)
→ DuckDuckGo fallback       LLM re-query × 3 (temp=0.7)
all-MiniLM-L6-v2 filter     DeBERTa-v3-small NLI
DeBERTa-v3-base BiNLI       BERTScore (distilbert) tie-breaker
Struct. contradiction        Entropy variance penalty
    │         │
    └────┬────┘
         │
    Stage 3: Meta-Classifier
    XGBoost (trained) OR Heuristic ensemble
    + LLM-as-a-Judge for ambiguous cases (0.35–0.90)
    + SHAP explainability
         │
    Hallucination Risk Score [0–1]
         │
    ┌────┴────┬──────────────┐
    │         │              │
 FACTUAL  UNCERTAIN   HALLUCINATION
 <0.35    0.35–0.70    ≥0.70
```

---

## Features

| Feature | Details |
|---------|---------|
| **Atomic Verification** | Complex text broken into individual verifiable claims with full pronoun resolution |
| **Dual-Track Evidence** | Real-time semantic web search (Wikipedia primary, DDG fallback) + internal LLM self-consistency |
| **Bidirectional NLI** | DeBERTa-v3 NLI run in both directions — fixes asymmetric entailment failures |
| **Entropy Detection** | LLM re-queried 3× at temp=0.7; high variance across answers flags hallucination |
| **Structured Contradiction** | Year/entity slot mismatch detection beyond what NLI embeddings can catch |
| **LLM Judge Fallback** | Ambiguous claims (risk 0.35–0.90) routed to LLM Chain-of-Thought judge |
| **Explainable AI** | Every verdict includes a SHAP chart showing per-feature contribution |
| **Smart LLM Router** | Gemini 1.5 Flash → Groq LLaMA 3.1 8B auto-failover on rate limits |
| **BERTScore Tie-breaker** | Semantic F1 used alongside NLI to reward partial matches |

---

## Verdict Thresholds

| Risk Score | Verdict |
|------------|---------|
| `< 0.35` | ✅ LIKELY FACTUAL |
| `0.35 – 0.70` | ⚠️ UNCERTAIN — NEEDS REVIEW |
| `≥ 0.70` | ❌ HALLUCINATION DETECTED |

---

## Models Used

| Component | Model | Params |
|-----------|-------|--------|
| Claim Decomposer | Gemini 1.5 Flash / LLaMA 3.1 8B | API |
| Semantic Filter | `all-MiniLM-L6-v2` | 22M |
| Web NLI | `cross-encoder/nli-deberta-v3-base` | 184M |
| QA NLI | `cross-encoder/nli-deberta-v3-small` | 44M |
| Question Gen | `valhalla/t5-base-qg-hl` | 223M |
| BERTScore | `distilbert-base-uncased` | 66M |
| Meta-Classifier | XGBoost + SHAP | ~100 trees |

---

## Setup

### 1. Set API Secrets

If deploying to Hugging Face Spaces, add to **Settings → Repository secrets**:

| Secret | Value |
|--------|-------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `GROQ_API_KEY` | Your Groq API key |

### 2. Local Installation

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
streamlit run streamlit_app.py
```

---

## Project Structure

```
hallucination_detector/
├── streamlit_app.py          # Interactive Streamlit dashboard
├── pipeline/
│   ├── claim_decomposer.py   # Stage 1: Gemini/Groq LLM claim extraction
│   ├── web_nli.py            # Stage 2a: Wikipedia + bidirectional DeBERTa NLI
│   ├── qa_checker.py         # Stage 2b: T5-QG + SelfCheckGPT + BERTScore
│   ├── meta_classifier.py    # Stage 3: XGBoost/Heuristic + LLM Judge + SHAP
│   └── orchestrator.py       # Pipeline wiring and progress callbacks
├── models/
│   └── meta_classifier.pkl   # Trained XGBoost model (optional, falls back to heuristic)
├── requirements.txt
├── documentation.md          # Full technical documentation
└── README.md
```

---

## Notes

- **Web Search**: Wikipedia REST API is primary (no key required, authoritative). DuckDuckGo is fallback only.
- **Rate limiting**: 1.5s between claims (DDG protection), 2.0s between LLM calls.
- **CPU deployment**: All transformer models run on CPU — designed for HF Spaces free tier (~25–40s per claim).
- **XGBoost cold-start**: If `models/meta_classifier.pkl` is absent, a heuristic ensemble runs automatically.
- **Semantic filter threshold**: 0.35 cosine similarity (MiniLM embeddings).

## Detailed Documentation

See [documentation.md](documentation.md) for full technical deep-dive.