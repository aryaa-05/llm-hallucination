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
Every claim is verified independently using real-world evidence and internal
consistency checks, fused by a hybrid ML + LLM judge with SHAP explanations.

**Live demo:** [aryaa-05/llm-hallucination — Hugging Face Spaces](https://huggingface.co/spaces/aryaa-05/llm-hallucination)
**Repo:** [github.com/aryaa-05/llm-hallucination](https://github.com/aryaa-05/llm-hallucination)

---

## Architecture

```
Input: LLM-Generated Text
         │
    Stage 1: Claim Decomposer
    (Gemini primary / Groq fallback / OpenRouter emergency)
    Output: Atomic, pronoun-resolved claims  (max default = 5)
         │
    ┌────┴────┐
    │         │
Stage 2a                    Stage 2b
Web Evidence + NLI          QA Consistency (SelfCheckGPT-style)
──────────────────          ──────────────────────────────────
Wikipedia REST API          LLM-generated questions (primary)
→ DuckDuckGo fallback       T5-base-qg-hl (fallback)
all-MiniLM-L6-v2 filter     LLM re-query × 3 (temp=0.7)
DeBERTa-v3-base BiNLI       DeBERTa-v3-small NLI
(top semantic filter ≥0.35) BERTScore (distilbert) tie-breaker
Structured contradiction    Entropy variance penalty
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
| **Resilient LLM Router** | Gemini → Groq → OpenRouter auto-failover on rate limits/failure |
| **BERTScore Tie-breaker** | Semantic F1 used alongside NLI to reward partial matches |

---

## LLM Provider Routing

The claim decomposer and downstream LLM calls use a resilient 3-tier router:

| Provider | Model | Role | Trigger |
|----------|-------|------|---------|
| Gemini | `gemini-2.5-flash-lite` | Primary | Default |
| Groq | `openai/gpt-oss-20b` | Fallback | Gemini 429 / quota / failure |
| OpenRouter | `openrouter/free` | Emergency | Gemini + Groq both fail |

- **Rate limiter:** `MIN_CALL_INTERVAL = 2.0s` between LLM calls.
- **Cooldown:** On Gemini 429, retry delay is parsed from the error; Gemini auto-recovers and Groq covers the cooldown window.
- **Single key each:** `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`.

---

## Verdict Thresholds

| Risk Score | Verdict |
|------------|---------|
| `< 0.35` | ✅ LIKELY FACTUAL |
| `0.35 – 0.70` | ⚠️ UNCERTAIN — NEEDS REVIEW |
| `≥ 0.70` | ❌ HALLUCINATION DETECTED |

Special case: `INSUFFICIENT EVIDENCE` — no web evidence AND QA < 0.4.

---

## Models Used

| Component | Model | Params |
|-----------|-------|--------|
| Claim Decomposer | Gemini `gemini-2.5-flash-lite` / Groq `openai/gpt-oss-20b` / OpenRouter `openrouter/free` | API |
| Semantic Filter | `all-MiniLM-L6-v2` | 22M |
| Web NLI | `cross-encoder/nli-deberta-v3-base` | 184M |
| QA NLI | `cross-encoder/nli-deberta-v3-small` | 44M |
| Question Gen | `valhalla/t5-base-qg-hl` | 223M |
| BERTScore | `distilbert-base-uncased` | 66M |
| Meta-Classifier | XGBoost (optional) + heuristic fallback + LLM judge + SHAP | ~100 trees |

---

## Setup

### 1. Set API Secrets

If deploying to Hugging Face Spaces, add to **Settings → Variables and secrets**
(as **secrets**, one per key):

| Secret | Value |
|--------|-------|
| `GEMINI_API_KEY` | Your Gemini API key (Google AI Studio) |
| `GROQ_API_KEY` | Your Groq API key |
| `OPENROUTER_API_KEY` | Your OpenRouter API key (emergency provider) |

> **Note:** Hugging Face secret *names* may not contain underscores and secret
> *values* are capped at 64 characters. If your OpenRouter key is longer than
> 64 chars, split it across multiple secrets (e.g. `OPENROUTERKEYPART1`,
> `OPENROUTERKEYPART2`, ...) and concatenate them in the code.

### 2. Local Installation

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
streamlit run streamlit_app.py
```

### 3. HF Space build step

Space `setup.sh` installs an extra spaCy model:
```bash
python -m spacy download en_core_web_sm
```

---

## Deployment & Syncing (GitHub + HuggingFace)

The project is a single git repo with **two remotes**: GitHub (`origin`) and the
HF Space (`hf`). To commit and push to both:

```powershell
# sync.ps1 — commit, push GitHub, push HF Space (auto-rebuilds)
.\sync.ps1 "your commit message"

# or manually
git push origin main
git push hf main
```

Remotes:
```
origin  https://github.com/aryaa-05/llm-hallucination.git
hf      https://huggingface.co/spaces/aryaa-05/llm-hallucination
```

---

## Project Structure

```
llm-hallucination-detector/
├── streamlit_app.py          # Interactive Streamlit dashboard
├── pipeline/
│   ├── __init__.py
│   ├── claim_decomposer.py   # Stage 1: Gemini/Groq/OpenRouter LLM claim extraction
│   ├── web_nli.py            # Stage 2a: Wikipedia + bidirectional DeBERTa NLI
│   ├── qa_checker.py         # Stage 2b: T5-QG + SelfCheckGPT + BERTScore
│   ├── meta_classifier.py    # Stage 3: XGBoost/Heuristic + LLM Judge + SHAP
│   └── orchestrator.py       # Pipeline wiring and progress callbacks
├── models/
│   └── README.md             # (optional meta_classifier.pkl goes here)
├── tests / analysis
│   ├── comprehensive_test.py
│   ├── trace_water.py
│   ├── test_cos.py / test_logic.py / test_nli.py / test_run.py
├── requirements.txt
├── setup.sh
├── documentation.md          # Full conference-grade technical documentation
└── README.md
```

---

## Notes

- **Web Search:** Wikipedia REST API is primary (no key required, authoritative). DuckDuckGo is fallback only.
- **Rate limiting:** 1.5s between claims (DDG protection), 2.0s between LLM calls.
- **CPU deployment:** All transformer models run on CPU — designed for HF Spaces free tier (~25–40s per claim).
- **Meta-classifier:** If `models/meta_classifier.pkl` is absent, a heuristic ensemble runs automatically (`model_used = "heuristic"`, or `"heuristic+llm_judge"`).
- **Semantic filter threshold:** 0.35 cosine similarity (MiniLM embeddings).
- **Confidentiality:** API keys are only stored as HF Space secrets — never committed to git (see `.gitignore`).

## Detailed Documentation

See [documentation.md](documentation.md) for a full technical deep-dive suitable
for research conference presentation.
