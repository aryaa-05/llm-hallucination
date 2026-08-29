# HalluciDetect: Multi-Source, Explainable Detection of Factual Hallucinations in Large Language Model Outputs

---

## Abstract

Large Language Models (LLMs) generate fluent text that frequently contains
**factual hallucinations** — confident statements that are untrue or
unsupported by evidence. Automatically detecting these is challenging because
hallucinations are grammatical, plausible, and internally consistent.

We present **HalluciDetect**, a multi-stage, evidence-grounded hallucination
detection system. Given a span of LLM-generated text, the system:

1. **Decomposes** the text into atomic, pronoun-resolved claims using a
   resilient 3-tier LLM router (Gemini → Groq → OpenRouter).
2. **Verifies** each claim along two independent tracks:
   - **Stage 2a** — real-time *semantic web evidence* retrieval (Wikipedia
     primary, DuckDuckGo fallback) combined with *bidirectional* DeBERTa-v3
     Natural Language Inference (NLI), semantic relevance filtering, and
     structured contradiction detection.
   - **Stage 2b** — *internal self-consistency checking* (a SelfCheckGPT-style
     approach) using LLM-generated questions, stochastic re-querying,
     BERTScore, and entropy-based variance penalties.
3. **Fuses** the per-claim signals with a **hybrid meta-classifier**
   (XGBoost when a trained model is available, otherwise a transparent
   heuristic ensemble), an **LLM-as-a-Judge** for ambiguous cases, and
   **SHAP** explainability.

The output is a hallucination risk score in **[0, 1]** per claim, an overall
verdict for the whole text, and an interactive SHAP explanation of each
decision. The system runs end-to-end on CPU, is deployed as a public Streamlit
application on Hugging Face Spaces, and is designed to be explainable and
auditable.

---

## 1. Background and Motivation

Large language models are now used for summarization, question answering,
knowledge-intensive assistants, and automated fact-checking support. A
persistent failure mode is **hallucination** — generating content that is
faithful in *form* but not in *fact*. Hallucinations can be divided roughly
into:

- **Intrinsic hallucination** — information that contradicts the source/input.
- **Extrinsic hallucination** — information that is plausible but not
  present in (or supported by) the source.

Because the text is internally coherent, purely distributional signals are
insufficient. Effective detection typically requires either
**(a)** comparison against an authoritative external corpus, or
**(b)** probing the *consistency* of the model's own generations. HalluciDetect
combines both paradigms.

**Key insight guiding this work:** no single signal is reliable. Web evidence
proves facts; self-consistency detects fabrication; but both have blind spots.
A **defense-in-depth** architecture — where independent signals are fused
rather than trusted individually — yields more robust and *explainable*
verdicts.

---

## 2. Related Work

| Line of work | Representative approach | Relevance to HalluciDetect |
|--------------|------------------------|----------------------------|
| **FactCC / Falsesum** | Token-masking + Roberta-based faithfulness classification (Kryściński et al., 2020) | Basis for classifier-based verification of claims |
| **SelfCheckGPT** | Sampling-based consistency; sample multiple answers and compare | Directly inspires Stage 2b QA consistency |
| **QAGS** | Question-generation + answer matching for faithfulness (Wang et al., 2020) | Inspires the LLM-generated question-answer track |
| **HaluEval** | Large hallucination evaluation benchmark (Li et al., 2023) | Training/validation data for the XGBoost layer |
| **NLI-based verification** | Entailment between claim and evidence (e.g., DeBERTa SNLI/MNLI) | Core building block for both Stage 2a and 2b |
| **Fact-checking pipelines** | Retrieval-augmented verification (e.g., FEVER) | Motivates the web-evidence retrieval track (Stage 2a) |

HalluciDetect differentiates itself by **combining** external-evidence NLI
*and* internal-consistency QA in a single explainable, CPU-deployable pipeline
with a hierarchical fusion stage, rather than relying on any single method.

---

## 3. System Design and Methodology

### 3.1 Overview

```
Input: LLM-Generated Text
         │
    Stage 1: Claim Decomposer        (claim_decomposer.py)
    (3-tier LLM router: Gemini → Groq → OpenRouter)
    Output: atomic, pronoun-resolved claims
         │
    ┌────┴────┐
    │         │
Stage 2a (web_nli.py)          Stage 2b (qa_checker.py)
Web Evidence + NLI             QA Consistency
Wikipedia REST (primary)       LLM question generation
└→ DuckDuckGo (fallback)       LLM re-query × 3 (temp=0.7)
all-MiniLM-L6-v2 filter        DeBERTa-v3-small NLI
DeBERTa-v3-base BiNLI          BERTScore tie-breaker
Structured contradiction       Entropy variance penalty
    │         │
    └────┬────┘
         │
    Stage 3: Meta-Classifier    (meta_classifier.py)
    XGBoost OR heuristic ensemble
    + LLM-as-a-Judge (0.35 ≤ risk ≤ 0.90)
    + SHAP explainability
         │
    Per-claim risk [0–1] → Aggregated verdict (orchestrator.py)
```

### 3.2 The Resilient LLM Router (Stage 1)

**File:** `pipeline/claim_decomposer.py`

All LLM calls throughout the pipeline travel through one router. This makes the
system robust to provider outages, rate limits, and quota exhaustion.

| Provider | Model | Role | Trigger |
|----------|-------|------|---------|
| Gemini | `gemini-2.5-flash-lite` | Primary | Default |
| Groq | `openai/gpt-oss-20b` | Fallback | Gemini 429 / quota / any failure |
| OpenRouter | `openrouter/free` | Emergency | Both Gemini and Groq fail |

Design properties:

- **Keys.** A single secret per provider: `GEMINI_API_KEY`, `GROQ_API_KEY`,
  `OPENROUTER_API_KEY`.
- **Rate limiting.** A global `MIN_CALL_INTERVAL = 2.0s` throttles calls.
- **Cooldown & auto-recovery.** On a Gemini `429`/quota error, the retry delay
  is parsed from the error message (default 60s); Groq serves calls during the
  cooldown and Gemini automatically recovers.
- **Failover order.** Gemini is attempted first; on any Gemini failure the
  caller `continue`s to Groq; if Groq also fails, **OpenRouter** is invoked as
  the emergency tier before the pipeline raises.
- **Diagnostics.** The raised error includes which keys are set plus the
  underlying per-provider exception text, so configuration problems are
  immediately visible.

#### 3.2.1 Decomposition Prompt

The decomposer prompt enforces five rules critical to downstream verification:

1. **Faithfulness rule** — never "correct" a factual error in the input
   (we must verify the claim *as written*).
2. **Pronoun resolution** — every claim names its subject explicitly
   (no `She`, `He`, `They`, `It`, `This`).
3. **Verifiability** — only extract factual assertions; drop opinions/filler.
4. **Non-redundancy** — output the most distinct, independently verifiable
   claims.
5. **Hard cap** — at most `max_claims` claims (default `5`, UI-adjustable).

Generation parameters:
```
temperature       = 0.0     (deterministic)
max_output_tokens = 512
min_claim_length  = 10 chars (claims shorter than this are dropped)
```

---

## 4. Stage 2a — Web Evidence and NLI

**File:** `pipeline/web_nli.py`

Goal: answer *"Does the real world support this claim?"* by retrieving
authoritative text and measuring whether it entails, contradicts, or is
neutral toward the claim.

### 4.1 Search Query Generation

An LLM call generates a concise 2–4 word search query from each claim.
Fallback: the first 60 characters of the claim.

### 4.2 Tiered Retrieval

1. **Primary — Wikipedia REST API** (`https://en.wikipedia.org/api/rest_v1/page/summary/{entity}`)
   - Authoritative, free, no API key.
   - Article summaries are split into individual sentences (better NLI
     granularity).
   - Falls back to the Wikipedia search API if a direct lookup fails.
   - Cap: 8 sentences per article.
2. **Fallback — DuckDuckGo** (`duckduckgo_search`)
   - Used only when Wikipedia returns nothing.
   - `max_results=5`, 2 retry attempts, 2s sleep between attempts.

### 4.3 Semantic Relevance Filter

- Model: `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim).
- Cosine similarity between the claim embedding and each snippet embedding.
- **Threshold 0.35** — snippets below this are discarded.
- Top-3 relevant snippets retained (`top_k=3`).

### 4.4 Bidirectional NLI

- Model: `cross-encoder/nli-deberta-v3-base` (184M params, SNLI+MultiNLI).
- Standard NLI is **asymmetric**; HalluciDetect runs it in **both directions**
  to fix false neutrals when evidence contains extra context:

```
forward  = NLI(evidence → claim)
backward = NLI(claim   → evidence)

entailment    = max(forward_ent,  backward_ent)   # any direction confirming is enough
neutral       = min(forward_neu,  backward_neu)
contradiction = max(forward_con,  backward_con)
```

*Example of why this matters:* evidence "Marie Curie won the 1903 Nobel Prize
*along with Pierre Curie and Henri Becquerel*" may yield `neutral` in the
forward direction, but the backward direction correctly captures
`entailment`.

### 4.5 Structured Contradiction Detection

Catches single-token factual errors that NLI embeddings blur:

| Check | Logic | Risk boost |
|-------|-------|-----------|
| Year mismatch | Claim years ≠ evidence years AND shared named entity | +0.15 |
| Explicit numerical caps | — | +0.40 |

Guard: the check only fires when both sides share at least one named entity,
preventing spurious triggers.

### 4.6 Aggregation

```python
entailment_score    = max(snippet entailments)                 # best-case support
contradiction_score = mean(snippet contradictions)             # requires agreement
neutral_score       = mean(snippet neutrals)
final_contradiction = min(1.0, contradiction_score + 0.5 * struct_contradiction)
retrieval_confidence = min(1.0, num_snippets * 0.2 + (0.3 if wikipedia else 0.0))
```

> **Design note.** `max` for entailment (any supporting snippet is enough);
> `mean` for contradiction (one noisy snippet must not condemn a claim).

`retrieval_source` is reported as `wikipedia`, `duckduckgo`, or `none`.

---

## 5. Stage 2b — QA Consistency (SelfCheckGPT-style)

**File:** `pipeline/qa_checker.py`

Goal: answer *"Is the model consistently saying the same thing?"* — probing
whether a claim survives the model's own re-generation under stochastic
sampling.

### 5.1 Question Generation (3-tier fallback)

| Priority | Method | Trigger |
|----------|--------|---------|
| 1st | LLM-generated questions (via router) | Primary |
| 2nd | `valhalla/t5-base-qg-hl` with claim context | Missing slots |
| 3rd | Template `"Is it true that {claim}?"` | Emergency |

LLM-generated questions must name the specific subject and be Wh-questions
(validated by a `_subject_is_present()` regex guard). The T5-QG model
(223M, SQuAD) uses beam search (`num_beams=4`, `no_repeat_ngram_size=2`,
`max_length=64`).

### 5.2 Stochastic Re-query

Each question is asked **3 times** at `temperature=0.7` to induce variance
across samples.

### 5.3 Scoring Each Sample

For each sampled answer:
1. **Bidirectional NLI** (`cross-encoder/nli-deberta-v3-small`, 44M).
2. **BERTScore F1** (`distilbert-base-uncased`, 66M).
3. **Structured contradiction** (year/number mismatch): boost +0.4.
4. **Explicit contradiction phrases** ("was not", "is incorrect",
   "is wrong"): penalty +0.5–0.8.

**Entropy penalty** (variance across the 3 samples):
```python
ent_variance = variance(entailment_scores_across_3_samples)
if ent_variance > 0.15:
    entropy_penalty += 0.30
```

**Final score:**
```python
base_score    = max(avg_entailment, avg_bertscore)
nli_con_pen   = max(0, max_contradiction - avg_bertscore)
total_penalty = max(nli_con_pen, max_struct_boost, explicit_phrase_penalty)
final_score   = clamp(base_score - total_penalty - entropy_penalty, 0, 1)
```

### 5.4 Thresholds

```
normal threshold         = 0.40
with-contradiction thr.  = 0.50   (stricter)
contradiction_fired = (max_penalty > 0.3) or (max_entropy > 0.2)
label = "FACTUAL" if final_score >= threshold else "HALLUCINATED"
```

### 5.5 Why QA is Auxiliary

> **Critical caveat.** LLMs can be *consistently wrong* — repeating the same
> hallucination across all samples (e.g., always claiming "Einstein won for
> General Relativity"). Consistency therefore ≠ accuracy. QA is a supporting
> signal, never the sole driver of a verdict. This is why Stage 3 blends QA
> with external web evidence.

---

## 6. Stage 3 — Meta-Classifier

**File:** `pipeline/meta_classifier.py`

Goal: fuse Stage 2a and Stage 2b into a single, explainable risk score.

### 6.1 Input Features (8)

| Feature | Source | Description |
|---------|--------|-------------|
| `entailment_score` | 2a | Max bidirectional entailment across snippets |
| `contradiction_score` | 2a | Mean contradiction + 0.5×struct boost |
| `neutral_score` | 2a | Mean neutral across snippets |
| `num_snippets` | 2a | Number of retrieved & filtered snippets (verifiability proxy) |
| `consistency_score` | 2b | QA final score (clamped 0–1) |
| `electra_avg` | 2b | Average NLI entailment support (QA) |
| `bertscore_avg` | 2b | Average BERTScore F1 (QA) |
| `max_contradiction_qa` | 2b | Max contradiction or struct boost (QA) |

### 6.2 Scoring Mode

**Primary — XGBoost:** if `models/meta_classifier.pkl` exists, the model is
loaded and `predict_proba(X)[0][1]` gives the risk; SHAP uses
`shap.TreeExplainer` for exact values. `model_used = "xgboost"`.

**Fallback — Heuristic ensemble** (cold start, no `.pkl` file). This is a
transparent hierarchical scoring function returning a factuality score
`[0,1]` (converted to risk as `1 - factuality`):

```
A: Retrieval quality gate
   └── no web AND QA < 0.4 → risk = 0.5 (INSUFFICIENT EVIDENCE)

B: Evidence agreement
   ├── B0: BERTScore override — BERTScore>0.80 AND ent<0.30 AND con<0.65
   │        → factual (0.15·qa + 0.65·bert + 0.20·qa_nli)
   ├── B1: web entailment > 0.65 → factual (0.75·ent + 0.25·qa)
   ├── B2: RAW NLI contradiction > 0.65 → hallucination (0.15·qa)
   ├── B3: no web AND QA>0.5 AND qa_nli>0.3 → weak factual (0.7·qa+0.3·qa_nli)
   └── B4: no web AND QA<0.25 → hallucination signal (qa)

C: Structured contradiction override
   └── struct_con > 0.35 → boost: (0.3·qa − 0.3·struct_con)

Fallback blend:
   with web:    0.40·ent + 0.35·qa + 0.25·qa_nli
   without web: 0.60·qa + 0.40·qa_nli
```

### 6.3 LLM-as-a-Judge

Activated when: **0.35 ≤ risk ≤ 0.90** AND web snippets exist AND claim is
non-empty.

```
Input:  claim + top-3 web snippets
Prompt: Chain-of-Thought → "...respond with [SUPPORTED], [CONTRADICTED], or [UNVERIFIABLE]"
Output: [SUPPORTED]→0.15, [CONTRADICTED]→0.85, [UNVERIFIABLE]→0.50
```

The judge is only trusted when it makes a *confident* decision (`≠ 0.50`);
otherwise the numerical risk stands. `model_used` becomes `heuristic+llm_judge`
or `xgboost+llm_judge`.

### 6.4 Branch Weights (reported to the UI)

```
With web evidence:    web=0.45,  qa=0.55
Without web evidence: web=0.10,  qa=0.90
```

### 6.5 Verdict Thresholds

| Risk Score | Verdict |
|------------|---------|
| `< 0.35` | ✅ LIKELY FACTUAL |
| `0.35 – 0.70` | ⚠️ UNCERTAIN — NEEDS REVIEW |
| `≥ 0.70` | ❌ HALLUCINATION DETECTED |

Special: `INSUFFICIENT EVIDENCE` — no web AND consistency < 0.4.

**Confidence score:**
```python
confidence = min(abs(risk_score - 0.5) * 2, 1.0)
```

### 6.6 Aggregation (response level)

`aggregate_results` averages per-claim risks for `overall_risk` and labels
the response `HALLUCINATION DETECTED` if any claim is confirmed, else
`UNCERTAIN — NEEDS REVIEW` if any claim is uncertain, else `LIKELY FACTUAL`.

### 6.7 SHAP Explainability

- **XGBoost path:** exact SHAP values via `shap.TreeExplainer`.
- **Heuristic path:** approximate per-feature contributions using branch
  weights (only surrogate SHAP).
- Every verdict ships a **per-feature SHAP bar chart** to the UI, making each
  classification auditable by a human reviewer — a key requirement for
  conference demonstration and ethics review.

---

## 7. Models Summary

| Component | Model | Params | Fine-tuned? | Task |
|-----------|-------|--------|-------------|------|
| Web NLI | `cross-encoder/nli-deberta-v3-base` | 184M | No (SNLI+MNLI) | 3-class NLI on web evidence |
| QA NLI | `cross-encoder/nli-deberta-v3-small` | 44M | No (SNLI+MNLI) | 3-class NLI on QA pairs |
| QGen | `valhalla/t5-base-qg-hl` | 223M | No (SQuAD) | Question generation |
| Semantic Filter | `all-MiniLM-L6-v2` | 22M | No (MS MARCO) | Sentence embeddings |
| BERTScore | `distilbert-base-uncased` | 66M | No | Token-level semantic F1 |
| Claim Decomposer | Gemini `gemini-2.5-flash-lite` | API | No (prompting) | Atomic claim extraction |
| Claim Decomposer (fallback) | Groq `openai/gpt-oss-20b` | API | No (prompting) | Atomic claim extraction |
| Claim Decomposer (emergency) | OpenRouter `openrouter/free` | API | No (prompting) | Atomic claim extraction |
| Meta-Classifier | XGBoost + SHAP | ~100 trees | **Yes** (HaluEval+FactCC) | Signal fusion |

---

## 8. Directory Structure

```
llm-hallucination-detector/
├── streamlit_app.py          # Interactive Streamlit dashboard (UI)
├── pipeline/
│   ├── __init__.py
│   ├── claim_decomposer.py   # Stage 1: LLM router + claim extraction
│   ├── web_nli.py            # Stage 2a: Web + bidirectional DeBERTa NLI
│   ├── qa_checker.py         # Stage 2b: QGen + SelfCheckGPT + BERTScore
│   ├── meta_classifier.py    # Stage 3: XGBoost/heuristic + LLM Judge + SHAP
│   └── orchestrator.py       # Pipeline wiring + progress callbacks
├── models/
│   ├── README.md
│   └── meta_classifier.pkl   # Optional trained XGBoost (else heuristic)
├── requirements.txt          # Dependencies
├── setup.sh                  # HF build step (spaCy en_core_web_sm)
├── sync.ps1                  # Commit + push to GitHub & HF
├── documentation.md          # This document
└── README.md
```

---

## 9. How to Run

### 9.1 Local Development

```bash
pip install -r requirements.txt
# optionally, for the spaCy model used by some paths:
python -m spacy download en_core_web_sm

export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
export OPENROUTER_API_KEY=your_key

streamlit run streamlit_app.py
```

### 9.2 Hugging Face Spaces

1. Add `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` as **secrets**
   in **Settings → Variables and secrets**.
2. Push to the Space (git push to the `hf` remote) — the Space rebuilds via
   `set up: setup.sh` and runs `streamlit_app.py`.

> **HF secret constraints:** secret **names** cannot contain underscores
> (`^[a-zA-Z][a-zA-Z0-9]*$`) and **values** are capped at 64 characters. If a
> key exceeds 64 chars, split it across multiple secrets and concatenate in
> code.

### 9.3 Syncing GitHub + HF

The repo has two remotes and `sync.ps1` pushes to both:

```powershell
.\sync.ps1 "commit message"
# equivalent:
git push origin main   # GitHub
git push hf main       # Hugging Face (triggers rebuild)
```

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Wikipedia first, DDG fallback | Wikipedia is free, authoritative, no API key, reliable on HF Spaces free tier |
| 3-tier LLM router | Resilience against provider outage/rate limits; OpenRouter as final emergency tier |
| `max` for entailment, `mean` for contradiction | One confirming snippet is sufficient; one noisy snippet shouldn't condemn a claim |
| Bidirectional NLI | Extra information in evidence causes false neutral in forward-only NLI |
| QA is auxiliary | LLMs repeat hallucinations consistently — consistency ≠ accuracy |
| BERTScore tie-breaker | NLI confused by paraphrasing/negation; BERTScore gives raw semantic overlap |
| Faithfulness rule | Decomposer must never correct errors — would verify the corrected (true) claim |
| LLM Judge for 0.35–0.90 | Middle-ground risk needs contextual reasoning beyond numerical fusion |
| XGBoost with heuristic fallback | Works with 8 features, SHAP-compatible, cold-start safe |

---

## 11. Evaluation Notes

The system ships with several test/analysis scripts:

- `comprehensive_test.py`, `trace_water.py` — end-to-end behavioral tests on
  known-factual and known-hallucinated examples (e.g., the "Water boils at
  100°C" and "Titanic" cases in the UI).
- `test_cos.py`, `test_nli.py`, `test_logic.py`, `test_run.py` — targeted
  checks of cosine filtering, NLI behavior, decision logic, and the full run.

These serve as a **reproducible smoke-test suite** for conference
demonstrations. For a formal evaluation you would additionally:
1. **Benchmark** against public hallucination datasets (HaluEval, FactCC,
   TruthfulQA, FEVER) and report precision/recall/F1 per verdict class.
2. **Measure** per-claim latency and end-to-end wall time on CPU.
3. **Ablate** the components (drop QA track, drop web track, remove
   bidirectional NLI) to quantify each contribution.
4. **Analyze** agreement between the heuristic and trained-XGBoost paths, and
   the frequency with which the LLM judge overrides the numerical score.

---

## 12. Known Limitations and Future Work

### Current Limitations

| Limitation | Impact |
|------------|--------|
| Wikipedia coverage | Obscure/recent topics may lack Wikipedia articles |
| CPU inference latency | ~25–40s per claim on CPU |
| QA self-consistency flaw | LLMs can consistently repeat hallucinations |
| English only | Non-English claims fail or produce poor results |
| XGBoost cold-start | Heuristic surrogate is less accurate than a trained model |
| Input length | Very long texts are capped by `max_claims` |

### Planned Improvements

1. **Train XGBoost on labeled data** — HaluEval + FactCC labeling pipeline.
2. **Wikidata integration** — structured SPARQL slot-level verification.
3. **Multilingual support** — mDeBERTa for non-English claims.
4. **Batch NLI inference** — reduce per-claim latency.
5. **Confidence calibration** — Platt scaling on XGBoost outputs.
6. **Benchmark harness** — automated evaluation on public hallucination
   benchmarks with published metrics.

---

## 13. References

1. Kryściński, W., McCann, B., Xiong, C., & Socher, R. (2020). *Evaluating the
   Factual Consistency of Abstractive Text Summarization.* EMNLP.
2. Manakul, P., Liusie, A., & Gales, M. (2023). *SelfCheckGPT: Zero-Resource
   Black-Box Hallucination Detection for Generative Large Language Models.*
   EMNLP.
3. Wang, A., Cho, K., & Lewis, M. (2020). *Asking and Answering Questions to
   Evaluate the Factual Consistency of Summaries.* ACL.
4. Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). *HaluEval:
   A Large-Scale Hallucination Evaluation Benchmark for Large Language
   Models.* EMNLP.
5. He, P., Gao, J., & Chen, W. (2021). *DeBERTaV3: Improving DeBERTa using
   ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.*
   ICLR.
6. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks.* EMNLP.
7. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020).
   *BERTScore: Evaluating Text Generation with BERT.* ICLR.
8. Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018).
   *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL.
9. Lin, S., Hilton, J., & Evans, O. (2022). *TruthfulQA: Measuring How Models
   Mimic Human Falsehoods.* ACL.
10. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting
    Model Predictions.* NeurIPS.

---

## 14. Presenter Notes / Talking Points

- **The problem:** hallucination is the single biggest barrier to trustworthy
  LLM deployment; it cannot be solved by "more training" alone.
- **The approach:** evidence-grounded verification (external) + self-consistency
  (internal) fused with explainability — a defense-in-depth strategy.
- **Novelty:** integration of *both* verification paradigms in one
  CPU-deployable, explainable, auditable pipeline; resilient 3-provider LLM
  routing; transparent heuristic fallback so behavior is always interpretable.
- **Explainability:** every verdict ships a SHAP chart; reviewers can see
  *why*.
- **Live demo:** run the Titanic (hallucinated), Water (factual), and
  Apollo-11 (mixed) examples in the deployed Space.
