"""
Stage 2b: QA Consistency Checker
Stages B (QGen), C (LLM Re-query), D (NLI + BERTScore scoring).

Fixes applied:
  1. Question generation is claim-grounded (LLM-first) to prevent off-topic
     questions that caused false hallucination flags.
  2. Replaced google/electra-base-discriminator (token-level RTD model, not
     suitable for entailment) with cross-encoder/nli-deberta-v3-small, a
     proper NLI model that outputs entailment / neutral / contradiction logits.
     The support score is now p(entailment) - p(contradiction), which is
     semantically meaningful and directly comparable to the 0-1 thresholds.
"""

import re
import os
import numpy as np
import torch
from bert_score import score as bert_score_fn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from pipeline.claim_decomposer import call_llm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── QGen model ───────────────────────────────────────────────
QG_MODEL_NAME = "valhalla/t5-base-qg-hl"
_qg_tokenizer = None
_qg_model     = None

def _load_qg():
    global _qg_tokenizer, _qg_model
    if _qg_model is None:
        _qg_tokenizer = T5Tokenizer.from_pretrained(QG_MODEL_NAME)
        _qg_model     = T5ForConditionalGeneration.from_pretrained(QG_MODEL_NAME).to(device)
        _qg_model.eval()

# ── NLI model (replaces ELECTRA discriminator) ───────────────
# cross-encoder/nli-deberta-v3-small:
#   • Fine-tuned on SNLI + MultiNLI for 3-class NLI
#   • Label order: contradiction=0, entailment=1, neutral=2
#   • ~184 M params, fast on CPU, excellent on short claim/answer pairs
NLI_MODEL_REPO = os.environ.get(
    "NLI_MODEL_REPO", "cross-encoder/nli-deberta-v3-small"
)
_NLI_LABEL_MAP: dict[str, int] = {}   # populated on first load
_nli_tokenizer = None
_nli_model     = None

def _load_nli():
    global _nli_tokenizer, _nli_model, _NLI_LABEL_MAP
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_REPO)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_REPO
        ).to(device)
        _nli_model.eval()
        # Build label→index map from the model's own config so we never
        # hard-code an order that might differ across checkpoints.
        _NLI_LABEL_MAP = {
            v.lower(): int(k)
            for k, v in _nli_model.config.id2label.items()
        }

def load_models():
    """Pre-load all models at startup."""
    _load_qg()
    _load_nli()


# ── Stage B: Question Generation ────────────────────────────

# Prompt for LLM-based question generation, used as the primary strategy.
# Questions are generated *from the claim itself*, so the LLM re-query
# in Stage C must answer about the same subject and predicate.
_QGEN_PROMPT = """\
You are building a fact-checking pipeline. Given a factual claim, generate \
{n} short, specific questions whose answers would directly verify or refute \
that exact claim.

Rules:
- Every question MUST name the specific subject from the claim (e.g. "Marie Curie", not "Who").
- Questions should target a single verifiable fact each.
- Do NOT ask yes/no questions — ask Wh- questions that force a factual answer.
- Return ONLY a numbered list, one question per line, no extra text.

Claim: {claim}
Questions:"""


def _generate_questions_via_llm(claim: str, n: int = 2) -> list[str]:
    """
    Primary strategy: ask the LLM to write grounded questions for this claim.
    Returns up to `n` non-empty question strings.
    """
    prompt   = _QGEN_PROMPT.format(claim=claim, n=n)
    raw      = call_llm(prompt) or ""
    questions = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if line and line.endswith("?"):
            questions.append(line)
        if len(questions) >= n:
            break
    return questions


def _extract_answer_spans(claim: str) -> list[str]:
    """
    Fallback span extractor used only when the LLM question generator fails.
    Returns up to 3 spans, prioritising named entities over bare dates.
    """
    spans = []

    # Multi-word capitalized phrases (named entities like "Marie Curie")
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim):
        t = m.group().strip()
        if t not in spans:
            spans.append(t)

    # Dates: "in 1903", "1921", "19th century"
    for m in re.finditer(r'\b(in\s+)?\d{4}\b|\b\d+(?:st|nd|rd|th)\s+century\b', claim):
        t = m.group().strip()
        if t not in spans:
            spans.append(t)

    # Single capitalized words (org/place names), skip common stopwords
    _SKIP = {'The', 'A', 'An', 'He', 'She', 'It', 'They', 'His', 'Her', 'Its'}
    for m in re.finditer(r'\b([A-Z][a-zA-Z]{3,})\b', claim):
        t = m.group().strip()
        if t not in spans and t not in _SKIP:
            spans.append(t)

    return spans[:3]


def _generate_question_for_span(claim: str, span: str) -> str:
    """
    Fallback: use T5-QG to generate a question by highlighting `span` inside
    `claim`.  The full claim is passed as context so T5 sees the subject.
    """
    _load_qg()
    highlighted = claim.replace(span, f"<hl> {span} <hl>", 1)
    # Prepend the full claim as context so T5 doesn't ignore the subject
    input_text  = f"generate question: context: {claim} question about: {highlighted}"
    inputs = _qg_tokenizer(
        input_text, return_tensors="pt", max_length=256, truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = _qg_model.generate(
            **inputs, max_length=64, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=2,
        )
    return _qg_tokenizer.decode(outputs[0], skip_special_tokens=True)


def _subject_is_present(question: str, claim: str) -> bool:
    """
    Heuristic guard: at least one capitalised token from the claim must
    appear in the generated question so we don't accept generic questions
    like "Who was a physicist?".
    """
    claim_tokens = set(re.findall(r'\b[A-Z][a-z]{2,}\b', claim))
    question_tokens = set(re.findall(r'\b[A-Z][a-z]{2,}\b', question))
    return bool(claim_tokens & question_tokens)


def generate_questions(claim: str) -> list[str]:
    """
    Build 2 grounded verification questions for `claim`.

    Strategy (in order of preference):
      1. LLM-generated questions  — most reliable, always claim-specific.
      2. T5-QG with claim context — good when span is a named entity.
      3. Hard-coded fallback       — guarantees at least one usable question.
    """
    questions: list[str] = []

    # 1. LLM generation (primary)
    try:
        llm_qs = _generate_questions_via_llm(claim, n=2)
        for q in llm_qs:
            if _subject_is_present(q, claim) and q not in questions:
                questions.append(q)
    except Exception:
        pass

    # 2. T5-QG fallback for any missing slots
    if len(questions) < 2:
        spans = _extract_answer_spans(claim)
        for span in spans:
            if len(questions) >= 2:
                break
            try:
                q = _generate_question_for_span(claim, span)
                if q and _subject_is_present(q, claim) and q not in questions:
                    questions.append(q)
            except Exception:
                pass

    # 3. Hard-coded fallback — always claim-specific by construction
    if not questions:
        questions.append(f"Is it true that {claim.rstrip('.')}?")

    return questions[:2]


# ── Stage C: LLM Re-query ────────────────────────────────────
ANSWER_PROMPT = """\
You are a fact-checker. Answer the following question in exactly one concise \
factual sentence. The answer must directly address the claim topic. Do not \
answer with a different entity if the question is about a specific one.

Question: {question}
Answer:"""

def _clean_llm_answer(answer: str, max_words: int = 60) -> str:
    if not answer:
        return ""
    phrases = [p.strip() for p in answer.split(",") if p.strip()]
    if len(phrases) > 6:
        phrase_counts = {}
        for p in phrases:
            phrase_counts[p] = phrase_counts.get(p, 0) + 1
        if max(phrase_counts.values()) >= 3:
            seen, clean_phrases = set(), []
            for p in phrases:
                if p in seen:
                    break
                seen.add(p)
                clean_phrases.append(p)
            answer = ", ".join(clean_phrases[:3])
    first_sent = answer.split(".")[0].strip()
    if first_sent:
        answer = first_sent + "."
    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words]) + "..."
    return answer.strip()

def requery_llm(question: str, num_samples: int = 3) -> list:
    prompt = ANSWER_PROMPT.format(question=question)
    answers = []
    for _ in range(num_samples):
        # sample at higher temp for variance
        ans = call_llm(prompt, temperature=0.7)
        clean = _clean_llm_answer(ans)
        if clean:
            answers.append(clean)
    # Deduplicate exact matches to avoid unnecessary NLI calls
    unique_answers = []
    for a in answers:
        if a not in unique_answers:
            unique_answers.append(a)
    return unique_answers if unique_answers else [""]


# ── Stage D: Scoring ─────────────────────────────────────────
THRESHOLD_WITH_CONTRADICTION    = 0.50
THRESHOLD_WITHOUT_CONTRADICTION = 0.40

def _compute_bertscore(claim: str, answer: str) -> float:
    try:
        _, _, F1 = bert_score_fn(
            cands=[answer], refs=[claim],
            lang="en", model_type="distilbert-base-uncased",
            verbose=False, device=device,
        )
        return float(F1[0])
    except Exception:
        return 0.5

def _detect_explicit_contradiction(claim: str, answer: str) -> float:
    claim_lower  = claim.lower()
    answer_lower = answer.lower()
    penalty = 0.0

    not_pattern = re.compile(r"not (?:specifically )?for ([a-z\s]+)")
    for match in not_pattern.finditer(answer_lower):
        if match.group(1).strip() in claim_lower:
            penalty = max(penalty, 0.8)

    negation_phrases = [
        "but not", "however not", "was not", "did not",
        "is incorrect", "is wrong", "is false",
    ]
    for phrase in negation_phrases:
        if phrase in answer_lower:
            penalty = max(penalty, 0.5)

    return penalty

def _nli_single_direction(text_a: str, text_b: str) -> dict:
    """Single-direction NLI inference."""
    _load_nli()
    enc = _nli_tokenizer(
        text_a, text_b,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)
    with torch.no_grad():
        logits = _nli_model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)[0]

    idx_ent = _NLI_LABEL_MAP.get("entailment", 1)
    idx_con = _NLI_LABEL_MAP.get("contradiction", 0)
    idx_neu = _NLI_LABEL_MAP.get("neutral", 2)

    return {
        "entailment":    float(probs[idx_ent]),
        "contradiction": float(probs[idx_con]),
        "neutral":       float(probs[idx_neu]),
    }


def _structured_contradiction_qa(claim: str, answer: str) -> float:
    """Detect entity/date/number mismatches between claim and answer."""
    boost = 0.0

    # Year mismatch
    claim_years = set(re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', claim))
    answer_years = set(re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', answer))
    if claim_years and answer_years and claim_years.isdisjoint(answer_years):
        boost += 0.4

    return min(boost, 0.7)


def _score_question_answers(claim: str, answers: list) -> dict:
    """Score a claim against multiple sampled answers for entropy."""
    _load_nli()
    
    if not answers or not answers[0]:
        return {"electra_score": 0.0, "bertscore_f1": 0.5, "contradiction": 0.0, "final_score": 0.5, "entropy_penalty": 0.0, "primary_answer": ""}

    # Score each answer individually
    individual_scores = []
    for answer in answers:
        forward  = _nli_single_direction(answer, claim)
        backward = _nli_single_direction(claim, answer)
        p_ent = max(forward["entailment"], backward["entailment"])
        p_con = max(forward["contradiction"], backward["contradiction"])
        struct_boost = _structured_contradiction_qa(claim, answer)
        penalty  = _detect_explicit_contradiction(claim, answer)
        bs_score = _compute_bertscore(claim, answer)
        
        individual_scores.append({
            "ent": p_ent, "con": p_con, "struct": struct_boost, 
            "penalty": penalty, "bs": bs_score, "answer": answer
        })
        
    # Check entropy / variance between sampled answers
    ents = [s["ent"] for s in individual_scores]
    cons = [s["con"] for s in individual_scores]
    
    avg_ent = float(np.mean(ents))
    max_con = float(np.max(cons))
    max_struct = float(np.max([s["struct"] for s in individual_scores]))
    max_penalty = float(np.max([s["penalty"] for s in individual_scores]))
    avg_bs = float(np.mean([s["bs"] for s in individual_scores]))
    
    # Entropy penalty: variance in entailment across samples
    # Threshold of 0.15 prevents minor LLM rephrasing from triggering false positives
    entropy_penalty = 0.0
    if len(individual_scores) > 1:
        ent_variance = float(np.var(ents))
        if ent_variance > 0.15:  # High variance = LLM is genuinely inconsistent
            entropy_penalty += 0.3

    # Base support: best of NLI entailment or BERTScore semantic overlap
    base_score = max(avg_ent, avg_bs)

    # NLI contradiction penalty: only penalise BEYOND what BERTScore confirms.
    # If BERTScore says answer is semantically close (e.g. 0.85) but NLI says
    # contradiction (e.g. 0.99), the NLI model is likely confused by truncation
    # or paraphrasing. We penalise only the excess (0.99 - 0.85 = 0.14).
    nli_con_penalty = max(0.0, max_con - avg_bs)
    total_penalty   = max(nli_con_penalty, max_penalty, max_struct)

    raw_score   = base_score - total_penalty - entropy_penalty
    final_score = max(0.0, min(1.0, raw_score))

    # Keep nli_support for UI compatibility
    nli_support = avg_ent - max_con - max_struct - entropy_penalty

    return {
        "electra_score" : round(nli_support, 3),
        "bertscore_f1"  : round(avg_bs, 3),
        "contradiction" : round(max(max_penalty, max_struct), 3),
        "entropy_penalty": round(entropy_penalty, 3),
        "final_score"   : round(final_score, 3),
        "primary_answer": individual_scores[0]["answer"]
    }



def score_claim_qa(claim: str, questions: list, answers_list: list) -> dict:
    pair_scores = []
    for q, a_list in zip(questions, answers_list):
        scores             = _score_question_answers(claim, a_list)
        scores["question"] = q
        scores["answer"]   = scores.pop("primary_answer")
        pair_scores.append(scores)

    avg_electra = float(np.mean([s["electra_score"] for s in pair_scores]))
    avg_bert    = float(np.mean([s["bertscore_f1"]  for s in pair_scores]))
    avg_final   = float(np.mean([s["final_score"]   for s in pair_scores]))
    max_penalty = float(max([s["contradiction"]      for s in pair_scores]))
    max_entropy = float(max([s.get("entropy_penalty", 0.0) for s in pair_scores]))

    contradiction_fired = max_penalty > 0.3 or max_entropy > 0.2
    threshold = (
        THRESHOLD_WITH_CONTRADICTION if contradiction_fired
        else THRESHOLD_WITHOUT_CONTRADICTION
    )
    label      = "FACTUAL" if avg_final >= threshold else "HALLUCINATED"
    confidence = round(min(abs(avg_final - threshold) * 2, 1.0), 3)

    return {
        "claim"             : claim,
        "consistency_score" : round(avg_final, 3),
        "electra_avg"       : round(avg_electra, 3),
        "bertscore_avg"     : round(avg_bert, 3),
        "max_contradiction" : round(max_penalty, 3),
        "max_entropy"       : round(max_entropy, 3),
        "threshold_used"    : threshold,
        "label"             : label,
        "confidence"        : confidence,
        "qa_pairs"          : pair_scores,
    }