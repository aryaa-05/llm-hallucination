"""
Stage 3: Meta-Classifier
XGBoost signal fusion from Stage 2a (Web+NLI) and Stage 2b (QA Consistency).
Dynamic branch weighting based on snippet availability.
SHAP explanations on output.
Trained on HaluEval + FactCC — or falls back to a heuristic ensemble
if no trained model is found (for cold-start on HF Spaces).
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
from pipeline.claim_decomposer import call_llm

# XGBoost
import xgboost as xgb

# SHAP
import shap

# ── Feature names (order matters for XGBoost) ────────────────
FEATURE_NAMES = [
    "entailment_score",       # Stage 2a
    "contradiction_score",    # Stage 2a
    "neutral_score",          # Stage 2a
    "num_snippets",           # Stage 2a (proxy for verifiability)
    "consistency_score",      # Stage 2b
    "electra_avg",            # Stage 2b
    "bertscore_avg",          # Stage 2b
    "max_contradiction_qa",   # Stage 2b
]

MODEL_PATH = Path(os.environ.get("MODEL_DIR", "models")) / "meta_classifier.pkl"


# ── Heuristic fallback (no trained model) ────────────────────
def _heuristic_score(features: dict) -> float:
    """
    Hierarchical decision scoring.
    Returns factuality score [0, 1] where 1 = certainly factual.
    """
    has_web    = features.get("num_snippets", 0) > 0
    ent        = features["entailment_score"]
    con        = features["contradiction_score"]   # already mildly struct-boosted
    raw_con    = con - 0.5 * features.get("structured_contradiction", 0.0)  # un-boosted NLI contradiction
    qa         = features["consistency_score"]
    qa_nli     = features["electra_avg"]
    struct_con = features.get("structured_contradiction", 0.0)

    # ── Stage A: Retrieval quality gate ──────────────────
    if not has_web and qa < 0.4:
        return 0.5

    # ── Stage B: Evidence agreement ─────────────────────
    # B0: BERTScore override — when web is confused but QA semantic similarity is high
    # This fires when Wikipedia returns historical/context snippets that confuse NLI
    # (e.g., pre-1745 reversed Celsius scale contradicting modern boiling point).
    # If BERTScore says the LLM's own answers are semantically close to the claim,
    # that is strong evidence the claim is factual.
    bert = features.get("bertscore_avg", 0.5)
    if has_web and bert > 0.80 and ent < 0.30 and con < 0.65:
        # Web confused (no clear entailment, mixed contradiction) but QA+BERTScore confident
        return float(np.clip(0.15 * qa + 0.65 * bert + 0.20 * max(qa_nli, 0), 0.0, 1.0))

    # B1: Strong web support
    if has_web and ent > 0.65:
        return float(np.clip(0.75 * ent + 0.25 * qa, 0.0, 1.0))

    # B2: Strong web contradiction — use RAW NLI score (not struct-inflated)
    # Threshold raised to 0.65 so mild structural mismatch doesn't trigger this
    if has_web and raw_con > 0.65:
        return float(np.clip(0.15 * qa, 0.0, 1.0))

    # B3: No web but strong QA agreement
    if not has_web and qa > 0.5 and qa_nli > 0.3:
        return float(np.clip(0.7 * qa + 0.3 * max(qa_nli, 0), 0.0, 1.0))

    # B4: No web and QA says contradiction
    if not has_web and qa < 0.25:
        return float(np.clip(qa, 0.0, 1.0))

    # ── Stage C: Structured contradiction override ────────────
    # Threshold raised to 0.35 — mild date overlaps no longer override the whole verdict
    if struct_con > 0.35:
        return float(np.clip(0.3 * qa - 0.3 * struct_con, 0.0, 1.0))

    # ── Fallback blend ─────────────────────────────────
    if has_web:
        return float(np.clip(0.4 * ent + 0.35 * qa + 0.25 * max(qa_nli, 0), 0.0, 1.0))
    else:
        return float(np.clip(0.6 * qa + 0.4 * max(qa_nli, 0), 0.0, 1.0))


# ── XGBoost model (trained) ───────────────────────────────────
_xgb_model    = None
_xgb_explainer = None

def _load_xgb():
    global _xgb_model, _xgb_explainer
    if _xgb_model is not None:
        return True
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _xgb_model = pickle.load(f)
        _xgb_explainer = shap.TreeExplainer(_xgb_model)
        return True
    return False


def _features_to_array(features: dict) -> np.ndarray:
    return np.array([[features.get(k, 0.0) for k in FEATURE_NAMES]], dtype=np.float32)


# ── SHAP explanation ─────────────────────────────────────────
def _heuristic_shap(features: dict) -> dict:
    """Approximate SHAP values for the heuristic model."""
    has_web = features.get("num_snippets", 0) > 0
    w_web   = 0.45 if has_web else 0.10
    w_qa    = 0.55 if has_web else 0.90

    contributions = {
        "entailment_score"     : -w_web * 0.7  * features["entailment_score"],
        "contradiction_score"  :  w_web * 0.5  * features["contradiction_score"],
        "neutral_score"        :  0.0,
        "num_snippets"         : -0.02 * min(features.get("num_snippets", 0), 5),
        "consistency_score"    : -w_qa  * 1.0  * features["consistency_score"],
        "electra_avg"          : -w_qa  * 0.3  * features["electra_avg"],
        "bertscore_avg"        : -w_qa  * 0.1  * features["bertscore_avg"],
        "max_contradiction_qa" :  0.15          * features.get("max_contradiction_qa", 0),
    }
    return contributions


# ── LLM-as-a-Judge (Fallback) ────────────────────────────────
def _llm_judge(claim: str, snippets: list) -> float:
    """Use LLM CoT to judge ambiguous claims. Returns a risk score."""
    context = "\n".join([f"- {s}" for s in snippets])
    prompt = f"""You are an expert fact-checker. Determine if the following claim is Supported, Contradicted, or Unverifiable based strictly on the provided evidence.

Evidence:
{context}

Claim: {claim}

Think step-by-step. End your response with exactly one of these labels on a new line: [SUPPORTED], [CONTRADICTED], [UNVERIFIABLE]."""
    
    try:
        response = call_llm(prompt, temperature=0.0).upper()
        if "[CONTRADICTED]" in response:
            return 0.85
        elif "[SUPPORTED]" in response:
            return 0.15
        else:
            return 0.50
    except:
        return 0.50

# ── Main entry point ─────────────────────────────────────────
def fuse_signals(web_nli_result: dict, qa_result: dict) -> dict:
    """
    Combine Stage 2a and Stage 2b outputs into a final hallucination risk score.
    Uses hierarchical decision logic with retrieval confidence awareness.
    """
    features = {
        "entailment_score"         : web_nli_result.get("entailment_score",    0.0),
        "contradiction_score"      : web_nli_result.get("contradiction_score", 0.0),
        "neutral_score"            : web_nli_result.get("neutral_score",       0.0),
        "num_snippets"             : float(web_nli_result.get("num_snippets",  0)),
        "consistency_score"        : qa_result.get("consistency_score",        0.5),
        "electra_avg"              : qa_result.get("electra_avg",             0.5),
        "bertscore_avg"            : qa_result.get("bertscore_avg",           0.5),
        "max_contradiction_qa"     : qa_result.get("max_contradiction",       0.0),
        "retrieval_confidence"     : web_nli_result.get("retrieval_confidence", 0.0),
        "structured_contradiction" : web_nli_result.get("structured_contradiction", 0.0),
    }

    has_web = features["num_snippets"] > 0

    # ── Score ────────────────────────────────────────────────
    if _load_xgb():
        X            = _features_to_array(features)
        risk_score   = float(_xgb_model.predict_proba(X)[0][1])
        shap_vals_raw = _xgb_explainer.shap_values(X)
        if isinstance(shap_vals_raw, list) and len(shap_vals_raw) > 1:
            shap_vals_raw = shap_vals_raw[1]
        elif isinstance(shap_vals_raw, list):
            shap_vals_raw = shap_vals_raw[0]
        shap_values  = dict(zip(FEATURE_NAMES, [float(v) for v in shap_vals_raw[0]]))
        model_used   = "xgboost"
    else:
        factuality   = _heuristic_score(features)
        risk_score   = 1.0 - factuality
        shap_values  = _heuristic_shap(features)
        model_used   = "heuristic"

    # ── Verdict — hierarchical, retrieval-aware ──────────────
    # Hybrid Fallback: If we have evidence, and the risk isn't an absolute glaring hallucination (>0.90)
    # or absolute factual (<0.35), ask the LLM judge to prevent false positives from noisy snippets.
    if 0.35 <= risk_score <= 0.90 and has_web:
        claim = qa_result.get("claim", "")
        snippets = [s["snippet"] for s in web_nli_result.get("snippet_scores", [])][:3]
        if claim and snippets:
            llm_risk = _llm_judge(claim, snippets)
            if llm_risk != 0.50: # If LLM made a confident decision
                risk_score = llm_risk
                model_used = model_used + "+llm_judge"

    if risk_score >= 0.70:
        verdict = "HALLUCINATION DETECTED"
    elif risk_score >= 0.35:
        # If we have no evidence at all, label as insufficient
        if not has_web and features["consistency_score"] < 0.4:
            verdict = "INSUFFICIENT EVIDENCE"
        else:
            verdict = "UNCERTAIN — NEEDS REVIEW"
    else:
        verdict = "LIKELY FACTUAL"

    confidence = round(min(abs(risk_score - 0.5) * 2, 1.0), 3)

    branch_weights = {
        "web" : 0.45 if has_web else 0.10,
        "qa"  : 0.55 if has_web else 0.90,
    }

    return {
        "hallucination_risk"      : round(risk_score, 3),
        "verdict"                 : verdict,
        "confidence"              : confidence,
        "shap_values"             : shap_values,
        "features"                : features,
        "branch_weights"          : branch_weights,
        "model_used"              : model_used,
        "retrieval_source"        : web_nli_result.get("retrieval_source", "none"),
        "retrieval_confidence"    : features.get("retrieval_confidence", 0.0),
    }





def aggregate_results(claim_results: list) -> dict:
    """
    Aggregate per-claim fusion results into a response-level verdict.
    """
    if not claim_results:
        return {
            "overall_risk"            : 0.5,
            "verdict"                 : "UNABLE TO VERIFY",
            "confirmed_hallucinations": 0,
            "uncertain_claims"        : 0,
            "factual_claims"          : 0,
            "total_claims"            : 0,
        }

    risks    = [r["meta"]["hallucination_risk"] for r in claim_results]
    overall  = round(float(np.mean(risks)), 3)

    confirmed    = sum(1 for r in claim_results if r["meta"]["verdict"] == "HALLUCINATION DETECTED")
    uncertain    = sum(1 for r in claim_results if r["meta"]["verdict"] == "UNCERTAIN — NEEDS REVIEW")
    factual      = len(claim_results) - confirmed - uncertain

    if confirmed > 0:
        verdict = "HALLUCINATION DETECTED"
    elif uncertain > 0:
        verdict = "UNCERTAIN — NEEDS REVIEW"
    else:
        verdict = "LIKELY FACTUAL"

    return {
        "overall_risk"            : overall,
        "verdict"                 : verdict,
        "confirmed_hallucinations": confirmed,
        "uncertain_claims"        : uncertain,
        "factual_claims"          : factual,
        "total_claims"            : len(claim_results),
    }
