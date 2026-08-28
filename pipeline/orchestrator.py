"""
Full pipeline orchestrator.
Wires Stage 2a (Web+NLI), Stage 2b (QA Consistency), Stage 3 (Meta-Classifier).
"""

from pipeline.claim_decomposer import decompose_claims
from pipeline.qa_checker        import generate_questions, requery_llm, score_claim_qa
from pipeline.web_nli           import run_web_nli
from pipeline.meta_classifier   import fuse_signals, aggregate_results
import time


def run_pipeline(llm_response: str, progress_cb=None, max_claims: int = 5) -> dict:
    """
    Full hallucination detection pipeline.

    Args:
        llm_response : The LLM-generated text to verify.
        progress_cb  : Optional callable(stage: str, pct: int) for UI progress.
        max_claims   : Maximum atomic claims to extract and verify (default 5).

    Returns:
        Full structured report dict.
    """
    def _progress(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    _progress("Decomposing claims...", 5)
    claims = decompose_claims(llm_response, max_claims=max_claims)

    if not claims:
        return {
            "input_response"          : llm_response,
            "verdict"                 : "UNABLE TO VERIFY",
            "overall_risk"            : 0.5,
            "total_claims"            : 0,
            "confirmed_hallucinations": 0,
            "uncertain_claims"        : 0,
            "factual_claims"          : 0,
            "claim_results"           : [],
        }

    claim_results = []
    n = len(claims)

    for i, claim in enumerate(claims):
        base_pct = int(5 + (i / n) * 90)

        # Delay between claims to avoid DDG rate limiting
        if i > 0:
            time.sleep(1.5)

        # Stage 2a

        _progress(f"[{i+1}/{n}] Web search + NLI: {claim[:60]}...", base_pct)
        web_result = run_web_nli(claim)

        # Stage 2b
        _progress(f"[{i+1}/{n}] QA consistency check...", base_pct + 5)
        questions  = generate_questions(claim)
        answers    = [requery_llm(q) for q in questions]
        qa_result  = score_claim_qa(claim, questions, answers)

        # Stage 3
        _progress(f"[{i+1}/{n}] Meta-classifier fusion...", base_pct + 8)
        meta       = fuse_signals(web_result, qa_result)

        claim_results.append({
            "claim"      : claim,
            "web_nli"    : web_result,
            "qa"         : qa_result,
            "meta"       : meta,
        })

    _progress("Aggregating results...", 97)
    summary = aggregate_results(claim_results)

    _progress("Done.", 100)

    return {
        "input_response"          : llm_response,
        "verdict"                 : summary["verdict"],
        "overall_risk"            : summary["overall_risk"],
        "total_claims"            : summary["total_claims"],
        "confirmed_hallucinations": summary["confirmed_hallucinations"],
        "uncertain_claims"        : summary["uncertain_claims"],
        "factual_claims"          : summary["factual_claims"],
        "claim_results"           : claim_results,
    }
