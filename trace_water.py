"""
Trace through the exact water claim pipeline with the fixed logic
using the real numbers from the JSON output provided by the user.
"""
import numpy as np

# ── Snippet scores from the real JSON ────────────────────────
snippet_scores = [
    {"entailment": 0.0,   "neutral": 0.997, "contradiction": 0.002, "structured_contradiction": 0.0},
    {"entailment": 0.0,   "neutral": 0.003, "contradiction": 0.997, "structured_contradiction": 0.0},
]

# ── OLD aggregation ───────────────────────────────────────────
old_ent  = float(np.max([s["entailment"]    for s in snippet_scores]))
old_con  = float(np.max([s["contradiction"] for s in snippet_scores]))
print(f"OLD: entailment={old_ent}, contradiction={old_con}")

# ── NEW aggregation (mean for contradiction) ──────────────────
new_ent  = float(np.max([s["entailment"]    for s in snippet_scores]))  # max stays
new_con  = float(np.mean([s["contradiction"] for s in snippet_scores])) # mean now
new_struct = float(np.max([s["structured_contradiction"] for s in snippet_scores]))
new_final_con = min(1.0, new_con + 0.5 * new_struct)
print(f"NEW: entailment={new_ent}, contradiction(mean)={new_con:.3f}, final_con={new_final_con:.3f}")

# ── QA values (updated with fixed NLI penalty in qa_checker) ─
# OLD: QA pair 2 scored 0.0 because max_con (0.99) > avg_bs (0.851)
# NEW: nli_con_penalty = max(0, max_con - avg_bs) = max(0, 0.99 - 0.851) = 0.139
#      base = max(avg_ent, avg_bs) = 0.851
#      final = 0.851 - 0.139 = 0.712

qa_pair1_final = 0.998
qa_pair2_final_old = 0.0
qa_pair2_final_new = 0.712

consistency_score_old = (qa_pair1_final + qa_pair2_final_old) / 2
consistency_score_new = (qa_pair1_final + qa_pair2_final_new) / 2
print(f"\nOLD QA consistency: {consistency_score_old:.3f}")
print(f"NEW QA consistency: {consistency_score_new:.3f}")

# ── Meta-classifier heuristic with new values ─────────────────
features_new = {
    "entailment_score":       new_ent,
    "contradiction_score":    new_final_con,
    "num_snippets":           2.0,
    "consistency_score":      consistency_score_new,
    "electra_avg":            0.008,
    "bertscore_avg":          0.875,
    "retrieval_confidence":   0.7,
    "structured_contradiction": new_struct,
}

has_web  = True
ent      = features_new["entailment_score"]
con      = features_new["contradiction_score"]
raw_con  = con - 0.5 * features_new["structured_contradiction"]
qa       = features_new["consistency_score"]
qa_nli   = features_new["electra_avg"]
bert     = features_new["bertscore_avg"]
struct_con = features_new["structured_contradiction"]

print(f"\n── Meta-classifier trace ──")
print(f"ent={ent}, con(mean+boost)={con:.3f}, raw_con={raw_con:.3f}")
print(f"qa={qa:.3f}, bert={bert}, struct_con={struct_con}")

# B0: BERTScore override
if has_web and bert > 0.80 and ent < 0.30 and con < 0.65:
    factuality = float(np.clip(0.15 * qa + 0.65 * bert + 0.20 * max(qa_nli, 0), 0.0, 1.0))
    risk = 1.0 - factuality
    print(f"\n→ B0 fires! factuality={factuality:.3f}, risk={risk:.3f}")
    print(f"  Verdict: {'LIKELY FACTUAL' if risk < 0.35 else 'UNCERTAIN' if risk < 0.70 else 'HALLUCINATION'}")
else:
    print("\nB0 did not fire, checking B1/B2...")
    if has_web and ent > 0.65:
        factuality = float(np.clip(0.75 * ent + 0.25 * qa, 0.0, 1.0))
        print(f"→ B1 fires! factuality={factuality:.3f}")
    elif has_web and raw_con > 0.65:
        factuality = float(np.clip(0.15 * qa, 0.0, 1.0))
        print(f"→ B2 fires! factuality={factuality:.3f}")
    else:
        factuality = float(np.clip(0.4 * ent + 0.35 * qa + 0.25 * max(qa_nli, 0), 0.0, 1.0))
        print(f"→ Fallback! factuality={factuality:.3f}")
    risk = 1.0 - factuality
    print(f"  risk={risk:.3f}")
    print(f"  Verdict: {'LIKELY FACTUAL' if risk < 0.35 else 'UNCERTAIN' if risk < 0.70 else 'HALLUCINATION'}")
