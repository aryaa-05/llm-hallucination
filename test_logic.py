def _score_question_answers(individual_scores: list) -> dict:
    ents = [s["entailment"] for s in individual_scores]
    cons = [s["contradiction"] for s in individual_scores]

    avg_ent = float(sum(ents) / len(ents))
    max_con = float(max(cons))
    max_struct = float(max([s.get("struct", 0.0) for s in individual_scores]))
    max_penalty = float(max([s.get("penalty", 0.0) for s in individual_scores]))
    avg_bs = float(sum([s["bs"] for s in individual_scores]) / len(individual_scores))
    
    entropy_penalty = 0.0
    if len(individual_scores) > 1:
        mean_ent = avg_ent
        ent_variance = sum((x - mean_ent) ** 2 for x in ents) / len(ents)
        if ent_variance > 0.05:
            entropy_penalty += 0.3

    base_score = max(avg_ent, avg_bs)
    total_penalty = max(max_con, max_penalty, max_struct)
    
    raw_score = base_score - total_penalty - entropy_penalty
    final_score = max(0.0, min(1.0, raw_score))

    nli_support = avg_ent - max_con - max_struct - entropy_penalty

    return {
        "electra_score" : round(nli_support, 3),
        "bertscore_f1"  : round(avg_bs, 3),
        "contradiction" : round(max_con, 3),
        "entropy_penalty": round(entropy_penalty, 3),
        "final_score"   : round(final_score, 3)
    }

print("--- Testing Claim 1 (Marie Curie Partial Match) ---")
scores_1 = [
    {"entailment": 0.002, "contradiction": 0.0, "bs": 0.798},
    {"entailment": 0.195, "contradiction": 0.0, "bs": 0.798}
]
res1 = _score_question_answers(scores_1)
print(res1)
assert res1["final_score"] > 0.4, "Failed to use BERTScore for base support!"

print("\n--- Testing Claim 2 (Einstein Contradiction) ---")
scores_2 = [
    {"entailment": 0.0, "contradiction": 0.99, "bs": 0.600}
]
res2 = _score_question_answers(scores_2)
print(res2)
assert res2["final_score"] == 0.0, "Failed to penalize contradiction!"

print("\n--- Testing Claim 3 (Marie Curie Factual) ---")
scores_3 = [
    {"entailment": 0.997, "contradiction": 0.0, "bs": 0.947},
    {"entailment": 0.997, "contradiction": 0.0, "bs": 1.0}
]
res3 = _score_question_answers(scores_3)
print(res3)
assert res3["final_score"] > 0.9, "Failed on perfect match!"

print("\nAll logic tests passed successfully!")
