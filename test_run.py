import os
import json
import unittest.mock as mock

def test_pipeline():
    # Mocking LLM calls because API keys are not available
    def mock_call_llm(prompt, temp=0.0):
        if "Marie Curie" in prompt and "atomic" in prompt:
            return "1. Marie Curie was a Polish-French physicist and chemist.\n2. She won the Nobel Prize in Physics in 1903.\n3. She won the Nobel Prize in Chemistry in 1911.\n4. Marie Curie was the first person to win Nobel Prizes in two different sciences."
        elif "Einstein" in prompt and "atomic" in prompt:
            return "1. Albert Einstein was born in Germany in 1879.\n2. He won the Nobel Prize for his theory of general relativity in 1921.\n3. He later moved to the United States in 1933."
        elif "generate 2-3 specific questions" in prompt.lower():
            return "1. What was the subject's profession?\n2. What year did they win the prize?"
        else:
            return "Marie Curie was a Polish-French physicist. She won in 1903 and 1911. Albert Einstein was born in 1879 and won for the photoelectric effect in 1921."

    with mock.patch("pipeline.claim_decomposer.call_llm", side_effect=mock_call_llm):
        from pipeline.orchestrator import run_pipeline
        texts = [
            "Marie Curie was a Polish-French physicist and chemist. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911, making her the first person to win Nobel Prizes in two different sciences.",
            "Albert Einstein was born in Germany in 1879. He won the Nobel Prize for his theory of general relativity in 1921. He later moved to the United States in 1933."
        ]

        for i, text in enumerate(texts):
            print(f"\n--- Testing Example {i+1} ---")
            try:
                report = run_pipeline(text)
                print("Verdict:", report["verdict"])
                print("Overall Risk:", report["overall_risk"])
                for claim in report["claim_results"]:
                    print(f"- Claim: {claim['claim']}")
                    print(f"  Verdict: {claim['meta']['verdict']} (Risk: {claim['meta']['hallucination_risk']})")
            except Exception as e:
                print("Error running pipeline:", e)

if __name__ == "__main__":
    test_pipeline()
