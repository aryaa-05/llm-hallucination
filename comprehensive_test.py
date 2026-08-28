import os
import json
import unittest.mock as mock

def test_pipeline():
    # Mocking LLM calls
    def mock_call_llm(prompt, temperature=0.0):
        # DECOMPOSER MOCKS
        if "Apollo 11" in prompt and "atomic" in prompt:
            return "1. The Apollo 11 mission was the first crewed moon landing.\n2. It was commanded by Neil Armstrong in 1969.\n3. They spent 12 days on the lunar surface.\n4. They discovered water ice."
        elif "Water boils" in prompt and "atomic" in prompt:
            return "1. Water boils at 100 degrees Celsius at sea level."
        elif "Abraham Lincoln" in prompt and "atomic" in prompt:
            return "1. Abraham Lincoln was the first President of the United States."
        
        # QA GENERATOR / ANSWER MOCKS
        elif "generate 2-3 specific questions" in prompt.lower():
            if "Apollo 11" in prompt:
                return "1. Who commanded Apollo 11?\n2. How long did they spend on the surface?"
            elif "Water boils" in prompt:
                return "1. At what temperature does water boil at sea level?"
            elif "Abraham Lincoln" in prompt:
                return "1. Who was the first President of the US?"
            else:
                return "1. Question 1?"
        else:
            if "Apollo" in prompt or "Armstrong" in prompt:
                return "Apollo 11 was commanded by Neil Armstrong. They spent about 21 hours on the lunar surface. Water ice was discovered by later missions, not Apollo 11."
            elif "Water" in prompt or "boils" in prompt:
                return "Water boils at 100 degrees Celsius (212 Fahrenheit) at sea level."
            elif "Lincoln" in prompt or "President" in prompt:
                return "George Washington was the first President of the United States. Abraham Lincoln was the 16th President."
            return "Generic answer."

    with mock.patch("pipeline.claim_decomposer.call_llm", side_effect=mock_call_llm):
        from pipeline.orchestrator import run_pipeline
        texts = [
            {
                "name": "Mixed Fact & Fiction (Long)",
                "text": "The Apollo 11 mission was the first crewed moon landing, commanded by Neil Armstrong in 1969. They spent 12 days on the lunar surface and discovered water ice."
            },
            {
                "name": "Factual (Short)",
                "text": "Water boils at 100 degrees Celsius at sea level."
            },
            {
                "name": "Hallucinated (Short)",
                "text": "Abraham Lincoln was the first President of the United States."
            }
        ]

        print("="*60)
        print("COMPREHENSIVE PIPELINE LOGIC TEST (MOCKED LLM)")
        print("="*60)

        for example in texts:
            print(f"\n--- Testing: {example['name']} ---")
            print(f"Input: {example['text']}")
            try:
                report = run_pipeline(example['text'])
                print(f"OVERALL VERDICT: {report['verdict']} (Risk: {report['overall_risk']:.3f})")
                print("Claims Breakdown:")
                for claim in report["claim_results"]:
                    print(f"  - '{claim['claim']}'")
                    print(f"    Verdict: {claim['meta']['verdict']} | Risk: {claim['meta']['hallucination_risk']:.3f} | Web: {claim['meta']['branch_weights']['web']:.2f}")
            except Exception as e:
                print("Error running pipeline:", e)

if __name__ == "__main__":
    test_pipeline()
