import sys
import json
from agent.strands import Agent
from agent.tools import (
    run_adversarial_fgsm,
    run_subgroup_fairness_audit,
    verify_biomarker_consensus,
    run_baseline_inference
)
from backend.cedar_gate import evaluate_cedar_gate

def run_audit(model_id: str):
    print(f"=====================================================")
    print(f" AWS Strands Autonomous Red-Teaming Agent Initialized")
    print(f" Target Model: {model_id}")
    print(f"=====================================================")

    tools = [
        run_baseline_inference,
        run_adversarial_fgsm,
        run_subgroup_fairness_audit,
        verify_biomarker_consensus
    ]
    
    # We are using "qwen2.5:7b" by default, assuming Ollama is running it. 
    # If the user has a different model, they can change it here.
    agent = Agent(model="qwen2.5:7b", tools=tools)

    prompt = f"""
You are an autonomous AWS Strands Red-Teaming Agent.
Your task is to audit the clinical candidate model: '{model_id}'

Follow this exact loop:
1. Run Baseline Inference.
2. Run targeted Adversarial Stress testing with FGSM (epsilon=0.05).
3. Check Subgroup Fairness (Equalized Odds).
4. Verify Biomarker Consensus (to check for Silent False Negatives).
5. Synthesize all collected metrics into a single pure JSON string and output ONLY the JSON string. Do not output markdown, just the raw JSON dict. Example format: {{"sensitivity": 95, "subgroup_fnr_disparity": 5, ...}}
"""

    print("Agent is reasoning... (this may take a few moments depending on local LLM speed)")
    try:
        final_report = agent.run(prompt)
        print("\n[Agent Metrics Payload]:")
        print(final_report)
        
        # Phase 2: Decoupled Deterministic Evaluator (Judge)
        print("\n\n" + "="*50)
        print(" PHASE 2: CEDAR POLICY GATE EVALUATION")
        print("="*50)
        
        try:
            metrics_payload = json.loads(final_report.strip("```json\n "))
            gate_result = evaluate_cedar_gate(metrics_payload, department="EmergencyICU")
            print(json.dumps(gate_result, indent=2))
        except json.JSONDecodeError:
            print("ERROR: Agent failed to output valid JSON. Cannot evaluate Cedar Gate.")

    except Exception as e:
        print(f"\n[Agent Error] Could not connect to local Ollama server or execution failed: {e}")
        print("Please ensure `ollama serve` is running and the 'qwen2.5:7b' model is pulled.")

if __name__ == "__main__":
    model_to_test = sys.argv[1] if len(sys.argv) > 1 else "dr_retinal_lcnet_edge.onnx"
    run_audit(model_to_test)
