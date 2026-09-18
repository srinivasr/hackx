from agent.strands import Agent
from agent.tools import (
    run_adversarial_fgsm,
    run_subgroup_fairness_audit,
    verify_biomarker_consensus
)

def test_agent_tool_registration():
    tools = [
        run_adversarial_fgsm,
        run_subgroup_fairness_audit,
        verify_biomarker_consensus
    ]
    agent = Agent(tools=tools)
    
    # Verify tools registered
    assert "run_adversarial_fgsm" in agent.tool_map
    
    # Verify schemas generated correctly
    schemas = agent._get_tool_schemas()
    assert len(schemas) == 3

def test_tool_execution():
    # Direct tool invocation to ensure mock returns valid data
    res1 = run_adversarial_fgsm(model_id="dr_retinal_lcnet_edge.onnx", epsilon=0.05)
    assert res1["status"] == "vulnerable"
    
    res2 = run_adversarial_fgsm(model_id="resnet_teacher.onnx", epsilon=0.05)
    assert res2["status"] == "robust"
