import inspect
import json
from typing import Callable, Any, Dict, List
from openai import OpenAI

class Agent:
    """Mock AWS Strands Agent backed by local Ollama API for tool execution."""
    def __init__(self, model: str = "llama3.1", tools: List[Callable] = None):
        self.model = model
        self.tools = tools or []
        # Connect to local Ollama instance running in OpenAI-compatible mode
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama" # required but ignored by ollama
        )
        self.tool_map = {t.__name__: t for t in self.tools}

    def _get_tool_schemas(self):
        schemas = []
        for name, func in self.tool_map.items():
            doc = inspect.getdoc(func) or f"Tool {name}"
            sig = inspect.signature(func)
            
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                
                properties[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                    
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": doc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas

    def run(self, prompt: str, max_iterations: int = 5) -> str:
        messages = [{"role": "user", "content": prompt}]
        tools = self._get_tool_schemas()
        
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
            )
            
            msg = response.choices[0].message
            # Append assistant message (if it has tool_calls, we must include them exactly as returned)
            assistant_msg = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
                print(f"[Strands Agent Reasoning]: {msg.content}")
            if msg.tool_calls:
                assistant_msg["tool_calls"] = []
                for tc in msg.tool_calls:
                    assistant_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            messages.append(assistant_msg)
            
            if not msg.tool_calls:
                return msg.content or ""
                
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                    
                    print(f"[Strands Tool Execution] => {name}({args})")
                    if name in self.tool_map:
                        try:
                            result = self.tool_map[name](**args)
                            result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                        except Exception as e:
                            result_str = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                    else:
                        result_str = json.dumps({"error": f"Tool {name} not found."})
                except json.JSONDecodeError:
                    result_str = json.dumps({"error": "Invalid tool schema provided. Arguments must be valid JSON."})
                    print(f"[Strands Tool Execution Error] => Bad JSON for {name}")
                
                print(f"  Result: {result_str}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result_str
                })
        
        # If we exit the loop, we hit max_iterations
        print(f"[Strands Agent Error] Halted after reaching max_iterations ({max_iterations}) without final response.")
        return "ERROR: Agent halted due to max_iterations budget exceeded."

def tool(func: Callable) -> Callable:
    """Mock decorator for Strands SDK tool annotation."""
    return func
