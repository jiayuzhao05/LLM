# mcp_agent_llm.py
# 一个集成了LLM作为决策大脑的、更智能的MCP Agent。

import requests
import json
import uuid
import os
from openai import OpenAI


import re

def resolve_placeholders(value: str, initial_request_data: dict, workflow_context: dict):
    """
    通用占位符解析：支持以下形式：
      - {{steps.1.asset_id}} 或 {steps.1.asset_id}
      - {{steps[1].output.asset_id}} 或 {steps[1].output.asset_id}
      - {{request.employee_id}} 或 {request.employee_id}
    返回解析后的值，若无法解析则返回 None 或抛异常（可根据需求调整）。
    """
    if not isinstance(value, str):
        return value

    raw = value.strip()
    # 如果是双大括号形式，去掉最外层 "{{...}}"
    if raw.startswith("{{") and raw.endswith("}}"):
        placeholder = raw[2:-2].strip()
    # 如果是单大括号形式 { ... }
    elif raw.startswith("{") and raw.endswith("}"):
        placeholder = raw[1:-1].strip()
    else:
        # 既没有 { } 也没有 {{ }}, 视为普通字面值
        return value

    # 解析 request.xxx
    if placeholder.startswith("request."):
        _, req_key = placeholder.split('.', 1)
        return initial_request_data.get(req_key)

    # 解析 steps[...] 或 steps.N...
    if placeholder.startswith("steps"):
        # 提取索引 idx 与后续属性 rest
        m = re.match(r"^steps\[(\d+)\](?:\.(.*))?$", placeholder)
        if m:
            idx = int(m.group(1))
            rest = m.group(2) or ""
        else:
            m2 = re.match(r"^steps\.(\d+)(?:\.(.*))?$", placeholder)
            if m2:
                idx = int(m2.group(1))
                rest = m2.group(2) or ""
            else:
                # 无法解析成预期格式
                # print(f"[WARNING] 无法解析 steps 占位符: '{placeholder}'")
                return None

        # 检查索引范围
        steps_list = workflow_context.get("steps", [])
        if idx < 0 or idx >= len(steps_list):
            # print(f"[WARNING] 占位符引用了不存在的步骤 idx={idx}, 当前只有 {len(steps_list)} 步")
            return None

        prev_output = steps_list[idx].get("output", {})
        # 如果 rest 为空，直接返回整个 output
        if not rest:
            return prev_output

        # 如果 rest 以 "output." 开头，去掉
        if rest.startswith("output."):
            rest = rest[len("output."):]
        # 链式取值
        keys = rest.split('.')
        resolved = prev_output
        for subkey in keys:
            if isinstance(resolved, dict):
                resolved = resolved.get(subkey)
            else:
                # 如果不是 dict，尝试属性访问
                resolved = getattr(resolved, subkey, None)
            if resolved is None:
                # print(f"[WARNING] 在取值链 '{placeholder}' 时，子属性 '{subkey}' 返回 None")
                break
        return resolved

    return None

# MCP客户端
class MCPClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.headers = {'Content-Type': 'application/json'}

    def _send_request(self, method, params=None):
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": str(uuid.uuid4())}
        print(f"\n[Agent] --> Sending request to server: {json.dumps(payload, indent=2)}")
        try:
            response = requests.post(self.server_url, data=json.dumps(payload), headers=self.headers)
            response.raise_for_status()
            response_json = response.json()
            if "error" in response_json:
                raise Exception(f"Server Error: {response_json['error']['message']}")
            print(f"[Agent] <-- Received result: {json.dumps(response_json.get('result'), indent=2)}")
            return response_json.get("result")
        except Exception as e:
            print(f"[Agent] An error occurred: {e}")
            return None

    def discover(self):
        return self._send_request("mcp/discover")

    def run(self, capability, parameters):
        params = {"capability": capability, "parameters": parameters}
        return self._send_request("mcp/run", params)

# --- 核心升级：工作流Agent ---
class LLMWorkflowAgent:
    def __init__(self, mcp_server_url):
        self.mcp_client = MCPClient(mcp_server_url)
        # 初始化LLM客户端
        try:
            self.llm_client = OpenAI(api_key="填入同学自己的api-key",
                                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        except KeyError:
            raise Exception("错误：OPENAI_API_KEY 环境变量未设置！")
            
        self.capabilities = None
        self.workflow_context = {} # 用于存储工作流的上下文信息

    def initialize(self):
        print("[Agent] Initializing... Discovering server capabilities.")
        discovery_result = self.mcp_client.discover()
        if discovery_result:
            self.capabilities = discovery_result.get("capabilities")
            print(f"[Agent] Discovery successful. Found {len(self.capabilities)} capabilities.")
            return True
        print("[Agent] Failed to discover capabilities. Agent cannot start.")
        return False

    def _generate_plan_with_llm(self, user_request):
        """
        **这是新增的核心方法**
        使用LLM根据用户请求和可用工具生成一个JSON格式的执行计划。
        """
        print("\n[Agent-Brain] Asking LLM to generate a plan...")
        
        # 将工具（能力）列表格式化，以便LLM理解
        tools_for_prompt = json.dumps(self.capabilities, indent=2)

        # 构建给LLM的系统提示（System Prompt），教会它如何扮演一个规划者
        system_prompt = f"""
You are a highly intelligent workflow orchestration agent. Your task is to create a step-by-step execution plan based on a user's request and a list of available tools.

You must respond with a valid JSON array of objects, where each object represents a step in the plan.
Each step object must have two keys:
1. "capability": The exact name of the tool to use from the provided list.
2. "parameters": An object containing the parameters needed for that tool.

The parameters can be:
- A literal value (e.g., a string or number).
- A placeholder string that references data from the initial user request, formatted as `{{request.key}}`.
- A placeholder string that references the output of a previous step, formatted as `{{steps[N].output.key}}`, where N is the zero-based index of the step.

Here is the list of available tools:
{tools_for_prompt}

Now, create a plan for the following user request.
"""
        
        # 发送请求给LLM
        response = self.llm_client.chat.completions.create(
            model="qwen-plus",  # 使用支持复杂指令和JSON模式的较新模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request}
            ],
            response_format={"type": "json_object"} # 强制要求LLM输出JSON
        )
        
        plan_json = response.choices[0].message.content
        print(f"[Agent-Brain] LLM generated the following plan:\n{plan_json}")
        
        # 返回解析后的JSON计划
        # 在真实应用中，这里需要更强的错误处理和校验
        return json.loads(plan_json)

    def _execute_plan(self, plan, initial_request_data):
        self.workflow_context = {"request": initial_request_data, "steps": []}
        for i, step in enumerate(plan):
            print(f"[Agent-Executor] Executing Step {i+1}: {step['capability']}")
            resolved_params = {}
            # 在这里替换旧解析逻辑
            for key, value in step["parameters"].items():
                print(f"[DEBUG] 参数 key={key}, 原始 value={repr(value)}")
                resolved = resolve_placeholders(value, initial_request_data, self.workflow_context)
                print(f"[DEBUG] 解析后 resolved={repr(resolved)}")
                if isinstance(value, str) and (value.strip().startswith("{") and value.strip().endswith("}")):
                    # 看起来像占位符形式
                    if resolved is None:
                        print(f"[Agent-Executor] WARNING: 占位符 '{value}' 解析失败或结果 None")
                        # 可抛错或让流程继续：这里示例继续并传 None
                    resolved_params[key] = resolved
                else:
                    # 普通字面值
                    resolved_params[key] = value

            print(f"[Agent-Executor] Resolved parameters: {resolved_params}")
            # 执行MCP调用
            try:
                result = self.mcp_client.run(step["capability"], resolved_params)
            except Exception as e:
                print(f"[Agent-Executor] 调用 MCP 出错: {e}")
                return False
            if not result:
                print(f"[Agent-Executor] FATAL: Step {i+1} failed. Aborting workflow.")
                return False
            # 保存结果
            self.workflow_context["steps"].append({"output": result})
            print(f"[Agent-Executor] Step {i+1} completed successfully. workflow_context steps count={len(self.workflow_context['steps'])}")
        return True

    def run_workflow(self, user_request, request_data):
        """主入口函数"""
        # 1. 生成计划
        plan = self._generate_plan_with_llm(user_request)
        if not plan:
            print("[Agent] Could not generate a valid plan. Aborting.")
            return

        # 2. 执行计划
        success = self._execute_plan(plan, request_data)

        if success:
            print("\n" + "="*50)
            print("[Agent] LLM-driven workflow completed successfully!")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("[Agent] LLM-driven workflow failed.")
            print("="*50)


if __name__ == '__main__':
    # 确保MCP Server正在运行
    SERVER_URL = "http://localhost:8000"

    # 初始化Agent
    agent = LLMWorkflowAgent(SERVER_URL)
    
    if agent.initialize():
        # --- 定义用户请求 ---
        # 这是一个更自然的、非结构化的用户请求
        natural_language_request = "我们需要为新来的员工'张三'办理入职手续。他的员工ID是'san.zhang'，职位是软件工程师。请走完整个流程。"
        
        # 将请求中的关键信息结构化，方便后续引用
        initial_data = {
            "employee_id": "san.zhang",
            "full_name": "张三",
            "position": "Software Engineer"
        }

        # --- 启动由LLM驱动的工作流 ---
        agent.run_workflow(natural_language_request, initial_data)