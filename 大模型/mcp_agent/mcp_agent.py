# mcp_agent.py
# 一个模拟处理复杂工作流的MCP Agent

import requests
import json
import uuid

class MCPClient:
    """简单的MCP客户端，用于与服务器通信"""
    def __init__(self, server_url):
        self.server_url = server_url
        self.headers = {'Content-Type': 'application/json'}

    def _send_request(self, method, params=None):
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": str(uuid.uuid4())
        }
        print(f"\n[Agent] --> Sending request to server: {method}")
        try:
            response = requests.post(self.server_url, data=json.dumps(payload), headers=self.headers)
            response.raise_for_status()
            response_json = response.json()
            if "error" in response_json:
                raise Exception(f"Server Error: {response_json['error']['message']}")
            return response_json.get("result")
        except requests.exceptions.RequestException as e:
            print(f"[Agent] Communication error: {e}")
            return None
        except Exception as e:
            print(f"[Agent] An error occurred: {e}")
            return None

    def discover(self):
        return self._send_request("mcp/discover")

    def run(self, capability, parameters):
        params = {"capability": capability, "parameters": parameters}
        return self._send_request("mcp/run", params)

class WorkflowAgent:
    """
    负责编排和执行整个入职工作流的Agent
    """
    def __init__(self, mcp_server_url):
        self.client = MCPClient(mcp_server_url)
        self.capabilities = None
        self.workflow_context = {} # 用于在工作流步骤之间传递数据

    def initialize(self):
        """发现服务器能力并准备好工作"""
        print("[Agent] Initializing... Discovering server capabilities.")
        discovery_result = self.client.discover()
        if discovery_result:
            self.capabilities = discovery_result.get("capabilities")
            print(f"[Agent] Discovery successful. Found {len(self.capabilities)} capabilities.")
            return True
        print("[Agent] Failed to discover capabilities. Agent cannot start.")
        return False

    def execute_onboarding_workflow(self, employee_id, full_name, position):
        """
        核心工作流逻辑。
        在真实场景中，这里计划可能是由一个大型语言模型（LLM）动态生成的。
        此处我们用一个硬编码的计划来模拟。
        """
        print(f"[Agent] Starting onboarding workflow for {full_name} ({employee_id})")

        # --- 步骤 1: 在HR系统中创建档案 ---
        print("\n[PLAN] Step 1: Create employee profile in HRIS.")
        profile_params = {"employee_id": employee_id, "full_name": full_name, "position": position}
        profile_result = self.client.run("hris_create_employee_profile", profile_params)
        if not profile_result: return # 如果失败则终止
        self.workflow_context['employee_id'] = profile_result['employee_id']
        print(f"[SUCCESS] Profile created for {self.workflow_context['employee_id']}.")

        # --- 步骤 2: 查找合适的笔记本电脑 ---
        # 模拟LLM的决策
        laptop_req = 'high-performance' if position.lower() == 'software engineer' else 'standard'
        laptop_result = self.client.run("itam_find_available_laptop", {"position_requirement": laptop_req})
        if not laptop_result: return
        self.workflow_context['laptop'] = laptop_result
        print(f"[SUCCESS] Found laptop: {laptop_result['model']} (ID: {laptop_result['asset_id']}).")

        # --- 步骤 3: 分配笔记本电脑 ---
        assign_params = {
            "asset_id": self.workflow_context['laptop']['asset_id'],
            "employee_id": self.workflow_context['employee_id']
        }
        assign_result = self.client.run("itam_assign_asset_to_employee", assign_params)
        if not assign_result: return
        print(f"[SUCCESS] {assign_result['status']}.")

        # --- 步骤 4: 创建用户账户 ---
        account_params = {"employee_id": employee_id, "full_name": full_name}
        account_result = self.client.run("iam_create_user_accounts", account_params)
        if not account_result: return
        self.workflow_context['accounts'] = account_result
        print(f"[SUCCESS] Accounts created: Email -> {account_result['email']}, Jira -> {account_result['jira_account']}.")

        # --- 步骤 5: 通知IT支持团队 ---
        it_message = (f"Please prepare laptop {self.workflow_context['laptop']['asset_id']} "
                      f"({self.workflow_context['laptop']['model']}) for new employee "
                      f"{full_name} ({employee_id}).")
        self.client.run("notifier_send_notification", {"recipient": "it-support@examplecorp.com", "message": it_message})
        print(f"[SUCCESS] IT support team notified.")

        # --- 步骤 6: 发送欢迎邮件给新员工 ---
        welcome_message = (f"Welcome to the team, {full_name}!\n\n"
                           f"Your accounts have been created:\n"
                           f"- Email: {self.workflow_context['accounts']['email']}\n"
                           f"- Jira: {self.workflow_context['accounts']['jira_account']}\n\n"
                           f"IT will contact you regarding your new {self.workflow_context['laptop']['model']}.")
        self.client.run("notifier_send_notification", {"recipient": self.workflow_context['accounts']['email'], "message": welcome_message})
        print(f"[SUCCESS] Welcome email sent.")


if __name__ == '__main__':
    # 确保你的MCP Server正在运行
    SERVER_URL = "http://localhost:8000"

    agent = WorkflowAgent(SERVER_URL)
    
    if agent.initialize():
        # --- 启动工作流 ---
        agent.execute_onboarding_workflow(
            employee_id="john.doe",
            full_name="John Doe",
            position="Software Engineer"
        )