# mcp_server.py
# 模拟企业内部系统的MCP服务器

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- 模拟的后台数据库和系统 ---
DB = {
    "employees": {},
    "assets": {
        "LAP001": {"model": "MacBook Pro 16-inch", "spec": "M3 Max, 32GB RAM", "status": "available"},
        "LAP002": {"model": "Dell XPS 15", "spec": "Core i9, 32GB RAM", "status": "available"},
        "LAP003": {"model": "MacBook Pro 14-inch", "spec": "M3 Pro, 16GB RAM", "status": "assigned"},
    },
    "accounts": {},
}

class MCPRequestHandler(BaseHTTPRequestHandler):
    """
    处理MCP请求的核心处理器
    """
    def _send_response(self, content):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(content).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)

        method = request_body.get("method")
        params = request_body.get("params", {})
        request_id = request_body.get("id")

        print(f"\n[Server] Received method: {method} with params: {params}")

        response = {"jsonrpc": "2.0", "id": request_id}

        if method == "mcp/discover":
            # --- 核心：定义并暴露服务器的能力 ---
            response["result"] = {
                "mcp_version": "0.1.0",
                "capabilities": {
                    "hris_create_employee_profile": {
                        "description": "在HR系统中为新员工创建档案。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "employee_id": {"type": "string", "description": "员工的唯一ID"},
                                "full_name": {"type": "string", "description": "员工全名"},
                                "position": {"type": "string", "description": "员工职位"}
                            },
                            "required": ["employee_id", "full_name", "position"]
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "employee_id": {"type": "string"}
                            }
                        }
                    },
                    "itam_find_available_laptop": {
                        "description": "查找一台可用的笔记本电脑。",
                        "parameters": {
                           "type": "object",
                           "properties": {
                                "position_requirement": {"type": "string", "description": "职位对电脑的要求, e.g., 'high-performance' for engineers"}
                           },
                           "required": ["position_requirement"]
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "asset_id": {"type": "string"},
                                "model": {"type": "string"},
                                "spec": {"type": "string"}
                            }
                        }
                    },
                    "itam_assign_asset_to_employee": {
                        "description": "将指定的IT资产分配给员工。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "asset_id": {"type": "string"},
                                "employee_id": {"type": "string"}
                            },
                            "required": ["asset_id", "employee_id"]
                        },
                         "returns": {"type": "object", "properties": {"status": {"type": "string"}}}
                    },
                    "iam_create_user_accounts": {
                        "description": "为员工创建公司账户（邮箱、Jira等）。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "employee_id": {"type": "string"},
                                "full_name": {"type": "string"}
                            },
                            "required": ["employee_id", "full_name"]
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "jira_account": {"type": "string"}
                            }
                        }
                    },
                    "notifier_send_notification": {
                        "description": "向指定收件人发送通知。",
                        "parameters": {
                           "type": "object",
                           "properties": {
                                "recipient": {"type": "string", "description": "收件人地址或ID"},
                                "message": {"type": "string", "description": "通知内容"}
                           },
                           "required": ["recipient", "message"]
                        },
                        "returns": {"type": "object", "properties": {"status": {"type": "string"}}}
                    }
                }
            }
        
        elif method == "mcp/run":
            capability_name = params.get("capability")
            capability_params = params.get("parameters", {})
            
            # --- 核心：执行具体的能力 ---
            try:
                if capability_name == "hris_create_employee_profile":
                    emp_id = capability_params['employee_id']
                    DB['employees'][emp_id] = {
                        "full_name": capability_params['full_name'],
                        "position": capability_params['position'],
                        "assigned_asset": None
                    }
                    result = {"status": "success", "employee_id": emp_id}

                elif capability_name == "itam_find_available_laptop":
                    # 工程师需要高性能电脑
                    req = capability_params['position_requirement']
                    found_asset = None
                    for asset_id, details in DB['assets'].items():
                        if details['status'] == 'available':
                            if req == 'high-performance' and ('MacBook Pro' in details['model'] or 'XPS' in details['model']):
                                found_asset = {"asset_id": asset_id, **details}
                                break
                    if not found_asset:
                        raise Exception("No suitable high-performance laptop available.")
                    result = found_asset

                elif capability_name == "itam_assign_asset_to_employee":
                    asset_id = capability_params['asset_id']
                    emp_id = capability_params['employee_id']
                    if DB['assets'][asset_id]['status'] != 'available':
                        raise Exception(f"Asset {asset_id} is not available.")
                    DB['assets'][asset_id]['status'] = 'assigned'
                    DB['employees'][emp_id]['assigned_asset'] = asset_id
                    result = {"status": f"Asset {asset_id} assigned to {emp_id}"}
                
                elif capability_name == "iam_create_user_accounts":
                    emp_id = capability_params['employee_id']
                    name_part = emp_id.split('.')[0]
                    DB['accounts'][emp_id] = {
                        "email": f"{emp_id}@examplecorp.com",
                        "jira_account": f"jira-{name_part}"
                    }
                    result = DB['accounts'][emp_id]

                elif capability_name == "notifier_send_notification":
                    print(f"  [Notification Sent] To: {capability_params['recipient']}, Message: '{capability_params['message']}'")
                    result = {"status": "notification sent"}

                else:
                    raise Exception("Capability not found.")

                response["result"] = result
                print(f"[Server] Executed successfully. Result: {result}")

            except Exception as e:
                response["error"] = {"code": -32000, "message": str(e)}
                print(f"[Server] Execution error: {e}")
        
        else:
             response["error"] = {"code": -32601, "message": "Method not found"}

        self._send_response(response)


def run_server(server_class=HTTPServer, handler_class=MCPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"MCP Server started on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("MCP Server stopped.")

if __name__ == '__main__':
    run_server()