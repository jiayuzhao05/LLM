import openai
import yaml
import json
import os
import random
import time
import logging
import math
from pathlib import Path
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from rouge_score import rouge_scorer

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 强健的 OpenAI 客户端 ---
class ResilientOpenAIClient:
    """一个具有重试和端点故障切换能力的稳健 OpenAI 客户端。"""
    def __init__(self, api_keys: Dict[str, str], addresses: List[str], resilience_config: Dict):
        self.clients = []
        self.addresses = addresses
        self.current_client_index = 0
        
        for addr in addresses:
            api_key = api_keys.get(addr, "EMPTY")  # 本地模型默认为 "EMPTY"
            self.clients.append(openai.OpenAI(api_key=api_key, base_url=addr))

        self.retry_decorator = retry(
            stop=stop_after_attempt(resilience_config.get('max_retries', 3)),
            wait=wait_exponential(
                multiplier=1,
                min=resilience_config.get('wait_initial_seconds', 2),
                max=10
            )
        )
        self.timeout = resilience_config.get('request_timeout', 120)

    def create_chat_completion(self, *args, **kwargs):
        """执行带重试和故障切换逻辑的 API 调用。"""
        kwargs['timeout'] = self.timeout
        
        @self.retry_decorator
        def _attempt_to_call():
            try:
                client = self.clients[self.current_client_index]
                logging.info(f"尝试调用 API: {client.base_url}...")
                return client.chat.completions.create(*args, **kwargs)
            except Exception as e:
                logging.warning(f"调用失败: {self.clients[self.current_client_index].base_url} 错误: {e}，正在重试...")
                raise

        for _ in range(len(self.clients)):
            try:
                return _attempt_to_call()
            except RetryError as e:
                logging.error(f"端点 {self.clients[self.current_client_index].base_url} 重试失败。错误: {e}")
                self.current_client_index = (self.current_client_index + 1) % len(self.clients)
                logging.info(f"切换到下一个客户端: {self.clients[self.current_client_index].base_url}")
        
        raise Exception("所有 API 端点在所有重试后均失败。")

# --- 2. 代理基础设施 (使用 ABC 重构) ---
class Agent(ABC):
    """所有数据合成代理的抽象基类。"""
    def __init__(self, client: ResilientOpenAIClient, judge_client: ResilientOpenAIClient, config: Dict):
        self.client = client
        self.judge_client = judge_client
        self.config = config

    @abstractmethod
    def generate(self, task: Dict) -> Dict:
        """基于任务执行一次数据生成。返回包含生成结果的字典。"""
        raise NotImplementedError

    @abstractmethod
    def verify(self, generated_data: Dict) -> bool:
        """验证生成结果是否满足内部质量标准。满足则返回 True，否则 False。"""
        raise NotImplementedError
    
    def run_generation_loop(self, task: Dict, max_attempts: int = 3) -> Dict:
        """生成-验证循环模板，确保各代理执行模式一致。"""
        for attempt in range(max_attempts):
            logging.info(f"任务 '{task.get('task_id', 'N/A')}' 第 {attempt + 1}/{max_attempts} 次生成尝试")
            generated_data = self.generate(task)
            if self.verify(generated_data):
                logging.info("验证通过。")
                return generated_data
            else:
                logging.warning("验证失败，重试中...")
        
        raise Exception(f"{max_attempts} 次生成尝试后未能生成有效实例。")

class SelfInstructAgent(Agent):
    """通过与现有任务池比较生成新指令。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_tasks = self._load_seed_tasks()
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        if not self.seed_tasks:
            raise ValueError("SelfInstructAgent 需要非空的种子任务文件。")

    def _load_seed_tasks(self):
        seed_path = self.config['seed_tasks_path']
        if not os.path.exists(seed_path):
            logging.warning(f"未找到种子任务文件: {seed_path}。")
            return []
        with open(seed_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def generate(self, task: Dict) -> Dict:
        """生成一条新指令。"""
        sampled_seeds = random.sample(
            self.seed_tasks,
            min(len(self.seed_tasks), self.config['agents']['self_instruct']['num_instructions_to_sample'])
        )
        prompt_instructions = "\n\n".join([f"- {s['problem']}" for s in sampled_seeds])
        prompt = (
            f"基于以下指令示例，生成一条全新且与示例不同的指令。\n\n"
            f"示例：\n{prompt_instructions}\n\n新指令："
        )
        response = self.client.create_chat_completion(
            model=self.config['generator_model']['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512
        )
        new_instruction = response.choices[0].message.content.strip()
        return {"generated_problem": new_instruction}

    def verify(self, generated_data: Dict) -> bool:
        """使用 ROUGE-L 验证新指令与种子池的相似度，不得过高才能通过。"""
        new_instruction = generated_data["generated_problem"]
        all_instructions = [s['problem'] for s in self.seed_tasks]
        scores = [
            self.scorer.score(new_instruction, old)['rougeL'].fmeasure
            for old in all_instructions
        ]
        max_rouge_l = max(scores) if scores else 0
        logging.info(f"验证新指令。最大 ROUGE-L: {max_rouge_l:.4f}")
        if max_rouge_l < self.config['agents']['self_instruct']['rouge_l_threshold']:
            self.seed_tasks.append({"problem": new_instruction})
            return True
        return False

class EvolInstructAgent(Agent):
    """使用广度和深度策略演化指令。"""
    def generate(self, task: Dict) -> Dict:
        """执行一次演化生成。"""
        original = task['problem']
        if random.random() < 0.5:
            logging.info("执行广度演化...")
            prompt = (
                f"我希望你作为提示词创造者。\n从以下提示词中创造一个全新提示词，"
                f"与原提示词同领域但更加罕见，长度和难度相似且合理可解。\n"
                f"原提示词：{original}\n新提示词："
            )
        else:
            logging.info("执行深度演化...")
            strategy = random.choice(self.config['agents']['evol_instruct']['deepening_strategies'])
            prompts = {
                "add_constraints": f"请为以下指令添加一个额外的约束或要求：\n{original}",
                "deepening": f"如果你是专家，请将以下指令改写得更专业、更深入：\n{original}",
                "concretizing": f"请将以下指令改写得更具体，例如添加具体场景：\n{original}",
                "increased_reasoning_steps": f"请将以下指令改写为需要更多推理步骤：\n{original}",
                "complicating_input": f"请复杂化以下指令的输入，使其更具挑战：\n{original}"
            }
            prompt = prompts.get(strategy, prompts["add_constraints"])
        response = self.client.create_chat_completion(
            model=self.config['generator_model']['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512
        )
        evolved = response.choices[0].message.content.strip()
        return {"original_problem": original, "evolved_problem": evolved}

    def verify(self, generated_data: Dict) -> bool:
        """两阶段验证：信息增益和可解性检查。"""
        orig = generated_data.get("original_problem")
        evo = generated_data.get("evolved_problem")
        if not orig or not evo:
            return False
        # 信息增益检查
        logging.info("验证信息增益...")
        info_prompt = (
            f"以下两条指令是否在约束、深度和广度上等同？仅回答 'Equal' 或 'Not Equal':\n"
            f"第一条：{orig}\n第二条：{evo}\n"
        )
        try:
            res = self.judge_client.create_chat_completion(
                model=self.config['judge_model']['model_name'],
                messages=[{"role": "user", "content": info_prompt}],
                temperature=0.0
            )
            ans = res.choices[0].message.content.strip().lower()
            if "equal" in ans:
                logging.warning("验证未通过：演化结果与原指令过于相似。")
                return False
        except Exception as e:
            logging.error(f"信息增益检查错误: {e}")
            return False
        # 可解性检查
        logging.info("验证可解性...")
        try:
            resp = self.client.create_chat_completion(
                model=self.config['generator_model']['model_name'],
                messages=[{"role": "user", "content": evo}],
                temperature=0.7, max_tokens=256
            )
            sol = resp.choices[0].message.content.strip()
            # 检查无法回答的关键词
            bad = ["sorry","cannot","unable","i can't","i am not able"]
            if any(k in sol.lower() for k in bad):
                logging.warning("验证未通过：模型回答表明困难。" )
                return False
            # 检查回答长度
            if len(sol.split()) < 80:
                logging.warning("验证未通过：回答过短。")
                return False
        except Exception as e:
            logging.error(f"可解性检查错误: {e}")
            return False
        logging.info("演化验证通过。")
        return True

class MCTSAgent(Agent):
    """使用蒙特卡洛树搜索生成复杂推理路径的简化代理。"""
    class MCTSNode:
        def __init__(self, state, parent=None):
            self.state = state  # state 可以是已生成的文本序列
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0

    def generate(self, task: Dict) -> Dict:
        root = self.MCTSNode(state="")
        c = self.config['agents']['mcts']['exploration_factor']
        for _ in range(self.config['agents']['mcts']['num_iterations']):
            # 1. 选择
            node = root
            while node.children:
                node = max(
                    node.children,
                    key=lambda n: (n.wins / n.visits)
                    + c * math.sqrt(math.log(node.visits) / n.visits)
                    if n.visits > 0 else float('inf')
                )
            # 2. 扩展
            if node.visits > 0 or node == root:
                prompt = (
                    f"给定问题 '{task['problem']}' 和当前推理路径 '{node.state}'，"
                    "请给出 3 个可能的下一个逻辑步骤，用 '|||' 分隔。"
                )
                resp = self.client.create_chat_completion(
                    model=self.config['generator_model']['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8, max_tokens=1024
                )
                steps = resp.choices[0].message.content.strip().split('|||')
                for step in steps:
                    if step.strip():
                        child_state = node.state + "\n" + step.strip()
                        child = self.MCTSNode(state=child_state, parent=node)
                        node.children.append(child)
            # 3. 模拟
            sim_node = node.children[0] if node.children else node
            sim_prompt = (
                f"问题: '{task['problem']}'\n部分解题路径:\n{sim_node.state}"
                "\n\n请基于此路径完成解答。"
            )
            sim_res = self.client.create_chat_completion(
                model=self.config['generator_model']['model_name'],
                messages=[{"role": "user", "content": sim_prompt}],
                temperature=0.5, max_tokens=2048
            )
            final_sol = sim_res.choices[0].message.content
            # 4. 反向传播
            judge_prompt = (
                f"[问题]\n{task['problem']}\n\n[解答]\n{final_sol}"
                "\n\n任务: 用 1-5 分评估解答质量，只返回整数分数。"
            )
            judge_res = self.judge_client.create_chat_completion(
                model=self.config['judge_model']['model_name'],
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            score = int(judge_res.choices[0].message.content.strip())
            reward = (score - 1) / 4.0  # 标准化到 [0,1]

            temp = sim_node
            while temp:
                temp.visits += 1
                temp.wins += reward
                temp = temp.parent
        # 选择访问次数最多的子节点路径
        best = max(root.children, key=lambda n: n.visits)
        final_prompt = (
            f"问题: '{task['problem']}'\n请按照以下推理路径给出完整解答:\n{best.state}"
        )
        final_res = self.client.create_chat_completion(
            model=self.config['generator_model']['model_name'],
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.1, max_tokens=2048
        )
        return {"reasoning_path": best.state, "final_solution": final_res.choices[0].message.content}

    def verify(self, generated_data: Dict) -> bool:
        """使用 Judge 模型对最终解答进行评分，确保奖励不低于阈值才能通过。"""
        sol = generated_data.get("final_solution")
        if not sol:
            return False
        prompt = (
            f"[解答] {sol}\n\n请对上述解答质量进行评分 (1-5)，仅返回整数。"
        )
        try:
            res = self.judge_client.create_chat_completion(
                model=self.config['judge_model']['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            score = int(res.choices[0].message.content.strip())
            normalized = (score - 1) / 4.0
            threshold = self.config['agents']['mcts'].get('verify_threshold', 0.5)
            if normalized >= threshold:
                logging.info(f"MCTS 验证通过: 评分 {score}，标准化后 {normalized:.2f}")
                return True
            else:
                logging.warning(f"MCTS 验证未通过: 评分 {score}，标准化后 {normalized:.2f} < 阈值 {threshold}")
                return False
        except Exception as e:
            logging.error(f"MCTS 验证错误: {e}")
            return False

# --- 3. 主流程 ---
class SynthesisPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.generator_client = self._init_generator_client()
        self.judge_client = self._init_judge_client()
        
        self.agents = {
            "self-instruct": SelfInstructAgent(self.generator_client, self.judge_client, self.config),
            "evol-instruct": EvolInstructAgent(self.generator_client, self.judge_client, self.config),
            "mcts": MCTSAgent(self.generator_client, self.judge_client, self.config),
        }
        logging.info(f"已初始化代理: {list(self.agents.keys())}")

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_generator_client(self) -> ResilientOpenAIClient:
        gen_conf = self.config['generator_model']
        return ResilientOpenAIClient({}, gen_conf['api_server_addresses'], self.config['resilience'])

    def _init_judge_client(self) -> ResilientOpenAIClient:
        judge_conf = self.config['judge_model']
        api_key = os.getenv("OPENAI_API_KEY") if judge_conf.get('api_key') == 'env' else judge_conf.get('api_key')
        if not api_key:
            raise ValueError("未找到 Judge API Key。")
        return ResilientOpenAIClient({judge_conf['api_server_address']: api_key}, [judge_conf['api_server_address']], self.config['resilience'])

    def _run_judgement(self, task_to_judge: Dict) -> Dict:
        prob = task_to_judge.get("original_problem") or task_to_judge.get("generated_problem") or "N/A"
        sol = task_to_judge.get("final_solution") or task_to_judge.get("evolved_problem") or task_to_judge.get("generated_problem") or "N/A"
        prompt = (
            f"[问题]\n{prob}\n\n[解答]\n{sol}\n\n"
            "请以 JSON 格式返回评分和理由 {\"score\": int, \"reasoning\": str}。"
        )
        response = self.judge_client.create_chat_completion(
            model=self.config['judge_model']['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"score": 0, "reasoning": "Judge 返回了无效 JSON。"}

    def run_pipeline(self):
        input_dir = Path(self.config['input_data_dir'])
        output_path = Path(self.config['output_data_path'])
        task_files = sorted(list(input_dir.glob("*.json")))
        
        logging.info(f"发现 {len(task_files)} 个任务，开始处理...")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, task_file in enumerate(task_files):
                logging.info(f"--- 处理任务 {i+1}/{len(task_files)}: {task_file.name} ---")
                try:
                    with open(task_file, 'r', encoding='utf-8') as f_in:
                        task_data = json.load(f_in)
                    algo = task_data.get('synthesis_algorithm')
                    agent = self.agents.get(algo)
                    if not agent:
                        logging.error(f"未知算法 '{algo}'，跳过。")
                        continue
                    generated_data = agent.run_generation_loop(task_data)
                    judgement = self._run_judgement(generated_data)
                    final_record = {
                        "original_task_file": str(task_file.name),
                        "synthesis_algorithm": algo,
                        "generated_data": generated_data,
                        "final_evaluation": judgement
                    }
                    f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    logging.critical(f"处理 {task_file.name} 时致命错误: {e}", exc_info=True)
                    error_record = {"error": str(e), "file": str(task_file.name)}
                    f_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        logging.info(f"流水线完成，结果已保存至 '{output_path}'。")

if __name__ == "__main__":
    pipeline = SynthesisPipeline(config_path="config_v2.yaml")
    pipeline.run_pipeline()
