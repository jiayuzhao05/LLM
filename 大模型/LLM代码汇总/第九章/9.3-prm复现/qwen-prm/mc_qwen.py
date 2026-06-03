import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import re
from openai import OpenAI
from prompt import *
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------------------
# (A) 步骤: 模拟MC估计 + 策略模型
# ------------------------------------------------------------------------------

def get_message(prompt: str, temperature: float):
    try:
        client = OpenAI(
            api_key="xxx",    # 替换为实际的API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature = temperature
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")

def evaluate_consistency(res):
    # 将res转换为numpy数组，以便进行数值操作
    res_array = np.array(res)  
    # 计算平均值
    mean_res = np.mean(res_array)  
    # 计算标准差（或方差）
    std_dev = np.std(res_array)   
    # 计算方差（标准差的平方）
    variance = np.var(res_array)
    # 评估结果的一致性
    if std_dev < 0.1 * mean_res:
        return 1
    else:
        return 0
           
def mock_mc_estimation(problem_text: str, solution_steps: List[str], groud_truth) -> List[int]:
    """
    蒙特卡洛(MC)估计函数：
    输入题目文本和其解答过程，每一步的推理步骤是solution_steps中的一段。
    
    1) 在当前步骤截断后，让策略模型多次(如8次)继续生成后续推理与答案
    2) 统计各次生成是否得到正确答案
    3) 如果有一次能得到正确答案，则MC认为该步骤为“潜在正确”(1)，否则标记为(0)
    """
    num_simulations = 8
    res = []
    for step in solution_steps:
        # 模拟若干步骤
        all_step_ans = []
        problem_text = zero_single_proposal_prompt_qwen.format(problem_text)
        for _ in range(num_simulations):
            solution = get_message(problem_text)
            ans = solution.split("最终答案是: ")[-1]
            all_step_ans.append(ans)
        is_true = evaluate_consistency(all_step_ans)
        res.append(is_true)
    return res

def split_steps(input_str):
    # 使用正则表达式匹配“step x”模式，并切分字符串
    steps = re.split(r'(?=step \d)', input_str)
    # 移除多余的空字符串
    steps = [step.strip() for step in steps if step]
    return steps

def mock_policy_model_generate(problem_text: str, num_solutions=6) -> List[List[str]]:
    """
    该函数为策略模型(Policy Model)输出函数
    给定一道题目，返回不同的num_solutions个解答（解答由多步组成）。
    
    返回结构: List[solution], 每个solution是List[str], 每个str表示一个推理步骤。
    """
    solutions = []
    for _ in range(num_solutions):
        # 模拟若干步骤
        ts = [round(0.1 * i, 1) for i in range(5, 13)]
        t = random.choice(ts)
        problem_text = policy_model_generate.format(problem_text=problem_text)
        solution = get_message(problem_text, temperature=t)
        steps = split_steps(solution)
        solutions.append(steps)
    return solutions


def mc_labeling_for_problem(problem_text: str) -> List[Dict]:
    """
    给定一个题目，生成若干解答，并对每条解答的每一步做MC估计打标。
    返回的数据格式示例：
    [
      {
        "problem": problem_text,
        "steps": [
          {"text": "Step_1_mock_explanation", "mc_label": 1},
          ...
        ]
      },
      ...
    ]
    """
    solutions = mock_policy_model_generate(problem_text, num_solutions=6)
    print(solutions)
    labeled_solutions = []
    for sol in solutions:
        mc_labels = mock_mc_estimation(problem_text, sol)
        step_data = []
        for step_text, label in zip(sol, mc_labels):
            step_data.append({"text": step_text, "mc_label": label})
        
        labeled_solutions.append({
            "problem": problem_text,
            "steps": step_data
        })
    return labeled_solutions


# ------------------------------------------------------------------------------
# (B) 步骤: 使用 LLM-as-a-judge 重新评估推理步骤
# ------------------------------------------------------------------------------

def mock_llm_judge(problem_text: str, step_text: str) -> int:
    """
    LLM-as-a-judge 函数:
    给定题目和推理步骤，返回一个0/1判断, 1表示正确，0表示错误。
    
    调用 Qwen2.5-72B-Instruct 等大模型来做推理并分析该步是否正确。
    """
    eval_from_llm_judge = eval_prompt.format(problem_text=problem_text, step_text=step_text)
    res = get_message(eval_from_llm_judge)
    num_res = res.split("正确性: ")[-1]
    return num_res

def llm_judge_labeling(problem_data: List[Dict]) -> List[Dict]:
    """
    对 mc_labeling_for_problem() 的输出数据做 LLM-as-a-judge 判断。
    这里遍历每条solution的每个step，增加 judge_label 字段。
    
    数据形式参考:
    [
      {
        "problem": "xxx",
        "steps": [
          {"text": "...", "mc_label": 1, "judge_label": ? },
          ...
        ]
      },
      ...
    ]
    """
    output_data = []
    for sol_dict in problem_data:
        new_steps = []
        for step_info in sol_dict["steps"]:
            step_text = step_info["text"]
            judge_res = mock_llm_judge(sol_dict["problem"], step_text)
            step_info["judge_label"] = judge_res
            new_steps.append(step_info)
        sol_dict["steps"] = new_steps
        output_data.append(sol_dict)
    return output_data


# ------------------------------------------------------------------------------
# (C) 步骤: 共识过滤(consensus filtering)
#     只有当 (mc_label == 1) 且 (judge_label == 1) 时，才认为该步“真正正确”。
# ------------------------------------------------------------------------------
def consensus_filtering(data: List[Dict]) -> List[Dict]:
    """
    对数据集中每条解答进行过滤：
    只保留那些MC和Judge一起认为正确的步骤；若出现矛盾或任何一方认为是0，则视为错误。
    同时，如果某一步是错误的，可在训练时直接截断后续步骤（因为后续步骤也没用了）或者根据需要保留。
    这里的实现：只返回label一致的步骤。
    """
    filtered_data = []
    for sol_dict in data:
        # 以列表形式收集步骤
        correct_steps = []
        for step_info in sol_dict["steps"]:
            mc_label_val = step_info.get("mc_label", 0)
            judge_label_val = step_info.get("judge_label", 0)
            # 共识：只有当两个label都为1时，才是真的“正确”
            # 如果只想要一致(都0或者都1)，可以改为 mc_label_val == judge_label_val
            if mc_label_val == 1 and judge_label_val == 1:
                # 我们此处可以保留，也可以对错误直接截断
                # 本示例只要相同且为1就保留
                correct_steps.append({
                    "text": step_info["text"],
                    "final_label": 1  # 最后确定此步正确
                })
            else:
                # 也可以这里“截断”后续
                # break
                # 如果要包含错误步骤:
                correct_steps.append({
                    "text": step_info["text"],
                    "final_label": 0
                })
                # 如论文中提到，如果第一次出现错误，可以后面全部丢弃
                # break
        # 保留结果
        filtered_data.append({
            "problem": sol_dict["problem"],
            "steps": correct_steps
        })
    return filtered_data

def get_emb(input: str):
    client = OpenAI(
        api_key="xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
    )
    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=input,
        dimensions=1024,
        encoding_format="float"
    )

    return completion.model_dump_json()['data']['embedding']

# ------------------------------------------------------------------------------
# (D) 建立一个简单的PRM模型 + 数据集与训练流程
# ------------------------------------------------------------------------------

class PRMDataset(Dataset):
    """
    构建 PRM 数据集类。
    输入是“步骤文本 + 最终标签(final_label)”
    用encoder将step_text转成向量。
    """
    def __init__(self, data: List[Dict], tokenizer=None):
        self.tokenizer = tokenizer
        # data包含若干解答，每个解答包含多个steps
        self.samples = []
        for sol_dict in data:
            for step_info in sol_dict["steps"]:
                step_text = step_info["text"]
                label = step_info["final_label"]
                # 在复杂场景下，step_text可以使用tokenizer后转换成embedding特征
                embedding = tokenizer([step_text])
                self.samples.append((embedding, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, label = self.samples[idx]
        return emb, label


class SimplePRM(nn.Module):
    """
    简单的过程奖励模型示例:
    输入: (batch_size, embedding_dim) 的向量 (模拟)
    输出: (batch_size, 1) 的分数, 用sigmoid映射到[0,1], 表示此步正确的概率
    """
    def __init__(self, model):
        super().__init__()
        self.net = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.net(x)
        scores = self.sigmoid(logits)  # 分数，表示“正确性”的概率
        return scores

def train_prm_model(filtered_data: List[Dict], epochs=3, batch_size=8, model=None):
    """
    用共识过滤后的数据来训练PRM。
    """
    dataset = PRMDataset(filtered_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimplePRM()
    criterion = nn.BCELoss(ignore_index=-100)  # 二分类: label=0/1
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for emb, label in dataloader:
            optimizer.zero_grad()
            scores = model(emb)  # shape: [batch_size, 1]
            label = label.float().unsqueeze(1)  # shape: [batch_size, 1]
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss={avg_loss:.4f}")
    return model


# ------------------------------------------------------------------------------
# (E) 简单示例: 评估 (Best-of-N) & PROCESSBENCH 风格
# ------------------------------------------------------------------------------
def evaluate_best_of_n(model: SimplePRM, problem_text: str, n=8, tokenizer=None):
    """
    给定一个训练好的 PRM 模型，针对单一道题，做 Best-of-N 评估。
    演示：让策略模型生成n个解答，用PRM对每个解答的每步打分，再用“某种合并方法”得到整条解答分数。
    """
    solutions = mock_policy_model_generate(problem_text, num_solutions=n)

    best_score = -1.0
    best_sol = None

    for sol in solutions:
        # 将每个step做文本embedding；随机模拟
        step_embeddings = [tokenizer([s]) for s in sol]
        step_scores = []
        for emb in step_embeddings:
            emb = emb.unsqueeze(0)  # shape=[1,16]
            with torch.no_grad():
                score = model(emb)  # shape=[1,1]
                step_scores.append(score.item())

        # 简单演示 "product" 策略, 也可改成 "min" 或 "last", 论文中有提到这几种组合方法
        import math
        product_score = 1.0
        for sc in step_scores:
            product_score *= sc
        # 取对数做稳定性
        # product_score = sum(math.log(sc+1e-9) for sc in step_scores)

        if product_score > best_score:
            best_score = product_score
            best_sol = sol  # 记录最优解

    print(f"[DEBUG] Best-of-{n} Score = {best_score:.5f}, solution = {best_sol}")
    return best_sol


def evaluate_processbench_style(model: SimplePRM, steps: List[str], tokenizer=None) -> int:
    """
    模拟 PROCESSBENCH 风格的步骤级评价：
    给定一个完整的推理过程(多步), 我们要求找出第一步出错的下标。
    如果模型判断所有步骤皆为正确，则返回 -1 表示全对。
    
    embedding + PRM打分，当分数<0.5时，就视为错误。
    """
    for idx, step_text in enumerate(steps):
        emb = tokenizer.encode([step_text])
        with torch.no_grad():
            score = model(emb).item()
        if score < 0.5:
            # 认为是第一个出现错误的步骤
            return idx
    return -1  # 全部正确

# ------------------------------------------------------------------------------
# 主函数演示
# ------------------------------------------------------------------------------
def main_demo():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct",trust_remote_code=True)
    # 1) 模拟一个题目
    problem_text = "【示例题目】求解 x^2 + 2x + 1 = 0 的根。"

    # 2) 用策略模型 + MC估计做数据生成
    mc_data = mc_labeling_for_problem(problem_text)
    # 3) 用 LLM-as-a-judge 评估
    judged_data = llm_judge_labeling(mc_data)
    # 4) 共识过滤
    filtered_data = consensus_filtering(judged_data)

    print(f"[DEBUG] 共识过滤后的数据:{filtered_data}")

    # 5) 训练 PRM
    print("开始训练 PRM...")
    prm_model = train_prm_model(filtered_data, epochs=3, batch_size=4, tokenizer=tokenizer)

    # 6) 做一个简单的 Best-of-N 评估演示
    print("\n Best-of-N 测试")
    evaluate_best_of_n(prm_model, problem_text, n=8, tokenizer=tokenizer)

    # 7) 做一个类似 PROCESSBENCH 风格的演示
    print("\n PROCESSBENCH 风格测试")
    test_steps = ["Step1", "Step2", "Step3", "Step4"]
    err_idx = evaluate_processbench_style(prm_model, test_steps, tokenizer=tokenizer)
    if err_idx == -1:
        print("PRM 认为所有步骤都正确 (在此示例中只是一种模拟)。")
    else:
        print(f"PRM 认为第 {err_idx} 步是第一个错误的步骤。")


if __name__ == "__main__":
    main_demo()