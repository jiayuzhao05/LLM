import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import pandas as pd

# -------------- 数据预处理函数 --------------
def process_func(example):
    """
    将单条记录的 messages 和 images 转换为模型输入格式
    """
    MAX_LENGTH = 8192

    # 从 example 中读取对话消息和图像路径列表
    chat_messages = example["messages"]      # List[dict(role, content)]
    image_paths = example.get("images", [])  # List[str]

    # 构造 processor 所需的 messages 格式：将图像消息插入到对话开始
    proc_messages = []
    # 如果有图像，插入为图像消息
    for img_path in image_paths:
        proc_messages.append({
            "role": "user",
            "content": [{
                "type": "image",
                "image": img_path
            }]
        })

    # 将文本对话添加进来
    for msg in chat_messages:
        proc_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # 拆分输入和输出
    input_msgs = proc_messages[:-1]
    output_msg = proc_messages[-1]

    # 模型输入预处理
    text = processor.apply_chat_template(
        input_msgs, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(input_msgs)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    )
    inputs = {k: v.tolist() for k, v in inputs.items()}

    # 编码输出标签
    response = tokenizer(
        output_msg["content"], add_special_tokens=False
    )

    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    }

# -------------- 推理函数 --------------
def predict(example, model):
    chat_messages = example["messages"]
    image_paths = example.get("images", [])

    proc_messages = []
    for img_path in image_paths:
        proc_messages.append({"role": "user", "content": [{"type": "image", "image": img_path}]})
    proc_messages.extend(chat_messages)

    text = processor.apply_chat_template(
        proc_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(proc_messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0]

# -------------- 模型加载 --------------
model_dir = snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="./", revision="master")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.enable_input_require_grads()

# -------------- 数据加载与转换 --------------
import pandas as pd
df = pd.read_parquet("data/train-00000-of-00001.parquet")
dataset = Dataset.from_pandas(df)
train_len = len(dataset) - 50
train_ds = dataset.select(range(train_len))
test_ds = dataset.select(range(train_len, len(dataset)))
train_dataset = train_ds.map(process_func)

# -------------- LoRA 配置 --------------
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    inference_mode=False, r=64, lora_alpha=16, lora_dropout=0.05, bias="none",
)
peft_model = get_peft_model(model, config)

# -------------- 训练 --------------
args = TrainingArguments(
    output_dir="./output/Qwen2-VL-2B",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[SwanLabCallback(
        project="Qwen2-VL-finetune", experiment_name="qwen2-vl-coco2014",
        config={
            "model": model_dir,
            "prompt": "COCO Yes: ",
            "train_data_number": train_len,
            "lora_rank": 64, "lora_alpha": 16, "lora_dropout": 0.1,
        }
    )],
)
trainer.train()

# -------------- 测试 --------------
val_peft = PeftModel.from_pretrained(
    model, model_id="./output/Qwen2-VL-2B/checkpoint-62",
    config=LoraConfig(task_type=TaskType.CAUSAL_LM,
                       target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                       inference_mode=True, r=64, lora_alpha=16, lora_dropout=0.05, bias="none")
)

for example in test_ds:
    out = predict(example, val_peft)
    print(out)

swanlab.finish()
