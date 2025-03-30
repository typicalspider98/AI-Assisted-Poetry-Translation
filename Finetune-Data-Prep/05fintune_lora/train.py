import json
import torch

import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.data_utils import format_dataset, parse_output
'''
pip install datasets
pip install -U peft

'''

# 配置加载函数
def load_configs(model_name):
    """动态加载配置"""
    base_dir = Path(__file__).parent
    
    # 加载训练配置
    with open(base_dir/"configs"/"train_config.json") as f:
        train_config = json.load(f)
    
    # 加载LoRA配置
    lora_config_path = base_dir/"configs"/"lora_config.json"
    with open(lora_config_path) as f:
        lora_config = json.load(f)

    # 移除非法字段（如 _comment）
    lora_config = {k: v for k, v in lora_config.items() if not k.startswith("_")}
    
    # 自动检测模型类型
    if "qwen" in model_name.lower():
        lora_config["target_modules"] = ["c_attn", "c_proj", "w1", "w2"]
    elif "llama" in model_name.lower():
        lora_config["target_modules"] = ["q_proj", "v_proj"]
    
    return train_config, LoraConfig(**lora_config)

def main():
    # 基础配置
    MODEL_PATH = "/workspace/AI-Assisted-Poetry-Translation/C2NZE/models/DeepSeek-R1-Distill-Qwen-14B"
    DATA_PATH = "/workspace/AI-Assisted-Poetry-Translation/Finetune-Data-Prep/05fintune_lora/dataset/唐_part1_demo_lora.jsonl"
    OUTPUT_DIR = "output"
    
    # 加载配置
    train_config, peft_config = load_configs(MODEL_PATH)
    
    # ==== 初始化模型 ====
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    # ==== 处理数据 ====
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    formatted_dataset = dataset.map(
        lambda x: format_dataset(x, tokenizer, train_config["max_length"]),
        remove_columns=dataset.column_names
    )
    
    # ==== 准备训练 ====
    model = get_peft_model(base_model, peft_config)
    print(f"可训练参数量: {model.print_trainable_parameters()}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        num_train_epochs=train_config["num_train_epochs"],
        logging_steps=train_config["logging_steps"],
        save_strategy=train_config["save_strategy"],
        bf16=train_config["bf16"],
        optim=train_config["optim"],
        max_grad_norm=train_config["max_grad_norm"],
        warmup_ratio=train_config["warmup_ratio"],
        report_to="none"
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([x['input_ids'] for x in data]),
            'attention_mask': torch.stack([x['attention_mask'] for x in data]),
            'labels': torch.stack([x['labels'] for x in data])
        }
    )
    
    # ==== 开始训练 ====
    print("🚀 开始训练")
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 训练完成，模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()