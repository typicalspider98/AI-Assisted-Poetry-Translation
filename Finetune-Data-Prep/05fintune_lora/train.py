
# pip install transformers datasets peft accelerate
# OR
# conda install -c conda-forge transformers datasets
# pip install peft accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# ==== 配置 ====
MODEL_NAME = "../C2NZE/models/DeepSeek-R1-Distill-Qwen-14B"
DATA_PATH = "dataset/唐_part1_lora.jsonl"
OUTPUT_DIR = "output/checkpoints"

# ==== 加载数据 ====
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 将多字段拼接为 prompt + 输出文本
def format_example(example):
    return {
        "input_ids": tokenizer(
            f"{example['instruction']}\n\n{example['input']}",
            truncation=True, padding="max_length", max_length=512
        )["input_ids"],
        "labels": tokenizer(
            example["output"]["keywords"] if isinstance(example["output"], str)
            else "\\n".join(example["output"]["keywords"]),
            truncation=True, padding="max_length", max_length=256
        )["input_ids"]
    }

# ==== 加载模型 ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

# ==== 配置 LoRA ====
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_config)

# ==== 格式化数据 ====
tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# ==== 训练参数 ====
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

# ==== 开始训练 ====
trainer.train()
model.save_pretrained(OUTPUT_DIR)
