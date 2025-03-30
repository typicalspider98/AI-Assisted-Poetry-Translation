from transformers import AutoTokenizer
import torch

def format_dataset(example, tokenizer, max_length):
    # 构建完整文本
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = parse_output(example["output"])
    
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
    answer = output + tokenizer.eos_token
    
    # Tokenization
    full_text = prompt + answer
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 创建标签掩码
    prompt_tokens = tokenizer(
        prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).input_ids
    prompt_len = prompt_tokens.shape[1]
    
    labels = tokenized["input_ids"].clone()
    labels[:, :prompt_len] = -100  # 忽略prompt部分的loss
    
    return {
        "input_ids": tokenized.input_ids.squeeze(),
        "attention_mask": tokenized.attention_mask.squeeze(),
        "labels": labels.squeeze()
    }

def parse_output(output):
    """统一处理不同格式的output字段"""
    if isinstance(output, dict):
        return "\\n".join(output.get("keywords", []))
    elif isinstance(output, list):
        return "\\n".join(output)
    else:
        return str(output)