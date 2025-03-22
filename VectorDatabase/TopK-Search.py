import redis
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 连接 Redis
r = redis.Redis(host="localhost", port=6379, db=2)

# 加载 BGE 模型
MODEL_PATH = "./models/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)


def get_embedding(text):
    """
    计算文本的向量表示（均值池化）
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def search_topk_similar(query, top_k=5):
    """
    在 Redis 中查找最相似的 top_k 词条
    """
    query_vector = get_embedding(query)  # 计算查询的向量
    words = r.keys("*")  # 获取所有存储的 key
    similarities = []

    for word in words:
        stored_vector = np.frombuffer(r.get(word), dtype=np.float32)  # 读取 Redis 存储的向量
        similarity = np.dot(query_vector, stored_vector)  # 计算余弦相似度
        similarities.append((word.decode("utf-8"), similarity))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]  # 返回 top_k 结果


if __name__ == "__main__":
    query = "I like wear gumboot"  # 测试查询词
    # query = "bright moon"  # 测试查询词
    results = search_topk_similar(query, top_k=5)
    print("🔍 最相似的词条:")
    for word, score in results:
        print(f"{word}: {score:.4f}")
