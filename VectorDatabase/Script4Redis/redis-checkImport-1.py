import numpy as np
import redis

# 连接 Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# 要查询的词条
key = "Sa-0"  # 你想查看的词条
# key = "A.C.-0"  # 你想查看的词条
stored_embedding = r.get(key)  # 使用 Redis 连接实例 r

# 确保 stored_embedding 不为空
if stored_embedding:
    stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)  # 转换为 numpy 数组
    print(f"🔍 Redis 中的 {key} 向量: {stored_embedding[:5]} ...")  # 只显示前 5 个数值
else:
    print(f"⚠️ 词条 {key} 不存在于 Redis")

# Redis 中的 A.C.-0 向量: [-0.20556037  0.33388066 -0.30673784 -0.3723724   0.0624488 ] ...
