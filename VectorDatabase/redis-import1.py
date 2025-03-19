import json
import redis
import numpy as np

# 连接 Redis
r = redis.Redis(host="localhost", port=6379, db=0)
def restore_redis(file_path="redis_backup.json"):
    """
    从 JSON 备份文件恢复 Redis 数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        backup_data = json.load(f)

    for key, value in backup_data.items():
        stored_embedding = bytes.fromhex(value)  # 转换回二进制格式
        r.set(key, stored_embedding)

    print(f"✅ Redis 数据已从 {file_path} 恢复")

# 执行恢复
restore_redis()
