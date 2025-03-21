import redis
import json
import os
import sys
import subprocess
import numpy as np

# 默认连接参数
REDIS_HOST = "localhost"
REDIS_PORT = 6379
DOCKER_CONTAINER_NAME = "redis-server_NZDdictionary"

def start_docker_container():
    try:
        result = subprocess.run([
            "docker", "start", DOCKER_CONTAINER_NAME
        ], capture_output=True, text=True)

        if DOCKER_CONTAINER_NAME in result.stdout:
            print(f"🚀 Docker 容器 '{DOCKER_CONTAINER_NAME}' 已启动")
        else:
            print(f"⚠️ 启动容器失败，返回信息：{result.stderr.strip()}")
            if "Cannot connect to the Docker daemon" in result.stderr:
                print("💡 提示：Docker 服务未运行，请根据操作系统手动启动：")
                print("Windows/Mac：启动 Docker Desktop 应用")
                print("Linux：运行命令 sudo nohup dockerd > dockerd.log 2>&1 &")
                sys.exit(1)
    except Exception as e:
        print(f"❌ 启动 Docker 容器失败: {e}")
        print("💡 提示：请确保 Docker 服务已启动。")
        print("Windows/Mac：启动 Docker Desktop 应用")
        print("Linux：运行命令 sudo systemctl start docker")
        sys.exit(1)

def connect_to_redis(db=0):
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db)
        r.ping()
        print(f"✅ Redis 数据库 {db} 已连接")
        return r
    except redis.ConnectionError:
        print("❌ 无法连接 Redis，请确认服务是否已启动。")
        ask = input("是否尝试启动 Docker Redis 容器？(y/n)：").strip().lower()
        if ask == 'y':
            start_docker_container()
        else:
            print("⚠️ 未启动 Redis 容器，无法继续连接。")
        sys.exit(1)

def check_redis_status():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        print("✅ Redis 服务正在运行")
    except redis.ConnectionError:
        print("❌ Redis 服务未启动")
        ask = input("是否尝试启动 Docker Redis 容器？(y/n)：").strip().lower()
        if ask == 'y':
            start_docker_container()
        else:
            print("⚠️ 用户选择不启动 Redis 容器")

def show_vector_by_key():
    db = int(input("请输入数据库编号 (0~15)："))
    key = input("请输入要查询的 Key：")
    r = connect_to_redis(db)
    if not r.exists(key):
        print(f"❌ 数据库 {db} 中未找到 key '{key}'")
        return
    value = r.get(key)
    try:
        vector = np.frombuffer(value, dtype=np.float32)
        print(f"🔍 key '{key}' 的向量维度: {vector.shape[0]}，前5项: {vector[:5]}")
    except Exception:
        print(f"⚠️ key '{key}' 的值无法转换为向量")

def show_database_summary():
    print("📊 当前 Redis 数据库使用情况：")
    any_data = False
    for db_index in range(16):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db_index)
            size = r.dbsize()
            if size > 0:
                any_data = True
                print(f"  📁 数据库 {db_index} 包含 {size} 个 key")
                example_key = next(r.scan_iter(), None)
                if example_key:
                    value = r.get(example_key)
                    try:
                        vector = np.frombuffer(value, dtype=np.float32)
                        print(f"    🔍 示例 key: {example_key.decode('utf-8')}，向量维度: {vector.shape[0]}，前5项: {vector[:5]}")
                    except Exception:
                        print(f"    ⚠️ 示例 key: {example_key.decode('utf-8')} 不是有效的向量格式")
        except redis.ConnectionError:
            print(f"❌ 无法连接数据库 {db_index}")
    if not any_data:
        print("⚠️ 所有 Redis 数据库均为空。")

def clear_database(db=0):
    r = connect_to_redis(db)
    r.flushdb()
    print(f"🗑️ 数据库 {db} 已清空")

def import_from_json_folder(json_folder, db=0):
    r = connect_to_redis(db)
    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"📁 发现 {len(files)} 个 JSON 文件，开始导入...")
    for file in files:
        path = os.path.join(json_folder, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            word = entry["word"]
            for idx, definition in enumerate(entry["definitions"]):
                key = f"{word}-{idx}"
                content = definition.get("meaning", "") or " ".join(definition.get("examples", []))
                if content:
                    r.set(key, content)
                    print(f"✅ 写入 {key}")
    print("🎉 所有文件已导入")

def show_menu():
    print("\n🔧 Redis 数据库管理菜单：")
    print("1. 检查 Redis 服务是否运行")
    print("2. 查看指定 Key 的向量内容（已解码）")
    print("3. 查看当前各数据库的数据量和示例向量")
    print("4. 清空指定数据库")
    print("5. 从文件夹导入 JSON 数据到数据库")
    print("0. 退出程序")

if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("请输入操作编号：")

        if choice == "0":
            print("👋 程序已退出。")
            break
        elif choice == "1":
            check_redis_status()
        elif choice == "2":
            show_vector_by_key()
        elif choice == "3":
            show_database_summary()
        elif choice == "4":
            db = int(input("请输入要清空的数据库编号 (0~15)："))
            clear_database(db)
        elif choice == "5":
            folder = input("请输入 JSON 文件夹路径：")
            db = int(input("请输入目标数据库编号 (0~15)："))
            import_from_json_folder(folder, db)
        else:
            print("⚠️ 无效的选项，请重新输入。")
