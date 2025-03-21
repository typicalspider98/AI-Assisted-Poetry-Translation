import redis
import json
import os
import sys
import subprocess

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
                print("Linux：运行命令 sudo systemctl start docker")
    except Exception as e:
        print(f"❌ 启动 Docker 容器失败: {e}")
        print("💡 提示：请确保 Docker 服务已启动。")
        print("Windows/Mac：启动 Docker Desktop 应用")
        print("Linux：运行命令 sudo systemctl start docker")

def check_docker_status():
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if DOCKER_CONTAINER_NAME in result.stdout:
            print(f"✅ Docker 容器 '{DOCKER_CONTAINER_NAME}' 正在运行")
            return True
        else:
            print(f"⚠️ Docker 容器 '{DOCKER_CONTAINER_NAME}' 未运行")
            return False
    except Exception as e:
        print(f"❌ 检查 Docker 状态失败: {e}")
        return False

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

def get_db_index_by_key(key):
    for db_index in range(16):  # 默认支持 0~15
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db_index)
        if r.exists(key):
            print(f"🔍 发现 key '{key}' 存在于数据库 {db_index}")
            return db_index
    print(f"❌ 未找到 key '{key}'")
    return None

def query_key(key, db=0):
    r = connect_to_redis(db)
    value = r.get(key)
    if value:
        print(f"🔍 {key} -> {value[:80]}...")  # 简略显示
    else:
        print(f"⚠️ key '{key}' 不存在")

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
    print("2. 查找 Key 所在数据库")
    print("3. 查询指定 Key 的值")
    print("4. 清空指定数据库")
    print("5. 从文件夹导入 JSON 数据到数据库")
    print("6. 检查 Docker Redis 容器状态")
    print("7. 启动 Docker Redis 容器")
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
            key = input("请输入要查找的 Key：")
            get_db_index_by_key(key)
        elif choice == "3":
            db = int(input("请输入数据库编号 (0~15)："))
            key = input("请输入要查询的 Key：")
            query_key(key, db)
        elif choice == "4":
            db = int(input("请输入要清空的数据库编号 (0~15)："))
            clear_database(db)
        elif choice == "5":
            folder = input("请输入 JSON 文件夹路径：")
            db = int(input("请输入目标数据库编号 (0~15)："))
            import_from_json_folder(folder, db)
        elif choice == "6":
            check_docker_status()
        elif choice == "7":
            start_docker_container()
        else:
            print("⚠️ 无效的选项，请重新输入。")
