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

VECTOR_JSON_FOLDER = "vectordatabases"
BACKUP_FOLDER = "redis-backup"

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
        if db == 0:
            data = json.loads(value)
            print(f"📘 key '{key}' 的释义内容：")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            vector = np.frombuffer(value, dtype=np.float32)
            print(f"🔍 key '{key}' 的向量维度: {vector.shape[0]}，前5项: {vector[:5]}")
    except Exception:
        print(f"⚠️ key '{key}' 的值无法解析")

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
                        if db_index == 0:
                            decoded = json.loads(value)
                            print(f"    📘 示例 key: {example_key.decode('utf-8')} 是词典释义记录")
                            print(json.dumps(decoded, indent=2, ensure_ascii=False))
                        else:
                            vector = np.frombuffer(value, dtype=np.float32)
                            print(f"    🔍 示例 key: {example_key.decode('utf-8')}，向量维度: {vector.shape[0]}，前5项: {vector[:5]}")
                    except Exception:
                        print(f"    ⚠️ 示例 key: {example_key.decode('utf-8')} 不是有效格式")
        except redis.ConnectionError:
            print(f"❌ 无法连接数据库 {db_index}")
    if not any_data:
        print("⚠️ 所有 Redis 数据库均为空。")

def clear_database(db=0):
    r = connect_to_redis(db)
    r.flushdb()
    print(f"🗑️ 数据库 {db} 已清空")

def import_all_from_vector_folder():
    folder = VECTOR_JSON_FOLDER
    if not os.path.exists(folder):
        print(f"❌ 文件夹不存在: {folder}")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".json") and f[0].isdigit()]
    if not files:
        print("⚠️ 未发现符合命名规则的 JSON 文件。格式示例：0-definitions.json 或 1-vectors.json")
        return

    for file in files:
        db_index = int(file.split("-")[0])
        r = connect_to_redis(db_index)

        if r.dbsize() > 0:
            print(f"⛔ 数据库 {db_index} 非空，跳过导入文件: {file}")
            continue

        path = os.path.join(folder, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"❌ 文件格式错误（不是 dict）：{file}，跳过。")
            continue

        print(f"📥 正在导入 {file} 到数据库 {db_index}...")

        for key, value in data.items():
            try:
                if db_index == 0:
                    if isinstance(value, dict):
                        r.set(key, json.dumps(value, ensure_ascii=False))
                        print(f"📘 [释义] 导入 key: {key}")
                    else:
                        print(f"⚠️ 非法释义结构，已跳过 key: {key}")
                else:
                    if isinstance(value, list):
                        binary_value = np.array(value, dtype=np.float32).tobytes()
                        r.set(key, binary_value)
                        print(f"📦 [向量] 导入 key: {key}")
                    elif isinstance(value, str):
                        try:
                            binary_value = bytes.fromhex(value)
                            r.set(key, binary_value)
                            print(f"📦 [向量(hex)] 导入 key: {key}")
                        except Exception:
                            print(f"⚠️ 无法解析 hex 向量值，跳过 key: {key}")
                    else:
                        print(f"⚠️ 非法向量结构，已跳过 key: {key}")
            except Exception as e:
                print(f"⚠️ 跳过 key: {key}，原因: {e}")

        print(f"✅ 文件 {file} 导入完成")

def export_database_to_file():
    db = int(input("请输入要导出的数据库编号 (0~15)："))
    r = connect_to_redis(db)
    if not os.path.exists(BACKUP_FOLDER):
        os.makedirs(BACKUP_FOLDER)
    output_path = os.path.join(BACKUP_FOLDER, f"{db}-backup.json")
    all_data = {}
    for key in r.scan_iter():
        value = r.get(key)
        try:
            key_str = key.decode("utf-8")
            if db == 0:
                decoded_value = json.loads(value)
                all_data[key_str] = decoded_value
            else:
                vector = np.frombuffer(value, dtype=np.float32)
                all_data[key_str] = vector.tolist()
        except Exception as e:
            print(f"⚠️ 跳过 key: {key}，原因: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"💾 数据库 {db} 已备份至 {output_path}")

def show_menu():
    print("\n🔧 Redis 数据库管理菜单：")
    print("1. 检查 Redis 服务是否运行")
    print("2. 查看指定 Key 的内容（释义或向量）")
    print("3. 查看当前各数据库的数据量和示例记录")
    print("4. 清空指定数据库")
    print("5. 自动导入 vectordatabases 文件夹中的 JSON 到对应数据库")
    print("6. 导出指定数据库内容到 redis-backup 文件夹")
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
            import_all_from_vector_folder()
        elif choice == "6":
            export_database_to_file()
        else:
            print("⚠️ 无效的选项，请重新输入。")
