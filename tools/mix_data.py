import json
import random

def merge_and_shuffle(file1, file2, output_file):
    combined_data = []

    # 1. 读取第一个文件
    print(f"正在读取文件 1: {file1}...")
    with open(file1, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                combined_data.append(json.loads(line))

    # 2. 读取第二个文件
    print(f"正在读取文件 2: {file2}...")
    with open(file2, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                combined_data.append(json.loads(line))

    print(f"合并完成，总计 {len(combined_data)} 条数据。")

    # 3. 随机打乱（混合）
    print("正在随机打乱...")
    random.seed(42)  # 固定种子保证结果可复现
    random.shuffle(combined_data)

    # 4. 保存结果
    print(f"正在保存至 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("合并并混合成功！")

# 使用方法
merge_and_shuffle("./data/reward/dpo_reward_medical_2500.jsonl", "./data/reward/dpo_zh_500.jsonl", "./data/reward/dpo_reward_3k.jsonl")