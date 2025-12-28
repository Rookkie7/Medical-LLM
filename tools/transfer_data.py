import json
import random
import os

# 配置路径
input_file = "./medical_data/reward/train.json"  # 替换为你原始文件名
output_file = "./data/reward/dpo_reward_medical_2500.jsonl"
sample_count = 2500

# 定义统一的 System Prompt
system_content = "你是一位专业且耐心的全科医生。请根据用户的提问提供专业、严谨且逻辑清晰的医学建议。"


def process_dpo_data():
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. 随机抽样
    if len(lines) > sample_count:
        sampled_lines = random.sample(lines, sample_count)
    else:
        sampled_lines = lines
        print(f"警告：数据总量不足{sample_count}条，已全部提取。")

        # --- 核心修复：自动创建保存目录 ---
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    processed_data = []
    for line in sampled_lines:
        try:
            item = json.loads(line.strip())

            # 2. 构造新字典，确保 system 和 history 在最前面
            new_item = {
                "system": system_content,
                "history": [],  # DPO通常是单轮对齐，history留空列表即可
                "question": item.get("question", ""),
                "response_chosen": item.get("response_chosen", ""),
                "response_rejected": item.get("response_rejected", "")
            }
            processed_data.append(new_item)
        except Exception as e:
            print(f"解析错误: {e}")

    # 3. 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"处理完成！已生成文件: {output_file}")


if __name__ == "__main__":
    process_dpo_data()