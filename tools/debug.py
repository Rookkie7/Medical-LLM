import json
from transformers import AutoTokenizer
# 导入你上传的 template.py 中的类和注册函数
from MedicalGPT.template import get_conv_template

# 1. 配置参数 (根据你的实际路径修改)
MODEL_PATH = "./models/Qwen2.5-3B-Instruct"  # 或者你的本地模型路径
DATA_PATH = "./data/sft_v1_shuffle15k/medical_sharegpt_15k.jsonl"  # 你的数据集路径
TEMPLATE_NAME = "chatml"  # 使用第一个模板
MAX_LENGTH = 512
IGNORE_INDEX = -100


def debug_preprocess():
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    prompt_template = get_conv_template(TEMPLATE_NAME)

    # 读取前5条数据
    examples = {"conversations": [], "system_prompt": []}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5: break
            data = json.loads(line)
            examples["conversations"].append(data["conversations"])
            # 如果数据里没写 system_prompt，给个空的
            examples["system_prompt"].append(data.get("system_prompt", ""))

    print(f"--- 成功加载 {len(examples['conversations'])} 条测试数据 ---\n")

    # 模拟 get_dialog 逻辑
    roles = ["human", "gpt"]

    # 遍历处理
    for i in range(len(examples['conversations'])):
        source = examples['conversations'][i]
        system_prompt_val = examples['system_prompt'][i]

        # 简单清洗逻辑
        messages = []
        for j, sentence in enumerate(source):
            if sentence["from"] == "human":
                messages.append(sentence["value"])
            elif sentence["from"] == "gpt":
                messages.append(sentence["value"])

        if len(messages) % 2 != 0: continue
        history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]

        # 调用模板拼接字符串
        # 这里的 dialog 是一个列表: [Prompt1, Answer1, Prompt2, Answer2...]
        dialog = prompt_template.get_dialog(history_messages, system_prompt=system_prompt_val)

        input_ids = []
        labels = []

        print(f"=== 样本 {i + 1} 详情 ===")
        for j in range(len(dialog) // 2):
            # 这里的逻辑完全复刻你的 supervised_finetuning.py
            source_ids = tokenizer.encode(text=dialog[2 * j], add_special_tokens=(j == 0))
            target_ids = tokenizer.encode(text=dialog[2 * j + 1], add_special_tokens=False)

            # 拼接并加上 EOS
            input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
            # 生成 Labels: User 部分 Mask 掉，Assistant 部分保留
            labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        # --- 打印结果 ---
        full_text = tokenizer.decode(input_ids)
        print("【全量解码文本】:")
        print(full_text)

        print("\n【Loss 计算部分 (Labels)】:")
        # 只解码非 -100 的部分，看看模型到底在学什么
        loss_part = [t for t in labels if t != IGNORE_INDEX]
        print(tokenizer.decode(loss_part))

        print("-" * 50)


if __name__ == "__main__":
    debug_preprocess()