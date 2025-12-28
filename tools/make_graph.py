import json
import matplotlib.pyplot as plt
import os

# 1. 设置路径
log_path = "./outputs-sft-qwen-v2/trainer_state.json"

if not os.path.exists(log_path):
    print(f"找不到文件: {log_path}")
else:
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 提取数据
    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []

    for entry in data["log_history"]:
        if "loss" in entry:
            train_loss.append(entry["loss"])
            train_steps.append(entry["step"])
        elif "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_steps.append(entry["step"])

    # --- 第一张图：Train Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, color='#1f77b4', label='Training Loss')
    plt.title('Training Loss per Step')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('./images/train_loss_sft_v2.png', dpi=300)
    plt.close()
    print("训练 Loss 图已保存为: train_loss.png")

    # --- 第二张图：Eval Loss ---
    if eval_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, eval_loss, color='#d62728', marker='o', linestyle='-', label='Evaluation Loss')
        plt.title('Evaluation Loss per 100 Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('./images/eval_loss_sft_v2.png', dpi=300)
        plt.close()
        print("验证 Loss 图已保存为: eval_loss.png")
    else:
        print("日志中未发现 eval 数据，请确认训练命令中是否开启了 --do_eval")