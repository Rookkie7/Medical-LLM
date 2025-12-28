# import json
# import os
# import matplotlib.pyplot as plt
#
#
# def plot_dpo_metrics(output_dir):
#     # 1. 加载训练状态数据
#     state_path = os.path.join(output_dir, 'trainer_state.json')
#     if not os.path.exists(state_path):
#         print(f"错误：找不到文件 {state_path}")
#         return
#
#     with open(state_path, 'r') as f:
#         state = json.load(f)
#
#     # 2. 提取数据点
#     log_history = state['log_history']
#
#     steps = []
#     loss = []
#     rewards_acc = []
#     rewards_margin = []
#     lr = []
#
#     for entry in log_history:
#         # 只提取包含训练指标的 entry (过滤掉 eval 或仅有 epoch 的行)
#         if 'loss' in entry and 'rewards/accuracies' in entry:
#             steps.append(entry['step'])
#             loss.append(entry['loss'])
#             rewards_acc.append(entry['rewards/accuracies'])
#             rewards_margin.append(entry['rewards/margins'])
#             lr.append(entry['learning_rate'])
#
#     if not steps:
#         print("警告：没有在日志中提取到有效的 DPO 指标数据点！")
#         return
#
#     # 3. 开始绘图
#     plt.figure(figsize=(15, 10))
#     plt.suptitle(f"DPO Training Analysis: {output_dir}", fontsize=16)
#
#     # 图 1: Loss 曲线
#     plt.subplot(2, 2, 1)
#     plt.plot(steps, loss, color='red', label='Train Loss')
#     plt.xlabel('Steps')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#
#     # 图 2: Accuracy 曲线
#     plt.subplot(2, 2, 2)
#     plt.plot(steps, rewards_acc, color='blue', label='Reward Accuracy')
#     plt.axhline(y=0.5, color='black', linestyle=':', label='Baseline (0.5)')
#     plt.xlabel('Steps')
#     plt.ylabel('Accuracy')
#     plt.title('Reward Accuracy (Chosen > Rejected)')
#     plt.ylim(0, 1.05)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#
#     # 图 3: Reward Margin 曲线
#     plt.subplot(2, 2, 3)
#     plt.plot(steps, rewards_margin, color='green', label='Reward Margin')
#     plt.xlabel('Steps')
#     plt.ylabel('Margin')
#     plt.title('Reward Margin (Chosen - Rejected)')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#
#     # 图 4: Learning Rate 曲线
#     plt.subplot(2, 2, 4)
#     plt.plot(steps, lr, color='purple', label='Learning Rate')
#     plt.xlabel('Steps')
#     plt.ylabel('LR')
#     plt.title('Learning Rate Decay')
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     # 保存图片
#     save_name = './images/dpo_training_curves.png'
#     plt.savefig(save_name)
#     print(f"✅ 图表已保存为: {save_name}")
#     plt.show()
#
#
# # 使用方法：
# # 指定你的训练输出目录即可
# plot_dpo_metrics('./outputs-sft-v2-dpo')


import json
import os
import matplotlib.pyplot as plt


def plot_dpo_eval_metrics(output_dir):
    # 1. 读取 trainer_state.json
    state_path = os.path.join(output_dir, 'trainer_state.json')
    if not os.path.exists(state_path):
        print(f"错误：找不到文件 {state_path}")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    # 2. 提取评估数据
    eval_steps = []
    eval_loss = []
    eval_acc = []
    eval_margin = []

    for entry in state['log_history']:
        # 只有包含 eval_loss 的条目才是评估点
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
            # 部分版本键名可能略有不同，做个兼容处理
            eval_acc.append(entry.get('eval_rewards/accuracies', 0))
            eval_margin.append(entry.get('eval_rewards/margins', 0))

    if not eval_steps:
        print("警告：日志中没有找到 Eval 评估记录点！")
        return

    # 3. 绘图
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"DPO Evaluation Metrics Analysis", fontsize=16)

    # 子图 1: Eval Loss (判断是否过拟合)
    plt.subplot(2, 1, 1)
    plt.plot(eval_steps, eval_loss, 'o-', color='darkred', linewidth=2, label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss (Down is better)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图 2: Eval Accuracy & Margin (核心表现)
    ax2 = plt.subplot(2, 1, 2)
    lns1 = ax2.plot(eval_steps, eval_acc, 's-', color='darkblue', linewidth=2, label='Eval Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('Steps')

    # 在同一个图中使用双坐标轴画 Margin
    ax3 = ax2.twinx()
    lns2 = ax3.plot(eval_steps, eval_margin, '^--', color='darkgreen', linewidth=2, label='Eval Margin')
    ax3.set_ylabel('Margin')

    # 合并图例
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='lower right')

    plt.title('Eval Accuracy & Reward Margin')
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存结果
    save_name = './images/dpo_eval_analysis.png'
    plt.savefig(save_name, dpi=300)
    print(f"✅ Eval 分析图已保存为: {save_name}")
    plt.show()


# 执行绘图
plot_dpo_eval_metrics('./outputs-sft-v2-dpo')