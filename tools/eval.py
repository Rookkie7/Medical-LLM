import os

# 1. 设置镜像和缓存路径（解决网络和持久化问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface/models'

# 2. 创建结果目录
results_dir = "/root/autodl-tmp/ceval_results"
os.makedirs(results_dir, exist_ok=True)

# 3. C-Eval的三个医疗任务
tasks = "ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician"
custom_config_path = "./lm-evaluation-harness/lm_eval/tasks/ceval"

# 4. 直接运行命令
# 该命令为加5shot的测试
# cmd = f"""
# lm_eval --model hf \
#     --model_args "pretrained=/root/autodl-tmp/models/merged-sft-v2-dpo,trust_remote_code=true,dtype=bfloat16" \
#     --tasks {tasks} \
#     --device cuda:0 \
#     --batch_size 4 \
#     --num_fewshot 5 \
#     --include_path {custom_config_path} \
#     --output_path {results_dir}/dpo_test_results_5shot.json
# """

cmd = f"""
lm_eval --model hf \
    --model_args "pretrained=/root/autodl-tmp/models/merged-sft-v2-dpo,trust_remote_code=true,dtype=bfloat16" \
    --tasks {tasks} \
    --device cuda:0 \
    --batch_size 4 \
    --include_path {custom_config_path} \
    --output_path {results_dir}/dpo_test_results.json
"""


print("执行命令：")
print(cmd)
print("\n" + "="*50)

os.system(cmd)