import torch
from peft import LoraConfig
from reward import format_reward, format_reward2
from datasets import load_dataset
from trl_main import GRPOConfig, GRPOTrainer
# from transformers import Qwen2ForCausalLM

dataset = load_dataset('parquet', data_files = ['C:/Users/zxd/Desktop/00_手写实现/data/tldr/test-00000-of-00001.parquet'],)['train']

print(dataset)

training_args = GRPOConfig(
    output_dir="E:/Model/Qwen2-0.5B-Instruct",
    learning_rate=1e-4,
    logging_steps=10,
    gradient_accumulation_steps=2,
    max_completion_length=128,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    num_train_epochs=1,
    save_steps=1000,
)

trainer = GRPOTrainer(
    model="E:/Model/Qwen2-0.5B-Instruct",
    # reward_funcs="model/weqweasdas/RM-Gemma-2B",  # 也可以用 函数 当作reward
    reward_funcs=[format_reward, format_reward2], 
    args=training_args,
    train_dataset=dataset,
    peft_config=LoraConfig(task_type="CAUSAL_LM"),
)

trainer.train()
trainer.save_model(training_args.output_dir)