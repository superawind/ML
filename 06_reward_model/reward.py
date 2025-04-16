from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch

tokenizer = AutoTokenizer.from_pretrained("E:/Model/Qwen2-0.5B-Instruct")
model = AutoModelForSequenceClassification.from_pretrained(
    "E:/Model/Qwen2-0.5B-Instruct", num_labels=1, torch_dtype=torch.float16, device_map="cuda:0"
)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset('parquet', data_files=['C:/Users/zxd/Desktop/00_手写实现/06_dpo/data/test-00000-of-00001.parquet'], split="train").select(range(17))
# dataset = load_dataset("./data", split="train").select(range(17))

training_args = RewardConfig(output_dir="Qwen2.5-0.5B-Reward", per_device_train_batch_size=1, max_length=512)

trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset
)
trainer.train()