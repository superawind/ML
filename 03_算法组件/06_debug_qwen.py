from transformers.models.qwen2 import Qwen2Config, Qwen2Model
from transformers import Trainer
import torch

def run_qwen2():
    qwen2config = Qwen2Config(vocab_size=32000, hidden_size =4096//2,
                              intermediate_size=11008//2,
                              num_hidden_layers=32//2,
                              num_attention_heads=32//2,
                              num_key_value_heads=2,
                              max_position_embeddings=2048//2)
    
    qwen2model = Qwen2Model(config=qwen2config)

    input_ids = torch.randint(
        low=0, high=qwen2config.vocab_size, size=(4, 30))
    
    res = qwen2model(input_ids)
    print(res)

if __name__ == '__main__':
    run_qwen2()