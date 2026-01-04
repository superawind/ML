python run.py \
--output_dir E:/code/00_demo/FlagEmbedding/FlagEmbedding/baai_general_embedding/save_model \
--model_name_or_path E:/Model/bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 32 \
--passage_max_len 64 \
--train_group_size 3 \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 


# python run.py --output_dir E:/code/00_demo/FlagEmbedding/FlagEmbedding/baai_general_embedding/save_model --model_name_or_path E:/Model/bge-large-zh-v1.5 --train_data ./toy_finetune_data.jsonl --learning_rate 1e-5 --fp16 --num_train_epochs 5 --per_device_train_batch_size 2 --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 32 --passage_max_len 64 --train_group_size 3 --negatives_cross_device --logging_steps 10 --save_steps 30 --query_instruction_for_retrieval "" 