python run.py \
--output_dir E:/code/00_demo/FlagEmbedding/FlagEmbedding/reranker/mytest/save_model \
--model_name_or_path E:/Model/bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 1 \
--dataloader_drop_last True \
--train_group_size 3 \
--max_len 43 \
--weight_decay 0.01 \
--logging_steps 10 


# python run.py --output_dir E:/code/00_demo/FlagEmbedding/FlagEmbedding/reranker/mytest/save_model --model_name_or_path E:/Model/bge-large-zh-v1.5 --train_data ./toy_finetune_data.jsonl --learning_rate 6e-5 --fp16 --num_train_epochs 5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --dataloader_drop_last True --train_group_size 3 --max_len 43 --weight_decay 0.01 --logging_steps 10 