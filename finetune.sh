# # prepare 100M data that are evenly distributed across all domains to prevent forgetting during fine-tuning
# python -m main.pretrain_data --output_dir data/pretrain/mistral-16K_100M-even --num_token 16384:100m --config data/config/even.json --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2
output_name=ultragist-mistral-7b-inst-ft

torchrun --nproc_per_node 6 -m main.train \
--lora_tune True --lora_rank 4 --lora_alpha 16 \
--output_dir data/outputs/$output_name \
--model_name_or_path ./checkpoint_spp/ultragist-mistral-7b-inst \
--train_data ultragist:gpt/one_detail_book.train.16K.json \
--max_length 20400 \
--min_length 7200 \
--group_by_stride strict \
--enable_ultragist \
--ultragist_window 2048 \
--ultragist_stride 2048 \
--ultragist_attn step-expansion \
--ultragist_attend_prev False \
--ultragist_sink_size 1 \
--ultragist_ratio 2 4 8 \
--ultragist_ratio_mix step-random \
--ultragist_param q k v o \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--dtype "bf16" \
--per_device_train_batch_size 1 \
--chat_template mistral \
--deepspeed data/deepspeed/stage2.json
