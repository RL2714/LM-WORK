nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29501 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /root/data1/LLM/Pretrain/pt_model/Qwen3-VL-7b-pt-260112/v0-20260112-224450/checkpoint-1188 \
    --train_type lora \
    --dataset /root/data1/LLM/All_Data/Qwen2_5+3/sft_data/1932_image_text.jsonl \
              /root/data1/LLM/All_Data/Qwen2_5+3/sft_data/4508_image01_20251031.jsonl \
              /root/data1/LLM/All_Data/Qwen2_5+3/sft_data/finetune_2000.jsonl \
              /root/data1/LLM/All_Data/Qwen2_5+3/sft_data/sft_20250430.jsonl \
              /root/data1/LLM/All_Data/Qwen2_5+3/sft_data/sft_20250509.jsonl \
    --torch_dtype bfloat16 \
    --template qwen3_vl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 12 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 3 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096  \
    --output_dir /root/data1/LLM/Finetune/SFT_model/Qwen3_VL_SFT_20260114 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn