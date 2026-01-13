
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="medplib-7b-stage2"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"
python -m deepspeed.launcher.runner --include=localhost:0,1,2,3 --master_port=65001 train_ds_medplib.py \
  --version="/mnt/disk4t0/publicData/huggingface_models/llava-v1.5-7b" \
  --vision_tower='/home/wuhanqing/MedPLIB/huggingface_models/clip-vit-large-patch14-336' \
  --pretrain_mm_mlp_adapter='/mnt/disk4t0/publicData/huggingface_models/llava-v1.5-7b/mm_projector.bin \
  --data_path='/mnt/disk4t0/publicData/MeCoVQA/MeCoVQA_Complex+Region_VQA_train+Public_VQA.json' \
  --val_data_path='/mnt/disk4t0/publicData/MeCoVQA/MeCoVQA_Complex_VQA_test_rand200.json' \
  --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
  --vision_pretrained="/mnt/disk4t0/huggingface_models/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=3 \
  --batch_size=16 \
  --workers=8 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 2 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 5.0 \
  --bce_loss_weight 1.0 \
  --lora_r 16 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --sft_modules "lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector" \
  --lr 0.0001 \
  --no_eval \
  --save_steps 400 \
  2>&1|tee -a runs/$exp_name/$time.log
