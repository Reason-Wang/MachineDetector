# MachineDetector



```bash
pip install gdown
pip install -r requirements.txt
cd data
gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc
```

## Training Encoder-based Models

```bash
python train.py \
  --name deberta-v3-small \
  --model_name_or_path microsoft/deberta-v3-small \
  --batch_size 16 \
  --seed 0 \
  --scheduler cosine \
  --lr 2e-5 \
  --epochs 2 \
  --max_len 512 \
  --gradient_clipping \
  --apex
```

## Training LLaMA-2-7B

*This may need an A100 with 80G memory to train. Takes about 7 hours.*

```bash
python train_llm.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir ckpts/llama-2-7b/ \
  --save_strategy "no" \
  --learning_rate 4e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --per_device_train_batch_size 2 \
  --max_text_length 512 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing False \
  --max_steps 5000 \
  --fp16 True \
  --logging_steps 1
```

### Inference

```bash
python inference.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --peft_id ckpts/llama-2-7b \
  --result_file results/llama-2-7b.json
```



