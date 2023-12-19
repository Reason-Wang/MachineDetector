name="deberta-v3-small"
model="microsoft/deberta-v3-small"
max_length="512"
batch_size="16"
lr="2e-5"
lr_scheduler="cosine"
warmup_ratio="0.01"
epochs="2"
random_seed="0"

cmd="python train.py \
  --name ${name} \
  --model_name_or_path ${model} \
  --batch_size ${batch_size} \
  --seed ${random_seed} \
  --scheduler ${lr_scheduler} \
  --lr ${lr} \
  --epochs ${epochs} \
  --max_len ${max_length} \
  --gradient_clipping \
  --apex"

echo $cmd
eval $cmd
