python train.py \
  --name test_run \
  --model_name_or_path bert-base-uncased \
  --batch_size 16 \
  --seed 42 \
  --scheduler cosine \
  --lr 2e-5 \
  --epochs 3 \
  --num_samples 50
