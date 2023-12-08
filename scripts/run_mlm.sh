python run_mlm_no_trainer.py \
  --train_file data/SubtaskA/subtaskA_train_monolingual.json \
  --model_name_or_path roberta-base \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 512 \
  --line_by_line True
