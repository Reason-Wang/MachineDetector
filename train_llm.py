import os
from dataclasses import field, dataclass
from typing import Optional, Any
import huggingface_hub
import numpy as np
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from data.dataset import load_SubtaskA_data, get_prompts_and_labels
huggingface_hub.login("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")
import torch
import transformers
from transformers import Trainer, GPTNeoXTokenizerFast, BitsAndBytesConfig
from data.dataset import CausalLMDataset, CausalLMCollator
from typing import List
import logging
from sklearn import metrics

logging.basicConfig(level=logging.INFO)

def evaluate(model, tokenizer, test_sources, test_targets, max_text_length):
    print("Evaluating...")
    label_1 = "Yes"
    label_0 = "No"
    label_1_id = tokenizer.encode(label_1, add_special_tokens=False)[0]
    label_0_id = tokenizer.encode(label_0, add_special_tokens=False)[0]
    preds = []
    model.eval()
    test_prompts, _ = get_prompts_and_labels(test_sources, test_targets, tokenizer, max_text_length)
    for test_prompt in tqdm(test_prompts):
        inputs = tokenizer(test_prompt, return_tensors="pt")
        logits = model(**inputs).logits
        logits = logits[0, -1, :][[label_0_id, label_1_id]]
        pred = torch.argmax(torch.softmax(logits, dim=0))
        preds.append(pred.item())
    preds = np.array(preds)
    labels = np.array(test_targets)
    macro_f1 = metrics.f1_score(labels, preds, average='macro')
    micro_f1 = metrics.f1_score(labels, preds, average='micro')
    accuracy = metrics.accuracy_score(labels, preds)
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'accuracy': accuracy
    }



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat-hf")
    architecture: str = field(default='causal')
    data_path: str = field(default="./alpaca_instructions_df.pkl")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size = 8
    learning_rate: float = 5e-5
    max_text_length: int = 512
    num_train_epochs: int = 3
    num_samples: int = None


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def train():
    parser = transformers.HfArgumentParser((TrainingArguments, LoraArguments))
    args, lora_args = parser.parse_args_into_dataclasses()

    compute_dtype = (
        torch.float16
        if args.fp16
        else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    # LlamaTokenizer seems not compatible with AutoTokenizer
    if "llama" in args.model_name_or_path.lower():
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        model = transformers.LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.model_name_or_path,
            use_fast=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        )
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train, test = load_SubtaskA_data(args.num_samples)
    train_sources = [e['text'] for e in train]
    train_targets = [e['label'] for e in train]
    test_sources = [e['text'] for e in test]
    test_targets = [e['label'] for e in test]

    dataset = CausalLMDataset(tokenizer, train_sources, train_targets, 2048, args.max_text_length)
    print(dataset[0])
    collator = CausalLMCollator(tokenizer)

    trainer = Trainer(
        model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)
    result = evaluate(model, tokenizer, test_sources, test_targets, args.max_text_length)
    print(result)
    with open(args.output_dir + '/result.txt', "w") as file:
        file.write(str(result))

'''
deepspeed --num_gpus=4 train.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --deepspeed src/deepspeed_z3_config.json \
  --cache_dir /root/autodl-tmp/llama/hf \
  --architecture causal \
  --output_dir /root/autodl-tmp/InstructLLM/ckpts \
  --save_strategy no \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_p3_data 2000 \
  --num_code_data 0 \
  --num_instruction_data 0 \
  --simple_responses False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --gradient_checkpointing False \
  --bf16 \
  --logging_steps 10

python train_lora.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --architecture causal \
  --output_dir ckpts/llama-2-7b/ \
  --save_strategy "no" \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing False \
  --cache_dir /root/autodl-tmp/llama/hf \
  --fp16 True \
  --logging_steps 1
'''

if __name__ == "__main__":
    train()