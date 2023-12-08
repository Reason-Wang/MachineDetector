import json
import os.path

import click
import numpy as np
import torch
import transformers
from peft import LoraConfig, PeftModel
from sklearn import metrics
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import huggingface_hub
huggingface_hub.login("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")
from data.dataset import load_SubtaskA_data, get_prompts_and_labels
from data.format_checker import check_format

def evaluate(model, tokenizer, test_sources, test_targets, test_ids, max_text_length):
    print("Evaluating...")
    label_0 = "No"
    label_1 = "Yes"
    label_0_id = tokenizer.encode(label_0, add_special_tokens=False)[0]
    label_1_id = tokenizer.encode(label_1, add_special_tokens=False)[0]
    preds = []
    model.eval()
    test_prompts, _ = get_prompts_and_labels(test_sources, test_targets, tokenizer, max_text_length)
    print(test_prompts[0])
    has_print = False
    for test_prompt in tqdm(test_prompts):
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if not has_print:
            print(inputs)
            has_print = True
        logits = model(**inputs).logits
        # print(logits.shape)
        logits = logits[0, -1, :][[label_0_id, label_1_id]]
        # print(logits)
        pred = torch.argmax(torch.softmax(logits, dim=0))
        preds.append(pred.item())
    preds = np.array(preds)
    labels = np.array(test_targets)
    # print(preds, labels)
    macro_f1 = metrics.f1_score(labels, preds, average='macro')
    micro_f1 = metrics.f1_score(labels, preds, average='micro')
    accuracy = metrics.accuracy_score(labels, preds)
    preds = preds.tolist()
    predictions = []
    for id, pred in zip(test_ids, preds):
        predictions.append({
            'id': id,
            'label': pred,
        })

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'accuracy': accuracy,
    }, predictions

@click.command()
@click.option('--model_name_or_path', help='Model name or path')
@click.option('--peft_id', type=str, default=None, help='If set, use peft model')
@click.option('--cache_dir', default=None, help='Cache directory')
@click.option('--max_length', type=int, default=2048, help='Max tokenizer length')
@click.option('--max_text_length', type=int, default=512, help='Max text length')
@click.option('--num_samples', type=int, default=None, help='Number of samples')
@click.option('--result_file', type=str, default=None, help='Result file')
def main(
    model_name_or_path: str,
    peft_id: str=None,
    cache_dir: str=None,
    max_length: int=2048,
    max_text_length: int=512,
    num_samples: int=None,
    result_file: str=None,
):
    if peft_id:
        load_in_4bit = True
    else:
        load_in_4bit = False
    compute_dtype = torch.bfloat16
    if "llama" in model_name_or_path.lower():
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_name_or_path,
            max_length=max_length,
            cache_dir=cache_dir,
        )
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            ),
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=model_name_or_path,
            max_length=max_length,
            use_fast=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        )
    if peft_id:
        model = PeftModel.from_pretrained(model, peft_id, device_map="auto")
        print(f"Loaded peft model from {peft_id}")

    train, test = load_SubtaskA_data(num_samples=None) # use all data
    test_sources = [e['text'] for e in test][:num_samples]
    test_targets = [e['label'] for e in test][:num_samples]
    test_ids = [e['id'] for e in test][:num_samples]
    result, predictions = evaluate(model, tokenizer, test_sources, test_targets, test_ids, max_text_length=max_text_length)
    print(result)
    dir = os.path.dirname(result_file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(result_file, 'w') as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + '\n')
    check_result = check_format(result_file)
    result = 'Format is correct' if check_result else 'Something wrong in file format'
    print(result)


if __name__ == "__main__":
    main()