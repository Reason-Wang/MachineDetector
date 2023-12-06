from sklearn.model_selection import train_test_split
from data.dataset import load_SubtaskA_data, ClassificationDataset, ClassificationCollator
from model.custom_model import CustomModel
from util.option import Options
from util.utils import AverageMeter, get_logger, seed_everything, get_optimizer_params
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoTokenizer
import time
import torch
from tqdm import tqdm
import sys
import numpy as np
from sklearn import metrics
import gc

def validate_fn(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    # tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    start = end = time.time()

    with torch.no_grad():
        # for idx, (inputs, labels) in enumerate(tbar):
        for step, (inputs, labels) in enumerate(val_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            pred = model(inputs)
            # print(pred.shape)
            preds.append(pred.detach().cpu().numpy())
            loss = criterion(pred.squeeze(dim=1), labels)
            losses.update(loss.item(), batch_size)
            end = time.time()
    predictions = np.concatenate(preds, axis=0)
    return losses.avg, predictions


def train_fn(model, train_loader, criterion, optimizer, epoch, scheduler, device):
    # def train_fn(model, train_loader, val_loader, val_ds, criterion, optimizer, epoch, device):
    model.train()
    tbar = tqdm(train_loader, file=sys.stdout)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.apex)
    losses = AverageMeter()
    global_step = 0
    for step, (inputs, labels) in enumerate(tbar):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        pred = model(inputs)
        loss = criterion(pred.squeeze(dim=1), labels)
        if opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        grad_norm = 0
        if (step + 1) % opt.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()

        # end = time.time()
        # if CFG.do_eval and (step % CFG.eval_freq == CFG.eval_freq - 1):
        #     # eval
        #     avg_val_loss, predictions = validate_fn(model, val_loader, criterion, device)
        #     # scoring
        #     score = get_score(val_ds, predictions)
        #     LOGGER.info(f"step {step}: score {score:.4f}")

        tbar.set_description(
            f"Epoch {epoch + 1} Loss: {losses.avg:.4f} lr: {scheduler.get_last_lr()[0]:.8f} grad_norm: {grad_norm:.2f}")
        # tbar.set_description(f"Epoch {epoch+1} Loss: {losses.avg:.4f} lr: {CFG.lr:.8f} grad_norm: {grad_norm:.2f}")

    return losses.avg


def train_loop(train_ds, val_ds, opt):
    LOGGER.info(f"========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    model = CustomModel(opt.model_name_or_path)
    model.to(device)

    collator = ClassificationCollator(tokenizer, opt.max_len)
    val_loader = DataLoader(val_ds,
                            batch_size=opt.batch_size * 2,
                            shuffle=False,
                            collate_fn=collator,
                            num_workers=opt.num_workers,
                            pin_memory=False,
                            drop_last=False)

    train_loader = DataLoader(train_ds,
                              batch_size=opt.batch_size,
                              shuffle=not opt.no_shuffle_train,
                              collate_fn=collator,
                              num_workers=opt.num_workers,
                              pin_memory=False,
                              drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    optimizer_parameters = get_optimizer_params(model, opt)
    optimizer = AdamW(optimizer_parameters)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(opt, optimizer, num_train_steps):
        if opt.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_train_steps*opt.warmup_ratio, num_training_steps=num_train_steps
            )
        elif opt.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=num_train_steps*opt.warmup_ratio, num_training_steps=num_train_steps,
                num_cycles=opt.num_cycles
            )
        else:
            raise RuntimeError(f"scheduler {opt.scheduler} not supported")
        return scheduler


    num_train_steps = int(opt.epochs * len(train_ds) / (opt.batch_size * opt.gradient_accumulation_steps))
    scheduler = get_scheduler(opt, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    best_score = 0.0

    for epoch in range(opt.epochs):
        start_time = time.time()
        # train
        avg_loss = train_fn(model, train_loader, criterion, optimizer, epoch, scheduler, device)
        # avg_loss = train_fn(model, train_loader, val_loader, val_ds, criterion, optimizer, epoch, device)
        # eval
        avg_val_loss, predictions = validate_fn(model, val_loader, criterion, device)

        # scoring
        score = get_score(val_ds, predictions)['accuracy']

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR + f"{opt.model_name_or_path.replace('/', '_')}.pth")

    torch.cuda.empty_cache()
    gc.collect()

    return val_ds


def get_score(val_ds, preds):
    gold = []
    for i in range(len(val_ds)):
        gold_label = val_ds[i]['label']
        gold.append(gold_label)
    preds = preds > 0.5
    macro_f1 = metrics.f1_score(gold, preds, average='macro')
    micro_f1 = metrics.f1_score(gold, preds, average='micro')
    accuracy = metrics.accuracy_score(gold, preds)
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'accuracy': accuracy
    }



def get_final_score(ds, opt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    collator = ClassificationCollator(tokenizer, opt.max_len)
    data_loader = DataLoader(ds,
                        batch_size=opt.batch_size * 2,
                        shuffle=False,
                        collate_fn=collator,
                        num_workers=opt.num_workers,
                        pin_memory=False,
                        drop_last=False)

    criterion = nn.CrossEntropyLoss()
    model_state = torch.load(OUTPUT_DIR+f"{opt.model_name_or_path.replace('/', '_')}.pth")
    model = CustomModel(opt.model_name_or_path)
    model.to(device)
    model.load_state_dict(model_state)
    avg_val_loss, predictions = validate_fn(model, data_loader, criterion, device)
    score = get_score(ds, predictions)

    LOGGER.info(f'Score: {score}')
    return score


if __name__ == '__main__':
    options = Options()
    opt = options.parse()[0]

    OUTPUT_DIR = 'data/ckpts/' + opt.name + '/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    LOGGER = get_logger(OUTPUT_DIR)
    seed_everything(opt.seed)

    # Load data
    train, test = load_SubtaskA_data(opt.num_samples)
    train, dev = train_test_split(train, test_size=0.1, random_state=opt.seed)

    # Get dataset
    train_texts = [e['text'] for e in train]
    train_labels = [e['label'] for e in train]
    train_ds = ClassificationDataset(train_texts, train_labels)
    dev_texts = [e['text'] for e in dev]
    dev_labels = [e['label'] for e in dev]
    dev_ds = ClassificationDataset(dev_texts, dev_labels)
    test_texts = [e['text'] for e in test]
    test_labels = [e['label'] for e in test]
    test_ds = ClassificationDataset(test_texts, test_labels)

    LOGGER.info(train_ds[0])

    train_loop(train_ds, dev_ds, opt)
    dev_score = get_final_score(dev_ds, opt)
    test_score = get_final_score(test_ds, opt)

    paras_str = options.get_options(opt)
    paras_str = paras_str + '\n' + f'dev: {dev_score} \ntest: {test_score}'
    LOGGER.info(paras_str)
    with open(OUTPUT_DIR + '/result.txt', "w") as file:
        file.write(paras_str)