import json

with open("data/SubtaskA/subtaskA_train_monolingual.jsonl", "r") as f:
    train = [json.loads(line) for line in f.readlines()]
texts = [t["text"] for t in train]
# save texts to json file
with open("data/SubtaskA/subtaskA_train_monolingual.json", "a") as f:
    for text in texts:
        f.write(json.dumps({"text": text}) + "\n")

with open("data/SubtaskA/subtaskA_dev_monolingual.jsonl", "r") as f:
    dev = [json.loads(line) for line in f.readlines()]
texts = [t["text"] for t in dev]
# save texts to txt file
with open("data/SubtaskA/subtaskA_dev_monolingual.json", "a") as f:
    for text in texts:
        f.write(json.dumps({"text": text}) + "\n")
