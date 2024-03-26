import sys

sys.path.append(".")
sys.path.append("..")

import pickle
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import numpy as np
import torch

from config.constants import HF_CACHE

dataset_name = "ag_news"
output_dir = "/home/yonatanto/work/theza/NEEA/OTHERS/trained_base_models/agn_distillbert"  # Directory to save the model
model_name = "distilbert-base-uncased"

batch_size = 32
MAX_LEN = 350
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis = 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'weighted', warn_for = tuple())
    acc = accuracy_score(labels, preds)
    return dict(accuracy = acc, f1 = f1, precision = precision, recall = recall)


dataset = load_dataset(dataset_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation = True, padding = "max_length", max_length = MAX_LEN)


train_dataset = dataset["train"].shuffle(seed = 1)
tmp_train_ds = train_dataset.train_test_split(test_size = 6_000, seed = 42, stratify_by_column = "label")
train_dataset = tmp_train_ds["train"]
eval_dataset = tmp_train_ds["test"]

test_dataset = dataset["test"]

train_dataset = train_dataset.map(preprocess_function, batched = True, batch_size = batch_size)
eval_dataset = eval_dataset.map(preprocess_function, batched = True, batch_size = batch_size)
test_dataset = test_dataset.map(preprocess_function, batched = True, batch_size = batch_size)

model = DistilBertForSequenceClassification.from_pretrained(model_name,
                                                            num_labels = len(dataset["train"].features['label'].names),
                                                            cache_dir = HF_CACHE)
model.cuda()


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'targets': targets}


batch_size = batch_size

training_args = TrainingArguments(output_dir = output_dir, learning_rate = 4e-5,
                                  per_device_train_batch_size = batch_size, per_device_eval_batch_size = batch_size,
                                  num_train_epochs = 5, evaluation_strategy = "steps", save_strategy = "steps",
                                  save_steps = 150, logging_steps = 40, save_total_limit = 40,
                                  report_to = ["tensorboard"])

trainer = Trainer(model = model, args = training_args, train_dataset = train_dataset, eval_dataset = eval_dataset,
                  compute_metrics = compute_metrics)

trainer.train()

evaluation_results = trainer.evaluate(test_dataset)
print(evaluation_results)

with open(f"{output_dir}/_evaluation_results_.pkl", 'wb') as file:
    pickle.dump(evaluation_results, file)

model.save_pretrained(f"{output_dir}/last_ckp")
