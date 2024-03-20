import sys

from lightning_fabric import seed_everything

sys.path.append("..")
sys.path.append("../..")

import torch
from datasets import load_dataset
import numpy as np
from transformers import TrainingArguments, Trainer, LlamaForSequenceClassification, LlamaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import (LoraConfig, get_peft_model, TaskType)
import pickle

from config.constants import HF_CACHE

dataset_name = "ag_news"
output_dir = "../"  # Directory to save the model
model_name = "/home/yonatanto/work/theza/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf"

batch_size = 2
MAX_LEN = 370
seed = 42

seed_everything(seed)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis = 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'weighted', warn_for = tuple())
    acc = accuracy_score(labels, preds)
    return dict(accuracy = acc, f1 = f1, precision = precision, recall = recall)


dataset = load_dataset(dataset_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast = False, trust_remote_code = True,
                                           padding_side = "left")
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation = True, padding = "max_length", max_length = MAX_LEN)


train_dataset = dataset["train"].select(range(20)).shuffle(seed = 1)
# tmp_train_ds = train_dataset.train_test_split(test_size = 6_000, seed = 42, stratify_by_column = "label")
# train_dataset = tmp_train_ds["train"]
test_dataset = dataset["train"].select(range(5)).shuffle(seed = 1)

remove_cols = ['text', 'attention_mask']
train_dataset = train_dataset.map(preprocess_function, batched = True, batch_size = batch_size).remove_columns(
    remove_cols)
test_dataset = test_dataset.map(preprocess_function, batched = True, batch_size = batch_size).remove_columns(
    remove_cols)

model = LlamaForSequenceClassification.from_pretrained(model_name,
                                                       num_labels = len(dataset["train"].features['label'].names),
                                                       device_map = 'auto', torch_dtype = torch.bfloat16,
                                                       cache_dir = HF_CACHE)
model.config.pad_token_id = tokenizer.eos_token_id

lora_conf = LoraConfig(r = 3, lora_alpha = 16, task_type = TaskType.SEQ_CLS,
                       # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                       target_modules = ["q_proj"],
                       lora_dropout = 0.1, bias = "none",
                       modules_to_save = ["classifier"])
model = get_peft_model(model, lora_conf)

batch_size = batch_size

training_args = TrainingArguments(output_dir = output_dir, learning_rate = 4e-5, remove_unused_columns = False,
                                  per_device_train_batch_size = batch_size, per_device_eval_batch_size = batch_size,
                                  num_train_epochs = 5, evaluation_strategy = "steps", save_strategy = "steps",
                                  save_steps = 220, logging_steps = 40, eval_steps = 40, save_total_limit = 1,
                                  report_to = ["tensorboard"])

trainer = Trainer(model = model, args = training_args, train_dataset = train_dataset, eval_dataset = test_dataset,
                  compute_metrics = compute_metrics)

trainer.train()

evaluation_results = trainer.evaluate(test_dataset)
print(evaluation_results)

with open(f"{output_dir}/_evaluation_results_.pkl", 'wb') as file:
    pickle.dump(evaluation_results, file)

model.save_pretrained(f"{output_dir}")
