import sys

sys.path.append("..")

from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

from config.constants import HF_CACHE, LOCAL_MODELS_PREFIX

# Load SST-2 dataset
dataset = load_dataset("sst2")

llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf"
print(llama_model)
# Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(llama_model)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Model configuration
model = LlamaForSequenceClassification.from_pretrained(llama_model, cache_dir = HF_CACHE, num_labels = 2)

# Define training arguments
training_args = TrainingArguments(output_dir = "./results", num_train_epochs = 3, per_device_train_batch_size = 8,
                                  per_device_eval_batch_size = 8, logging_dir = "./logs", logging_steps = 100,
                                  save_steps = 1000, evaluation_strategy = "steps", eval_steps = 500,
                                  dataloader_num_workers = 4, gradient_accumulation_steps = 2, fp16 = True,
                                  # Enable mixed precision training if supported
                                  report_to = ["tensorboard"],  # Optional: for logging to TensorBoard
                                  logging_first_step = True, load_best_model_at_end = True,
                                  metric_for_best_model = "accuracy", greater_is_better = True,
                                  run_name = "gpt2-sst2-finetuning")

# Initialize Trainer
trainer = Trainer(model = model, args = training_args, train_dataset = dataset["train"].select(range(300)),
                  eval_dataset = dataset["validation"].select(range(300)), tokenizer = tokenizer)

# Start training
trainer.train()
