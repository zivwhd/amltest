import sys

sys.path.append("../..")

from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

from config.constants import HF_CACHE, LOCAL_MODELS_PREFIX

# Load SST-2 dataset

llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf"
# Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(llama_model)

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation = True, padding = "max_length", max_length = 20)


def preprocess(ds):
    ds = ds.map(tokenize_function, batched = False).rename_column("label", "labels")
    ds.set_format("torch")
    return ds


dataset = load_dataset("imdb")
train_dataset = preprocess(dataset["train"].select(range(10)))
test_dataset = preprocess(dataset["test"].select(range(5)))

# Model configuration
model = LlamaForSequenceClassification.from_pretrained(llama_model, cache_dir = HF_CACHE, num_labels = 2)

# Define training arguments
training_args = TrainingArguments(output_dir = "./results",  #
                                  num_train_epochs = 3,  #
                                  per_device_train_batch_size = 4,  #
                                  per_device_eval_batch_size = 4,  #
                                  logging_dir = "./logs",  #
                                  logging_steps = 100, save_steps = 1000,  #
                                  evaluation_strategy = "steps",  #
                                  eval_steps = 500,  #
                                  dataloader_num_workers = 4,  #
                                  gradient_accumulation_steps = 1,  #
                                  # fp16 = True,
                                  report_to = ["tensorboard"],  #
                                  logging_first_step = True,  #
                                  load_best_model_at_end = True,  #
                                  metric_for_best_model = "accuracy",  #
                                  greater_is_better = True,  #
                                  run_name = "gpt2-sst2-finetuning"  #
                                  )

# Initialize Trainer
trainer = Trainer(model = model, args = training_args, train_dataset = train_dataset, eval_dataset = test_dataset,
                  tokenizer = tokenizer)

# Start training
trainer.train()
trainer.evaluate()

trainer.save_model("./")
