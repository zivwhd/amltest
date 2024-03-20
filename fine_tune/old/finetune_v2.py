import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.constants import HF_CACHE

model_name = "roberta-base"

# load data
dataset = load_dataset("emotion")
num_labels = len(set(dataset["train"]["label"]))

# Prepare model
peft_config = LoraConfig(task_type = TaskType.SEQ_CLS, r = 8, lora_alpha = 8, lora_dropout = 0.1)
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/yonatanto/work/theza/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf", num_labels = num_labels,
    torch_dtype = torch.bfloat16, cache_dir = HF_CACHE)

# Prepare data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

def tokenize_ds(ds):
    tokenized_dataset = ds.map(
        lambda example: tokenizer(example["text"], max_length = 60, padding = True, truncation = True), batched = True)

    tokenized_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])
    tokenized_dataset = tokenized_dataset.remove_columns(["text"]).rename_column("label", "labels")
    return tokenized_dataset


train_ds = tokenize_ds(dataset["train"])
test_ds = tokenize_ds(dataset["test"])

if peft_config is not None:
    model = get_peft_model(model, peft_config)
model = model.to("cuda")

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 1,
    warmup_steps = 100,  # max_steps=TRAIN_STEPS,
    learning_rate = 3e-4,  #
    fp16 = False,  #
    logging_steps = 10,  #
    optim = "adamw_torch",  #
    evaluation_strategy = "steps",  #
    save_strategy = "steps",  #
    eval_steps = 5,  #
    save_steps = 5,  #
    output_dir = "../",  #
    save_total_limit = 1, load_best_model_at_end = True,  #
    report_to = ["tensorboard"]  #
)

data_collator = transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of = 8, return_tensors = "pt",
                                                    padding = True)

# Save model
trainer = transformers.Trainer(  #
    model = model,  #
    train_dataset = train_ds,  #
    eval_dataset = test_ds,  #
    args = training_arguments,  #
    data_collator = data_collator  #
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model,
                                                                                                      type(model))
# model = torch.compile(model)
trainer.train()
model.save_pretrained("./")
