import os

from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

import torch
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments)

print(f"pytorch version {torch.__version__}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

model_name = "/home/yonatanto/work/theza/LOCAL_MODELS/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf"
compute_dtype = getattr(torch, "float16")

model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device, torch_dtype = compute_dtype)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

output_dir = "trained_weigths"

peft_config = LoraConfig(lora_alpha = 16, lora_dropout = 0.1, r = 64, bias = "none", target_modules = "all-linear",
                         task_type = "CAUSAL_LM", )

training_arguments = TrainingArguments(output_dir = output_dir,  # directory to save and repository id
                                       num_train_epochs = 3,  # number of training epochs
                                       per_device_train_batch_size = 1,  # batch size per device during training
                                       gradient_accumulation_steps = 8,
                                       # number of steps before performing a backward/update pass
                                       gradient_checkpointing = True,  # use gradient checkpointing to save memory
                                       optim = "paged_adamw_32bit", save_steps = 0, logging_steps = 25,
                                       # log every 10 steps
                                       learning_rate = 2e-4,  # learning rate, based on QLoRA paper
                                       weight_decay = 0.001, fp16 = True, bf16 = False, max_grad_norm = 0.3,
                                       # max gradient norm based on QLoRA paper
                                       max_steps = -1, warmup_ratio = 0.03,  # warmup ratio based on QLoRA paper
                                       group_by_length = True, lr_scheduler_type = "cosine",
                                       # use cosine learning rate scheduler
                                       report_to = "tensorboard",  # report metrics to tensorboard
                                       evaluation_strategy = "epoch"  # save checkpoint every epoch
                                       )

dataset = load_dataset("sst2")


def preprocess_function(examples):
    examples["text"] = "select sentiment" + examples["sentence"]
    return examples


remove_cols = ['sentence', 'attention_mask']
train_dataset = dataset["train"].select(range(10)).map(preprocess_function, batched = False).remove_columns(remove_cols)

test_dataset = dataset["test"].select(range(5)).map(preprocess_function, batched = False).remove_columns(remove_cols)

trainer = SFTTrainer(model = model, args = training_arguments, train_dataset = train_dataset,
                     eval_dataset = test_dataset, peft_config = peft_config, dataset_text_field = "text",
                     tokenizer = tokenizer, max_seq_length = 200, packing = False,
                     dataset_kwargs = {"add_special_tokens": False, "append_concat_token": False, })

trainer.train()

trainer.save_model()
tokenizer.save_pretrained(output_dir)
