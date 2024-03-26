import os
from functools import partial

import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaForSequenceClassification, AutoTokenizer, MistralForSequenceClassification

from config.constants import HF_CACHE, LOCAL_MODELS_PREFIX, TEXT_PROMPT, LABEL_PROMPT
from config.types_enums import ModelBackboneTypes
from main.utils.baselines_utils import init_baseline_exp
from utils.dataclasses import Task

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)


def merge_prompt(example, tokenizer, task, is_use_prompt):
    txt = example[task.dataset_column_text]
    if is_use_prompt:
        txt = "\n\n".join([task.llm_task_prompt, "\n".join([TEXT_PROMPT + txt, LABEL_PROMPT])])
    tokenized = tokenizer.encode_plus(txt, truncation = True)
    example[task.dataset_column_text] = txt
    example["input_ids"] = tokenized.input_ids
    example["attention_mask"] = tokenized.attention_mask
    return example


class LlmFineTune:

    def __init__(self, task: Task, is_bf16: bool, is_use_prompt: bool, model_backbone: str, n_epochs: int):
        init_baseline_exp()
        self.output_dir = f"OUTPUT/{model_backbone}_{task.name}_is_bf16_{is_bf16}_is_use_prompt_{is_use_prompt}"
        os.makedirs(self.output_dir, exist_ok = True)
        self.task = task
        self.is_bf16 = is_bf16
        self.is_use_prompt = is_use_prompt
        self.model_backbone = model_backbone
        self.n_epochs = n_epochs

        # load data
        dataset = load_dataset(task.dataset_name)
        self.num_labels = len(set(dataset["train"][task.dataset_column_label]))

        if self.model_backbone == ModelBackboneTypes.LLAMA.value:
            self.model_path = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf"
        elif self.model_backbone == ModelBackboneTypes.MISTRAL.value:
            self.model_path = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1"
        else:
            raise ValueError("unsupported model_backbone")

        self.peft_config = LoraConfig(  #
            task_type = TaskType.SEQ_CLS,  #
            r = 8,  #
            lora_alpha = 16,  #
            lora_dropout = 0.1,  #
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.model, self.tokenizer = None, None
        self.load_model_and_tokenizer()
        self.train_ds = self.tokenize_ds(dataset["train"].shuffle(seed = 42))
        self.test_ds = self.tokenize_ds(dataset["test"])

    def load_model_and_tokenizer(self):
        if self.model_backbone == ModelBackboneTypes.LLAMA.value:
            model_class = LlamaForSequenceClassification
        elif self.model_backbone == ModelBackboneTypes.MISTRAL.value:
            model_class = MistralForSequenceClassification
        else:
            raise ValueError("unsupported model_backbone")

        if self.is_bf16:
            self.model = model_class.from_pretrained(self.model_path, num_labels = self.num_labels,
                                                     torch_dtype = torch.bfloat16, cache_dir = HF_CACHE)
        else:
            self.model = model_class.from_pretrained(self.model_path, num_labels = self.num_labels,
                                                     cache_dir = HF_CACHE)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def tokenize_ds(self, ds):
        modify_text_partial = partial(merge_prompt, tokenizer = self.tokenizer, task = self.task,
                                      is_use_prompt = self.is_use_prompt)
        tokenized_dataset = ds.map(modify_text_partial, batched = False)
        tokenized_dataset = tokenized_dataset.rename_column(self.task.dataset_column_label, "labels")
        tokenized_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
        tokenized_dataset = tokenized_dataset.remove_columns([self.task.dataset_column_text])
        return tokenized_dataset

    def run(self):
        self.model = get_peft_model(self.model, self.peft_config)
        self.model = self.model.to("cuda")

        training_arguments = transformers.TrainingArguments(  #
            num_train_epochs = self.n_epochs, per_device_train_batch_size = 5,  #
            per_device_eval_batch_size = 1,  #
            gradient_accumulation_steps = 4,  #
            warmup_ratio = 0.05,  #
            learning_rate = 4e-5,  #
            logging_steps = 20,  #
            optim = "adamw_torch",  #
            evaluation_strategy = "steps",  #
            save_strategy = "steps",  #
            eval_steps = 500,  #
            save_steps = 500,  #
            logging_dir = f"{self.output_dir}/logs",  #
            output_dir = f"{self.output_dir}/results",  #
            save_total_limit = 1,  #
            load_best_model_at_end = True,  #
            report_to = ["tensorboard"]  #
        )

        data_collator = transformers.DataCollatorWithPadding(self.tokenizer, return_tensors = "pt", padding = True)

        # Save model
        trainer = transformers.Trainer(  #
            model = self.model,  #
            train_dataset = self.train_ds,  #
            eval_dataset = self.test_ds,  #
            args = training_arguments,  #
            data_collator = data_collator,  #
            compute_metrics = compute_metrics)

        trainer.train()

        self.model.save_pretrained(f"{self.output_dir}/best_model", save_adapter = True, save_config = True)
