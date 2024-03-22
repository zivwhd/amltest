import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaForSequenceClassification, LlamaTokenizer

from config.constants import HF_CACHE, LOCAL_MODELS_PREFIX, TEXT_PROMPT, LABEL_PROMPT
from utils.dataclasses import Task

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)


class LlmFineTune:

    def __init__(self, task: Task, is_bf16: bool, is_use_prompt: bool):
        self.output_dir = f"is_bf16_{is_bf16}_is_use_prompt_{is_use_prompt}"
        self.task = task
        self.is_bf16 = is_bf16
        self.is_use_prompt = is_use_prompt

        # load data
        dataset = load_dataset(task.dataset_name)
        self.num_labels = len(set(dataset["train"][task.dataset_column_label]))

        # Prepare model
        self.llama_model = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf"
        self.peft_config = LoraConfig(  #
            task_type = TaskType.SEQ_CLS,  #
            r = 8,  #
            lora_alpha = 16,  #
            lora_dropout = 0.1,  #
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.model, self.tokenizer = None, None
        self.load_model_and_tokenizer()
        self.train_ds = self.tokenize_ds(dataset["train"].select(range(20)))
        self.test_ds = self.tokenize_ds(dataset["test"].select(range(10)))

    def load_model_and_tokenizer(self):
        if self.is_bf16:
            self.model = LlamaForSequenceClassification.from_pretrained(self.llama_model, num_labels = self.num_labels,
                                                                        torch_dtype = torch.bfloat16,
                                                                        cache_dir = HF_CACHE)
        else:
            self.model = LlamaForSequenceClassification.from_pretrained(self.llama_model, num_labels = self.num_labels,
                                                                        cache_dir = HF_CACHE)

        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_model)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def merge_prompt(self, example):
        txt = example[self.task.dataset_column_text]
        if self.is_use_prompt:
            txt = "\n\n".join([self.task.llm_task_prompt, "\n".join([TEXT_PROMPT + txt, LABEL_PROMPT])])

        return txt

    def tokenize_ds(self, ds):
        tokenized_dataset = ds.map(
            lambda example: self.tokenizer(self.merge_prompt(example), max_length = 100, truncation = True),
            batched = False)
        tokenized_dataset = tokenized_dataset.rename_column(self.task.dataset_column_label, "labels")
        tokenized_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
        tokenized_dataset = tokenized_dataset.remove_columns([self.task.dataset_column_text])
        return tokenized_dataset

    def run(self):
        self.model = get_peft_model(self.model, self.peft_config)
        self.model = self.model.to("cuda")

        training_arguments = transformers.TrainingArguments(  #
            per_device_train_batch_size = 5,  #
            per_device_eval_batch_size = 1,  #
            warmup_steps = 100, learning_rate = 3e-4,  #
            logging_steps = 10,  #
            optim = "adamw_torch",  #
            evaluation_strategy = "steps",  #
            save_strategy = "steps",  #
            eval_steps = 5,  #
            save_steps = 5,  #
            logging_dir = f"{self.output_dir}/logs",  #
            output_dir = f"{self.output_dir}/results",  #
            save_total_limit = 1, load_best_model_at_end = True,  #
            report_to = ["tensorboard"]  #
        )

        data_collator = transformers.DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of = 8,
                                                             return_tensors = "pt", padding = True)

        # Save model
        trainer = transformers.Trainer(  #
            model = self.model,  #
            train_dataset = self.train_ds,  #
            eval_dataset = self.test_ds,  #
            args = training_arguments,  #
            data_collator = data_collator,  #
            compute_metrics = compute_metrics)

        trainer.train()
        self.model.save_pretrained(self.output_dir)
