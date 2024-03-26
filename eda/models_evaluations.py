import os
from typing import List

import pandas as pd
import torch

from config.config import ExpArgs
from config.constants import TEXT_PROMPT, LABEL_PROMPT_NEW_LINE
from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from utils.utils_functions import get_device, is_model_encoder_only, merge_prompts, is_use_prompt


class EvalModel:
    def __init__(self, output_path: str):
        init_baseline_exp()
        self.task = ExpArgs.task
        os.makedirs(output_path, exist_ok = True)
        self.output_path = f"{output_path}/eval.csv"
        self.model, self.model_path = get_model()
        self.tokenizer = get_tokenizer(self.model_path)
        self.data = get_data()

        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preds: List[int] = []
        self.labels: List[int] = []

        if is_use_prompt():
            labels_tokens = [self.tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False) for l in
                             list(ExpArgs.task.labels_int_str_maps.keys())]
            ExpArgs.labels_tokens_opt = torch.stack(labels_tokens).squeeze()[:, -1]

    def run(self):
        with torch.no_grad():
            for i, row in enumerate(self.data):
                item_idx = row[2]
                label = row[1]
                txt = row[0]
                self.labels += [str(label)]

                tokenized_input = self.tokenizer.encode_plus(txt,
                                                             truncation = True, padding = False,
                                                             return_tensors = "pt",
                                                             add_special_tokens = not is_use_prompt()).to(self.device)
                input_ids = tokenized_input.input_ids
                attention_mask = tokenized_input.attention_mask

                if is_use_prompt():
                    task_prompt = "\n\n".join(
                        [self.task.llm_task_prompt, self.task.llm_few_shots_prompt, "\n".join([TEXT_PROMPT])])
                    tokenized_task_prompt = self.tokenizer.encode_plus(task_prompt, padding = False,
                                                                       return_tensors = "pt",
                                                                       add_special_tokens = True).to(self.device)
                    tokenized_label = self.tokenizer.encode_plus(LABEL_PROMPT_NEW_LINE, padding = False,
                                                                 return_tensors = "pt", add_special_tokens = False).to(
                        self.device)
                    input_ids, attention_mask = merge_prompts(  #
                        inputs = input_ids, attention_mask = attention_mask,
                        task_prompt_input_ids = tokenized_task_prompt.input_ids,
                        label_prompt_input_ids = tokenized_label.input_ids,
                        task_prompt_attention_mask = tokenized_task_prompt.attention_mask,
                        label_prompt_attention_mask = tokenized_label.attention_mask)

                if is_model_encoder_only() or self.task.is_llm_use_lora:
                    logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits.squeeze()
                    self.preds.append(str(torch.argmax(logits, dim = -1).squeeze().item()))
                else:
                    logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits[:, -1, :]
                    mask = torch.ones_like(logits, dtype = torch.bool)
                    mask[:, ExpArgs.labels_tokens_opt] = False
                    logits[mask] = float('-inf')

                    new_pred = self.tokenizer.batch_decode(torch.argmax(logits, dim = -1))[0]
                    self.preds.append(new_pred)

        err = sum(1 for x, y in zip(self.preds, self.labels) if x != y) / len(self.preds)
        acc = 1 - err
        df = pd.DataFrame(dict(  #
            model = [ExpArgs.explained_model_backbone],  #
            task = self.task.name,  #
            metric = ["Accuracy"],  #
            pred = [self.preds],  #
            labels = [self.labels],  #
            accuracy = [f"{acc * 100:.3f}%"]  #
        ))
        with open(self.output_path, 'a', newline = '', encoding = 'utf-8-sig') as f:
            df.to_csv(f, header = f.tell() == 0, index = False)
