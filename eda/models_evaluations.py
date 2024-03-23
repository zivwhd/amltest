import os
from typing import List

import pandas as pd
import torch

from config.config import ExpArgs
from config.constants import TEXT_PROMPT, LABEL_PROMPT
from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from utils.utils_functions import get_device, is_model_encoder_only, run_model


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

    def run(self):
        with torch.no_grad():
            for i, row in enumerate(self.data):
                item_idx = row[2]
                label = row[1]
                txt = row[0]
                self.labels += [str(label)]
                if not is_model_encoder_only():
                    txt = "\n\n".join([self.task.llm_task_prompt, self.task.llm_few_shots_prompt,
                                       "\n".join([TEXT_PROMPT + txt, LABEL_PROMPT])])

                batch = self.tokenizer([txt], truncation = True, padding = False, return_tensors = "pt").to(self.device)

                logits = run_model(model = self.model, input_ids = batch.inputs_ids.cuda(),
                                             attention_mask = batch.attention_mask.cuda(), is_return_logits = True).squeeze()

                new_pred = self.tokenizer.batch_decode(torch.argmax(logits, dim = -1))[0]
                self.preds.append(new_pred)

        err = sum(1 for x, y in zip(self.preds, self.labels) if x != y) / len(self.preds)
        acc = 1 - err
        df = pd.DataFrame(dict(  #
            model = [ExpArgs.explained_model_backbone],  #
            task = self.task.name, llama_float16 = [ExpArgs.llama_f16],  #
            metric = ["Accuracy"],  #
            pred = [self.preds],  #
            labels = [self.labels],  #
            accuracy = [f"{acc * 100:.3f}%"]  #
        ))
        with open(self.output_path, 'a', newline = '', encoding = 'utf-8-sig') as f:
            df.to_csv(f, header = f.tell() == 0, index = False)
