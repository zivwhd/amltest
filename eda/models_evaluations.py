import os
from typing import List

import pandas as pd
import torch

from config.config import ExpArgs
from config.constants import INPUT_IDS_NAME, TEXT_PROMPT, LABEL_PROMPT
from config.types_enums import ModelBackboneTypes
from main.utils.baselines_utils import get_model, get_data, get_tokenizer
from utils.utils_functions import get_device


class EvalModel:
    def __init__(self, output_path: str):
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
                if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
                    txt = "\n\n".join([self.task.llama_task_prompt, self.task.llama_few_shots_prompt,
                                       "\n".join([TEXT_PROMPT + txt, LABEL_PROMPT])])

                batch = self.tokenizer([txt], truncation = True, padding = False, return_tensors = "pt").to(self.device)

                if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
                    cls_output_logits = self.model(input_ids = batch[INPUT_IDS_NAME].cuda()).logits[:, -1, :]
                    new_pred = self.tokenizer.batch_decode(torch.argmax(cls_output_logits, dim = -1))[0]
                else:
                    cls_output_logits = self.model(input_ids = batch[INPUT_IDS_NAME].cuda()).logits
                    new_pred = str(torch.argmax(cls_output_logits, dim = -1)[0].item())
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
