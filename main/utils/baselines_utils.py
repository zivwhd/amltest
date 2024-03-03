import gc

import torch
from datasets import load_dataset
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer

from config.config import ExpArgs
from config.constants import HF_CACHE
from config.types_enums import ModelBackboneTypes
from utils.dataclasses import Task


def init_baseline_exp():
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(ExpArgs.seed)


def create_model_class(model_type):
    class ModelWithUnpackedForward(model_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            out = out[0]  # no longer a tuple
            return out

        def orig_forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    return ModelWithUnpackedForward


def get_model():
    task = ExpArgs.task
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
        from transformers import BertForSequenceClassification
        model_path = task.bert_fine_tuned_model
        model = BertForSequenceClassification.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
        from transformers import RobertaForSequenceClassification
        model_path = task.roberta_fine_tuned_model
        model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        from transformers import DistilBertForSequenceClassification
        model_path = task.distilbert_fine_tuned_model
        model = DistilBertForSequenceClassification.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        from transformers import LlamaForCausalLM
        model_path = task.llama_model
        if ExpArgs.llama_f16:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, cache_dir = HF_CACHE)
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, cache_dir = HF_CACHE)

    else:
        raise ValueError("unsupported explained_model_backbone selected")
    return model, model_path


def get_tokenizer(model_path: str):
    task = ExpArgs.task
    if task.llama_max_length and (ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value):
        return AutoTokenizer.from_pretrained(model_path, max_length = task.llama_max_length)
    return AutoTokenizer.from_pretrained(model_path)


def get_data():
    task = ExpArgs.task
    ds = load_dataset(task.dataset_name)[task.dataset_test].shuffle(seed = ExpArgs.seed)
    if task.test_sample:
        ds = ds.train_test_split(train_size = task.test_sample, seed = ExpArgs.seed,
                                 stratify_by_column = task.dataset_column_label)
        ds = ds["train"]
    idx_col = ds["idx"] if "idx" in ds.features.keys() else list(range(len(ds[task.dataset_column_label])))
    ds = list(zip(ds[task.dataset_column_text], ds[task.dataset_column_label], idx_col))
    return ds


def tokenize(tokenizer, model, txt, padding_type = True):
    return tokenizer(txt, truncation = True, padding = False, return_tensors = "pt").to(model.device)
