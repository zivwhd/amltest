import time
from typing import Tuple

import torch
from torch import Tensor

from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes


def conv_class_to_dict(item):
    obj = {}
    for key in item.keys():
        obj[key] = item[key]
    return obj


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_model_encoder_only(model = None):
    if model is None:
        model = ExpArgs.explained_model_backbone
    if model in [ModelBackboneTypes.LLAMA.value, ModelBackboneTypes.MISTRAL.value]:
        return False
    if model in [ModelBackboneTypes.ROBERTA.value, ModelBackboneTypes.BERT.value, ModelBackboneTypes.DISTILBERT.value]:
        return True

    raise ValueError(f"unsupported model: {model}")


def calculate_num_of_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_num_of_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def calculate_percentage_of_trainable_params(model) -> str:
    return f"{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%"


def print_number_of_trainable_and_not_trainable_params(model) -> None:
    print(
        f"Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}")


def get_current_time():
    return int(round(time.time()))


def run_model(model, input_ids: Tensor = None, attention_mask: Tensor = None, inputs_embeds: Tensor = None,
              is_return_logits: bool = False, is_return_output: bool = False):
    if is_model_encoder_only():
        model_output = model(input_ids = input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds)
        logits = model_output.logits
    else:
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.half()
            model_output = model(inputs_embeds = inputs_embeds).logits[:, -1, :]
        else:
            model_output = model(input_ids = input_ids).logits[:, -1, :]
        logits = model_output[:, ExpArgs.labels_tokens_opt]

    if is_return_logits and is_return_output:
        return model_output, logits

    elif is_return_logits:
        return logits

    elif is_return_output:
        return model_output

    else:
        raise ValueError(f"must choose model output option")


# works for input_ids / embeddings

def merge_prompts(inputs, attention_mask, task_prompt_input_ids: Tensor = None, label_prompt_input_ids: Tensor = None,
                  task_prompt_attention_mask: Tensor = None, label_prompt_attention_mask: Tensor = None) -> Tuple[
    Tensor, Tensor]:
    if is_model_encoder_only():
        return inputs, attention_mask

    if any(item is None for item in [task_prompt_input_ids, inputs, label_prompt_input_ids]):
        raise ValueError("can not be None")
    merged_inputs = torch.cat([task_prompt_input_ids, inputs, label_prompt_input_ids], dim = 1)
    merged_attention_mask = torch.cat([task_prompt_attention_mask, attention_mask, label_prompt_attention_mask],
                                      dim = 1)

    return merged_inputs, merged_attention_mask
