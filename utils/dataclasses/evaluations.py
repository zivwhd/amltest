from dataclasses import dataclass

from torch import Tensor


@dataclass
class DataForEvalInputs:
    input_ids: Tensor
    attention_mask: Tensor
    task_prompt_input_ids: Tensor
    label_prompt_input_ids: Tensor
    task_prompt_attention_mask: Tensor
    label_prompt_attention_mask: Tensor


@dataclass
class DataForEval:
    tokens_attr: Tensor
    input: DataForEvalInputs
    gt_target: Tensor
    pred_origin: Tensor
    pred_origin_logits: Tensor
    k: float = 0.0
