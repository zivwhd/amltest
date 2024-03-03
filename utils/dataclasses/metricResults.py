from dataclasses import dataclass
from typing import Union


@dataclass
class MetricResults:
    attr_score_function: str
    item_index: int
    task: str
    eval_metric: str
    explained_model_backbone: str
    metric_result: float
    metric_steps_result: any
    steps_k: any
    model_pred_origin: Union[int, None]
    gt_target: Union[int, None]
