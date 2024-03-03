import pandas as pd

from config.config import ExpArgs
from evaluations.metrics_aopc.metrics2 import Metrics2
from utils.dataclasses.metricResults import MetricResults


class Metrics2Sequel(Metrics2):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def map_result(self, metric_result):
        gt_target = self.outputs[0].gt_target
        if type(gt_target) == list:
            if len(gt_target) != 1:
                raise ValueError(f"update_results_df")
            gt_target = gt_target[0]
        gt_target = gt_target.item()
        res = MetricResults(attr_score_function = self.stage, item_index = self.item_index, task = ExpArgs.task.name,
                            eval_metric = ExpArgs.eval_metric,
                            explained_model_backbone = ExpArgs.explained_model_backbone, metric_result = metric_result,
                            metric_steps_result = None, steps_k = self.pretu_steps,
                            model_pred_origin = self.outputs[0].pred_origin.squeeze().item(),
                            gt_target = gt_target)
        return pd.DataFrame([res])

