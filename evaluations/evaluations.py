import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, EvalMetric
from evaluations.metrics.metrics import Metrics
from utils.dataclasses.evaluations import DataForEval


def evaluate_tokens_attr(model, explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEval,
                         experiment_path: str, item_index: str):
    with torch.no_grad():
        if ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                   EvalMetric.EVAL_LOG_ODDS.value, EvalMetric.AOPC_SUFFICIENCY.value,
                                   EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            eval_class = Metrics(model = model, explained_tokenizer = explained_tokenizer, ref_token_id = ref_token_id,
                                 data = data, experiment_path = experiment_path,
                                 item_index = item_index)
            return eval_class.run_perturbation_test()
        else:
            raise ValueError("unsupported ExpArgs.eval_metric selected")
