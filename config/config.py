import torch

from config.types_enums import *
from utils.dataclasses import Task


class ExpArgs:
    seed = 42
    default_root_dir = "OUT"
    ref_token_name: RefTokenNameTypes = RefTokenNameTypes.MASK.value
    verbose = True
    eval_metric: str = None
    is_save_support_results = True
    is_save_results = True
    task: Task = None
    explained_model_backbone: str = None
    validation_type = ValidationType.VAL.value
    labels_tokens_opt = None
    eval_tokens = None


ExpArgsDefault = type('ClonedExpArgs', (), vars(ExpArgs).copy())


class MetricsMetaData:
    directions = {EvalMetric.SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.EVAL_LOG_ODDS.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS.value: DirectionTypes.MAX.value}

    top_k = {EvalMetric.SUFFICIENCY.value: [20], EvalMetric.COMPREHENSIVENESS.value: [20],
             EvalMetric.EVAL_LOG_ODDS.value: [20], EvalMetric.AOPC_SUFFICIENCY.value: torch.arange(20, 100, 20).tolist(),
             EvalMetric.AOPC_COMPREHENSIVENESS.value: torch.arange(20, 100, 20).tolist()}


class BackbonesMetaData:
    name = {  #
        ModelBackboneTypes.BERT.value: "bert",  #
        ModelBackboneTypes.ROBERTA.value: "roberta",  #
        ModelBackboneTypes.DISTILBERT.value: "distilbert",  #
        ModelBackboneTypes.LLAMA.value: "model",  #
        ModelBackboneTypes.MISTRAL.value: "model"  #
    }
