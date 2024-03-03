import torch

from config.types_enums import *
from utils.dataclasses import Task


class ExpArgs:
    seed = 42
    default_root_dir = "OUT"
    ref_token_name: RefTokenNameTypes = RefTokenNameTypes.MASK.value
    verbose = True
    eval_metric: str = EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value
    is_save_results = True
    task: Task = None
    explained_model_backbone: str = ModelBackboneTypes.BERT.value
    validation_type = ValidationType.VAL.value
    labels_tokens_opt = None
    llama_f16 = True


ExpArgsDefault = type('ClonedExpArgs', (), vars(ExpArgs).copy())


class MetricsMetaData:
    directions = {EvalMetric.SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.EVAL_LOG_ODDS.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value: DirectionTypes.MIN.value,
                  EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value: DirectionTypes.MAX.value}

    top_k = {EvalMetric.SUFFICIENCY.value: [20], EvalMetric.COMPREHENSIVENESS.value: [20],
             EvalMetric.EVAL_LOG_ODDS.value: [20], EvalMetric.AOPC_SUFFICIENCY.value: torch.arange(20, 100, 20),
             EvalMetric.AOPC_COMPREHENSIVENESS.value: torch.arange(20, 100, 20),
             EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value: torch.arange(20, 100, 20),
             EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value: torch.arange(20, 100, 20)}


class BackbonesMetaData:
    name = {  #
        ModelBackboneTypes.BERT.value: "bert",  #
        ModelBackboneTypes.ROBERTA.value: "roberta",  #
        ModelBackboneTypes.LLAMA.value: "model"  #
    }
