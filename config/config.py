from config.types_enums import *
from utils.dataclasses import Task


class ExpArgs:
    seed = 42
    default_root_dir = "OUT"
    ref_token_name: RefTokenNameTypes = None
    evaluation_metric: str = None
    is_save_support_results = True
    is_save_results = True
    task: Task = None
    explained_model_backbone: str = None
    attribution_scores_function: str = None
    label_vocab_tokens = None
    token_evaluation_option = TokenEvaluationOptions.NO_SPECIAL_TOKENS.value
    is_evaluate = True
    is_save_times = False
    is_save_words = False



class MetricsMetaData:
    directions = {EvalMetric.SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.EVAL_LOG_ODDS.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS.value: DirectionTypes.MAX.value}

    top_k = {EvalMetric.SUFFICIENCY.value: [20], EvalMetric.COMPREHENSIVENESS.value: [20],
             EvalMetric.EVAL_LOG_ODDS.value: [20], EvalMetric.AOPC_SUFFICIENCY.value: [1, 5, 10, 20, 50],
             EvalMetric.AOPC_COMPREHENSIVENESS.value: [1, 5, 10, 20, 50]}


class BackbonesMetaData:
    name = {  #
        ModelBackboneTypes.BERT.value: "bert",  #
        ModelBackboneTypes.ROBERTA.value: "roberta",  #
        ModelBackboneTypes.DISTILBERT.value: "distilbert",  #
        ModelBackboneTypes.LLAMA.value: "model",  #
        ModelBackboneTypes.MISTRAL.value: "model"  #
    }
