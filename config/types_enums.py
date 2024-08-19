from enum import Enum


class ModelBackboneTypes(Enum):
    BERT = 'BERT'
    ROBERTA = 'ROBERTA'
    DISTILBERT = 'DISTILBERT'
    LLAMA = 'LLAMA'
    MISTRAL = 'MISTRAL'


class DefaultEvalMetric(Enum):
    SUFFICIENCY = 'SUFFICIENCY'
    COMPREHENSIVENESS = 'COMPREHENSIVENESS'
    EVAL_LOG_ODDS = 'EVAL_LOG_ODDS'
    AOPC_SUFFICIENCY = 'AOPC_SUFFICIENCY'
    AOPC_COMPREHENSIVENESS = 'AOPC_COMPREHENSIVENESS'

class EvalMetric(Enum):
    SUFFICIENCY = 'SUFFICIENCY'
    COMPREHENSIVENESS = 'COMPREHENSIVENESS'
    EVAL_LOG_ODDS = 'EVAL_LOG_ODDS'
    AOPC_SUFFICIENCY = 'AOPC_SUFFICIENCY'
    AOPC_COMPREHENSIVENESS = 'AOPC_COMPREHENSIVENESS'
    # COMPREHENSIVENESS_SUFFICIENCY = 'COMPREHENSIVENESS_SUFFICIENCY'


class DirectionTypes(Enum):
    MAX = 'MAX'
    MIN = 'MIN'


class RefTokenNameTypes(Enum):
    MASK = 'MASK'
    PAD = 'PAD'
    UNK = 'UNK'


class ValidationType(Enum):
    VAL = 'VAL'
    TEST = 'TEST'


class ModelPromptType(Enum):
    ZERO_SHOT = 'zero_shot'
    FEW_SHOT = 'few_shot'
    FEW_SHOT_CONTENT = 'few_shot_content'  # few-shot but map just content


class AttrScoreFunctions(Enum):
    deep_lift = 'deep_lift'
    gradient_shap = 'gradient_shap'
    lime = 'lime'
    input_x_gradient = 'input_x_gradient'
    integrated_gradients = 'integrated_gradients'
    sequential_integrated_gradients = 'sequential_integrated_gradients'
    decompX = 'decompX'
    decompX_class = 'decompX_class'
    alti = 'alti'
    glob_enc = 'glob_enc'
    glob_enc_dim_0 = 'glob_enc_dim_0'
    llm = "llm"
    solvability = "solvability"


class EvalTokens(Enum):
    # ALL_TOKENS = 'ALL_TOKENS'
    # NO_CLS = 'NO_CLS'
    NO_SPECIAL_TOKENS = 'NO_SPECIAL_TOKENS'
