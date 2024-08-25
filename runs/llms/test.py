import sys

sys.path.append("../..")

import argparse
from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, EvalMetric
from main.main import Baselines
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time, is_model_encoder_only


arg_task = "rtn"
arg_explained_model_backbone = "MISTRAL"
arg_attribution_score_function = "alti"
arg_metric = [EvalMetric.SUFFICIENCY]

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value

exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"

ExpArgs.task.is_llm_set_max_len=True
ExpArgs.task.llm_explained_tokenizer_max_length = 30
ExpArgs.task.llm_task_prompt = "do one of"
ExpArgs.task.llm_few_shots_prompt = "a:1, b:2, c:3"

print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
Baselines(exp_name = exp_path, attr_score_function = arg_attribution_score_function, metrics = arg_metric).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
