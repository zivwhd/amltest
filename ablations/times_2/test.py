import argparse
import sys

sys.path.append("../..")

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes
from main.main import Baselines
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time, is_model_encoder_only



arg_task = "emotions"
arg_explained_model_backbone = "BERT"
arg_attribution_score_function = "alti"
args_test_sample = int(10)
arg_metric = []

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone
ExpArgs.task.test_sample = args_test_sample

ExpArgs.is_evaluate = False
ExpArgs.is_save_times = True

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value
exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"
print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
Baselines(exp_name = exp_path, attr_score_function = arg_attribution_score_function, metrics = arg_metric).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
