import sys

sys.path.append("../..")

import argparse

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, ModelBackboneTypes
from main.main import Baselines
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time

# parser = argparse.ArgumentParser(description = 'Argument parser')
#
# parser.add_argument('task', type = str, help = '')
# parser.add_argument('attribution_score_function', type = str, help = '')
# parser.add_argument('explained_model_backbone', type = str, help = '')
#
# args = parser.parse_args()


arg_task = "imdb"
arg_explained_model_backbone = "BERT"
arg_attribution_score_function = "decompX"

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

is_llama = ExpArgs.task.name == ModelBackboneTypes.LLAMA.value
ExpArgs.ref_token_name = RefTokenNameTypes.UNK.value if is_llama else RefTokenNameTypes.MASK.value

exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"
print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
Baselines(exp_name = exp_path, attr_score_function = arg_attribution_score_function).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
