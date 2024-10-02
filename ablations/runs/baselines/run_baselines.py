import sys

sys.path.append("../..")

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, ModelBackboneTypes, AttrScoreFunctions, DefaultEvalMetric
from ablations.rpi.main_rpi import BaselinesRpi
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time, is_model_encoder_only

import argparse

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('task', type = str, help = '')
parser.add_argument('attribution_score_function', type = str, help = '')
parser.add_argument('explained_model_backbone', type = str, help = '')
parser.add_argument('metric', type = str, help = '')

args = parser.parse_args()

arg_task = args.task
arg_explained_model_backbone = args.explained_model_backbone
arg_attribution_score_function = args.attribution_score_function
arg_metric = DefaultEvalMetric if args.metric == "all" else [DefaultEvalMetric[args.metric]]

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value

exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"
print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
BaselinesRpi(exp_name = exp_path, attr_score_function = arg_attribution_score_function, metrics = arg_metric).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
