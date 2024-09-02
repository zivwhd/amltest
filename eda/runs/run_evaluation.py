import sys

from eda.models_evaluations import EvalModel

sys.path.append("../../..")

import argparse
from config.config import ExpArgs
from runs.runs_utils import get_task

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('task', type = str, help = '')
parser.add_argument('explained_model_backbone', type = str, help = '')
parser.add_argument('is_use_label_start_token', type = str, help = '')

args = parser.parse_args()

arg_task = args.task
arg_explained_model_backbone = args.explained_model_backbone
arg_is_use_label_start_token = args.is_use_label_start_token

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone
ExpArgs.is_use_label_start_token = arg_is_use_label_start_token == "1"

exp_path = f"EVAL_{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{ExpArgs.is_use_label_start_token}"
print("*" * 20, arg_task, arg_explained_model_backbone, "*" * 20)
EvalModel(output_path = exp_path).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, "*" * 20)
