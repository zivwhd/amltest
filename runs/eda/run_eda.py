import sys

sys.path.append("../..")

import argparse

from config.config import ExpArgs
from eda.models_evaluations import EvalModel
from runs.runs_utils import get_task

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('task', type = str, help = '')
parser.add_argument('explained_model_backbone', type = str, help = '')

args = parser.parse_args()

ExpArgs.task = get_task(args.task)
ExpArgs.explained_model_backbone = args.explained_model_backbone

exp_path = ExpArgs.default_root_dir
print("*" * 20, args.task, args.explained_model_backbone, "*" * 20)
EvalModel(output_path = exp_path).run()
print("*" * 20, "END OF ", args.task, args.explained_model_backbone, "*" * 20)
