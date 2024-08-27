import sys

from eda.models_evaluations import EvalModel

sys.path.append("../../..")

import argparse
from config.config import ExpArgs
from runs.runs_utils import get_task

parser = argparse.ArgumentParser(description = 'Argument parser')

arg_task = "sst"
arg_explained_model_backbone = "LLAMA"

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

exp_path = f"EVAL_{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}"
print("*" * 20, arg_task, arg_explained_model_backbone, "*" * 20)
EvalModel(output_path = exp_path).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, "*" * 20)
