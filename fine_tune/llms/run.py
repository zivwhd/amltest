import sys

sys.path.append("../..")

import argparse
from fine_tune.llms.fine_tune_llm import LlmFineTune
from runs.runs_utils import get_task
from config.tasks import EMOTION_TASK

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('--task', type = str, help = '')
parser.add_argument('--backbone', type = str, help = '')
parser.add_argument('--is_bf16', type = bool, default = False, help = '')
parser.add_argument('--is_use_prompt', type = bool, default = False, help = '')

args = parser.parse_args()

task = get_task(args.task)
model_backbone = args.backbone
is_bf16 = args.is_bf16
is_use_prompt = args.is_use_prompt

n_epochs = 6 if task.name == EMOTION_TASK.name else 3

LlmFineTune(task = task, is_bf16 = is_bf16, is_use_prompt = is_use_prompt, model_backbone = model_backbone,
            n_epochs = n_epochs).run()
