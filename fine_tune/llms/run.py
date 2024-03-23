import sys

sys.path.append("../..")

import argparse
from fine_tune.llms.fine_tune_llama import LlmFineTune
from runs.runs_utils import get_task

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('--task', type = str, help = '')
parser.add_argument('--is_bf16', type = bool, default = False, help = '')
parser.add_argument('--is_use_prompt', type = bool, default = False, help = '')

args = parser.parse_args()

task = get_task(args.task)
is_bf16 = args.is_bf16
is_use_prompt = args.is_use_prompt

LlmFineTune(task = task, is_bf16 = is_bf16, is_use_prompt = is_use_prompt).run()
