import sys

sys.path.append("../../..")

from ablations.reasoning.eraser_datasets.eraser_utils import load_jsonl
from ablations.reasoning.run_attr_scores.main_reasoning import BaselinesReasoning

import argparse
from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, EvalMetric
from runs.runs_utils import get_task
from utils.utils_functions import is_model_encoder_only

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('task', type = str, help = '')
parser.add_argument('attribution_score_function', type = str, help = '')
parser.add_argument('explained_model_backbone', type = str, help = '')
parser.add_argument('metric', type = str, help = '')
parser.add_argument('solvex_beam_size', type = str, help = '')

args = parser.parse_args()

arg_task = args.task
arg_explained_model_backbone = args.explained_model_backbone
arg_attribution_score_function = args.attribution_score_function
arg_metric = EvalMetric[args.metric].value
arg_solvex_beam_size = args.solvex_beam_size
arg_set_type = "long"



ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value
ExpArgs.BEAM_SIZE = int(arg_solvex_beam_size)
ExpArgs.evaluation_metric = arg_metric

import pandas as pd

from datasets import load_dataset

data = load_dataset(ExpArgs.task.dataset_name)[ExpArgs.task.dataset_test]
df = pd.DataFrame(dict(txt = data[ExpArgs.task.dataset_column_text]))
df["n_words"] = df.txt.apply(lambda x: len(x.split()))
longs = df[df["n_words"] > 30]
longs = longs[:200]
longs = longs.reset_index(drop = True)

print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
BaselinesReasoning(attr_score_function = arg_attribution_score_function).run(list(longs.txt))
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
